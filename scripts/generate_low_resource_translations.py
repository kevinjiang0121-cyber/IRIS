"""
Translate the matched behavior set into a low-resource target language.

Used by ``baselines/low_resource_lang`` to seed the cross-interface evaluation
Mirrors yong2023low's setup: literal Google-Translate rendering
of each English behavior label into the target language, with no decode
hint at inference time. Google Translate (via ``deep_translator``) is
chosen over LLM translation to (a) match yong2023low's original setup
and (b) avoid LLM-translator alignment refusals confounding the pilot.

Output: ``outputs/assets/low_resource_translations_<lang>.json``
    { "<behavior_id>": "<target-language rendering>", ... }

Usage::

    PYTHONUTF8=1 python scripts/generate_low_resource_translations.py \\
        --target_language zulu
    PYTHONUTF8=1 python scripts/generate_low_resource_translations.py \\
        --target_language yoruba
"""

import argparse
import csv
import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from deep_translator import GoogleTranslator  # noqa: E402

DEFAULT_BEHAVIORS_CSV = os.path.join(
    REPO_ROOT, "data", "behavior_datasets", "final_runnable_with_abc_judge.csv"
)
DEFAULT_PILOT_IDS = os.path.join(REPO_ROOT, "data", "full105", "behavior_ids_full105.txt")

GOOGLE_LANG_CODES = {
    "zulu": "zu",
    "yoruba": "yo",
}


def load_pilot_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_behaviors(csv_path, ids_subset):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row.get("BehaviorID")
            if bid and bid in ids_subset:
                rows.append(row)
    found_ids = {r["BehaviorID"] for r in rows}
    missing = [bid for bid in ids_subset if bid not in found_ids]
    return rows, missing


def ascii_ratio(text):
    if not text:
        return 1.0
    n = len(text)
    a = sum(1 for ch in text if ord(ch) < 128)
    return a / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_language",
        required=True,
        choices=list(GOOGLE_LANG_CODES.keys()),
    )
    parser.add_argument("--behaviors_path", default=DEFAULT_BEHAVIORS_CSV)
    parser.add_argument("--pilot_ids_path", default=DEFAULT_PILOT_IDS)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--failed_path", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--sleep_between", type=float, default=0.3)
    args = parser.parse_args()

    lang = args.target_language
    code = GOOGLE_LANG_CODES[lang]
    output_path = args.output_path or os.path.join(
        REPO_ROOT, "outputs", "assets", f"low_resource_translations_{lang}.json"
    )
    failed_path = args.failed_path or os.path.join(
        REPO_ROOT, "outputs", "assets", f"low_resource_translations_{lang}_failed.json"
    )

    pilot_ids = load_pilot_ids(args.pilot_ids_path)
    print(f"[trans/{lang}] {len(pilot_ids)} pilot behavior ids")

    behaviors, missing = load_behaviors(args.behaviors_path, set(pilot_ids))
    if missing:
        print(f"[trans/{lang}] WARNING: {len(missing)} ids missing: {missing[:5]}")
    print(f"[trans/{lang}] {len(behaviors)} behaviors loaded")

    existing = {}
    if args.skip_existing and os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[trans/{lang}] {len(existing)} existing will be skipped")

    print(f"[trans/{lang}] using GoogleTranslator (deep_translator) en→{code}")
    translator = GoogleTranslator(source="en", target=code)

    out = dict(existing)
    failed = {}
    for i, b in enumerate(behaviors):
        bid = b["BehaviorID"]
        if bid in existing:
            continue
        text = (b.get("Behavior_EN") or b.get("Behavior") or "").strip()
        if not text:
            failed[bid] = {"reason": "empty behavior text"}
            continue
        try:
            translated = translator.translate(text)
        except Exception as e:
            failed[bid] = {"reason": "exception", "detail": repr(e)}
            print(f"[trans/{lang}] [{i + 1}/{len(behaviors)}] {bid} EXCEPTION: {e}")
            if args.sleep_between > 0:
                time.sleep(args.sleep_between * 4)
            continue
        if not translated or not translated.strip():
            failed[bid] = {"reason": "empty_translation"}
            print(f"[trans/{lang}] [{i + 1}/{len(behaviors)}] {bid} EMPTY")
        else:
            translated = translated.strip()
            out[bid] = translated
            ratio = ascii_ratio(translated)
            print(
                f"[trans/{lang}] [{i + 1}/{len(behaviors)}] {bid} OK  "
                f"({len(translated)} chars, ascii={ratio:.2f})"
            )
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print(
        f"[trans/{lang}] DONE. {len(out)}/{len(behaviors)} translated → "
        f"{output_path}; {len(failed)} failed → {failed_path}"
    )
    if failed:
        print(f"[trans/{lang}] failed ids:", list(failed.keys()))


if __name__ == "__main__":
    main()
