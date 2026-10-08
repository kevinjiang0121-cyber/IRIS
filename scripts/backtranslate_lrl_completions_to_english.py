"""
Back-translate LowResourceLang completion artifacts into English for a
judge-side control condition.

Purpose:
- read an existing completions JSON produced from Zulu/Yoruba prompts
- translate both `test_case` and `generation` into English via Google
  Translate (`deep_translator.GoogleTranslator`)
- save a judge-facing English completions artifact that can be fed to the
  original v2.2 English judge prompt

This is intentionally a judge-side control only. It does not change the
target model outputs themselves; it produces a translated copy for
evaluation.
"""

import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from deep_translator import GoogleTranslator  # noqa: E402
from tqdm import tqdm  # noqa: E402


LANG_CODES = {
    "zulu": "zu",
    "yoruba": "yo",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang", required=True, choices=sorted(LANG_CODES))
    parser.add_argument("--completions_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--sleep_between", type=float, default=0.15)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def translate_text(translator, text):
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return text
    if len(stripped) <= 4500:
        return translator.translate(stripped)
    return translate_long_text(translator, stripped)


def _split_long_text(text, max_len=4500):
    # Prefer paragraph / sentence boundaries before falling back to hard cuts.
    units = re.split(r"(\n\n+|(?<=[\.\!\?\n])\s+)", text)
    chunks = []
    current = ""
    for unit in units:
        if not unit:
            continue
        if len(unit) > max_len:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(unit):
                chunks.append(unit[start : start + max_len])
                start += max_len
            continue
        if not current:
            current = unit
            continue
        if len(current) + len(unit) <= max_len:
            current += unit
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def translate_long_text(translator, text):
    parts = _split_long_text(text)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        out.append(translator.translate(part))
    return "\n\n".join(out)


def main():
    args = parse_args()
    lang_code = LANG_CODES[args.source_lang]

    with open(args.completions_path, "r", encoding="utf-8") as f:
        completions = json.load(f)

    existing = {}
    if args.skip_existing and os.path.isfile(args.save_path):
        with open(args.save_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    translator = GoogleTranslator(source=lang_code, target="en")

    out = deepcopy(existing)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    def flush():
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

    behavior_ids = list(completions.keys())
    pbar = tqdm(behavior_ids, desc=f"backtranslate-{args.source_lang}", unit="behavior")
    processed_since_flush = 0
    for behavior_id in pbar:
        if behavior_id in out:
            continue
        rows = completions.get(behavior_id, [])
        translated_rows = []
        for row in rows:
            new_row = dict(row)
            new_row["test_case"] = translate_text(translator, row.get("test_case", ""))
            new_row["generation"] = translate_text(
                translator, row.get("generation", "")
            )
            translated_rows.append(new_row)
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)
        out[behavior_id] = translated_rows
        processed_since_flush += 1
        if args.checkpoint_every > 0 and processed_since_flush >= args.checkpoint_every:
            flush()
            processed_since_flush = 0

    flush()
    print(f"[backtranslate] saved {len(out)} behaviors -> {args.save_path}")


if __name__ == "__main__":
    main()
