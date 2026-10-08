import json
import os

from ..baseline import RedTeamingMethod


class LowResourceLang(RedTeamingMethod):
    """
    Low-resource-language substrate red-teaming method.

    Translates each English behavior into a single low-resource target
    language (e.g., Zulu, Yoruba) via offline translation caches. The
    translated string is sent to the target model verbatim with no decode
    hint — mirroring yong2023low's zero-shot multilingual setup.

    Reference: Yong et al. 2023, "Low-Resource Languages Jailbreak GPT-4".
    Used in this repository as one substrate of the cross-substrate
    evaluation. Target languages chosen per
    wang2025 ("Refusal Direction is Universal Across Safety-Aligned
    Languages") which reports highest ASR for Zulu and Yoruba.

    Translation cache file format (one JSON per target language):
        { "<behavior_id>": "<low-resource-language rendering>", ... }
    Generate via ``scripts/generate_low_resource_translations.py``
    before running this baseline.
    """

    SUBSTRATE_TYPE = "low_resource_language"
    REFERENCE_PAPER = "yong2023low"
    SUPPORTED_LANGUAGES = ("zulu", "yoruba")

    def __init__(
        self,
        target_language: str = "zulu",
        translations_path: str = None,
        decode_hint: bool = False,
        decode_hint_template: str = None,
        **kwargs,
    ):
        target_language = target_language.lower().strip()
        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"target_language={target_language!r} not in {self.SUPPORTED_LANGUAGES}"
            )
        self.target_language = target_language
        self.translations_path = (
            translations_path
            or f"./data/full105/low_resource_translations_{target_language}.json"
        )
        self.decode_hint = bool(decode_hint)
        self.decode_hint_template = decode_hint_template
        self._translations = None

    def _load_translations(self):
        if self._translations is not None:
            return self._translations
        if not os.path.isfile(self.translations_path):
            raise FileNotFoundError(
                f"Low-resource-language translation cache not found at "
                f"{self.translations_path}. Generate it via "
                f"scripts/generate_low_resource_translations.py "
                f"--target_language {self.target_language} first."
            )
        with open(self.translations_path, "r", encoding="utf-8") as f:
            self._translations = json.load(f)
        return self._translations

    def _build_prompt(self, translated: str) -> str:
        # yong2023low default: zero-shot, no decode hint
        if not self.decode_hint:
            return translated
        if self.decode_hint_template:
            return self.decode_hint_template.format(text=translated)
        return f"Please respond to the following:\n\n{translated}"

    def generate_test_cases(self, behaviors, verbose=False, **kwargs):
        translations = self._load_translations()
        test_cases = {}
        logs = {}
        missing = []
        for behavior_dict in behaviors:
            bid = behavior_dict["BehaviorID"]
            behavior = behavior_dict.get("Behavior", "")
            translated = translations.get(bid)
            if not translated:
                missing.append(bid)
                continue
            prompt = self._build_prompt(translated)
            test_cases[bid] = [prompt]
            logs[bid] = [
                {
                    "substrate_type": self.SUBSTRATE_TYPE,
                    "reference_paper": self.REFERENCE_PAPER,
                    "target_language": self.target_language,
                    "decode_hint": self.decode_hint,
                    "original_behavior": behavior,
                    "translated_prompt": translated,
                }
            ]
        if missing:
            raise ValueError(
                f"{len(missing)} behaviors missing from translation cache "
                f"{self.translations_path}: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        return test_cases, logs

    @staticmethod
    def get_output_file_path(save_dir, behavior_id, file_type, run_id=None):
        return os.path.join(save_dir, f"{file_type}.json")

    def save_test_cases(
        self, save_dir, test_cases, logs=None, method_config=None, run_id=None
    ):
        os.makedirs(save_dir, exist_ok=True)
        with open(
            os.path.join(save_dir, "test_cases.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(test_cases, f, indent=4, ensure_ascii=False)
        if logs is not None:
            with open(os.path.join(save_dir, "logs.json"), "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=4, ensure_ascii=False)
        if method_config is not None:
            self._replace_tokens(method_config)
            method_config["dependencies"] = self._get_dependency_versions()
            with open(
                os.path.join(save_dir, "method_config.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(method_config, f, indent=4, ensure_ascii=False)
