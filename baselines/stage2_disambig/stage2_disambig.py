from ..baseline import RedTeamingMethod
import json
import os


def _clean(value):
    return str(value or "").strip()


class Stage2Disambig(RedTeamingMethod):
    """
    Use Stage II `tp_disambiguated` directly as the runnable prompt.
    """
    def __init__(self, include_context=False, **kwargs):
        self.include_context = include_context

    def _maybe_attach_context(self, behavior_dict, text):
        text = _clean(text)
        if not self.include_context:
            return text
        context = _clean(behavior_dict.get("ContextString"))
        if context:
            return f"{context}\n\n---\n\n{text}"
        return text

    def generate_test_cases(self, behaviors, verbose=False):
        test_cases = {}
        logs = {}
        for behavior_dict in behaviors:
            behavior_id = behavior_dict["BehaviorID"]
            text = _clean(behavior_dict.get("tp_disambiguated"))
            if not text:
                raise ValueError(f"{behavior_id} is missing tp_disambiguated.")
            test_cases[behavior_id] = [self._maybe_attach_context(behavior_dict, text)]
            logs[behavior_id] = [{"source_field": "tp_disambiguated"}]
        return test_cases, logs

    @staticmethod
    def get_output_file_path(save_dir, behavior_id, file_type, run_id=None):
        return os.path.join(save_dir, f"{file_type}.json")

    def save_test_cases(self, save_dir, test_cases, logs=None, method_config=None, run_id=None):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "test_cases.json"), "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=4, ensure_ascii=False)
        with open(os.path.join(save_dir, "logs.json"), "w", encoding="utf-8") as f:
            json.dump(logs or {}, f, indent=4, ensure_ascii=False)
        if method_config is not None:
            self._replace_tokens(method_config)
            with open(os.path.join(save_dir, "method_config.json"), "w", encoding="utf-8") as f:
                method_config["dependencies"] = self._get_dependency_versions()
                json.dump(method_config, f, indent=4, ensure_ascii=False)
