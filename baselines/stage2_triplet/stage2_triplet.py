from ..baseline import RedTeamingMethod
import json
import os


def _clean(value):
    return str(value or "").strip()


class Stage2Triplet(RedTeamingMethod):
    """
    Use Stage II T1/T2/T3 as three runnable test cases under the same BehaviorID.
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
            variants = []
            variant_logs = []
            for field in ("T1", "T2", "T3"):
                text = _clean(behavior_dict.get(field))
                if not text:
                    continue
                variants.append(self._maybe_attach_context(behavior_dict, text))
                variant_logs.append({"source_field": field})
            if not variants:
                raise ValueError(f"{behavior_id} is missing all of T1/T2/T3.")
            test_cases[behavior_id] = variants
            logs[behavior_id] = variant_logs
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
