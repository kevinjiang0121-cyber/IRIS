import os

import yaml

from ..baseline import SingleBehaviorRedTeamingMethod
from .jailbreak.base_prompt import GPTMaskPrompt, vitc_horizontal, vitc_vertical
from .utils.model import load_model

_DEFAULT_MODELS_YAML = "./configs/model_configs/models.yaml"

# Heuristic POS-skip set used as a local fallback when the masking model returns
# a multi-token phrase instead of a single trigger word. The downstream attack
# masks ONE word as ASCII art, so picking an adverb/article/preposition leaves
# the actual harmful trigger in plain text — defeating the obfuscation.
# Kept intentionally small + fast (no nltk/spacy dep); covers the common
# instruction-construction tokens that appear in red-teaming behaviors.
_POS_SKIP_TOKENS = frozenset(
    {
        # Articles & determiners
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        # Prepositions
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "into",
        "onto",
        "about",
        "over",
        "under",
        "via",
        # Conjunctions
        "and",
        "or",
        "but",
        "nor",
        "yet",
        "so",
        "if",
        "than",
        # Aux/copula verbs
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        # Pronouns
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "who",
        "whom",
        # Wh-words / question stems
        "how",
        "what",
        "where",
        "when",
        "why",
        "which",
        # Generic harm-neutral verbs that often lead red-team behaviors —
        # masking these does not obfuscate harmful intent.
        "make",
        "explain",
        "write",
        "tell",
        "generate",
        "give",
        "provide",
        "describe",
        "create",
        "produce",
        "show",
        "list",
        "compose",
        "draft",
    }
)


def _is_likely_trigger(token):
    """Return False for tokens that are unlikely to carry the harmful
    intent on their own: stopwords, generic instruction verbs, articles,
    and adverbs (heuristically detected as ``-ly``-ending tokens of
    length >= 5, e.g. ``secretly`` / ``viciously`` / ``quietly``).
    Short ``-ly`` words like ``fly`` / ``sly`` / ``ply`` are kept."""
    t = token.lower().strip(",.;:!?\"'()[]")
    if not t:
        return False
    if t in _POS_SKIP_TOKENS:
        return False
    if len(t) >= 5 and t.endswith("ly"):
        return False
    return True


def _is_placeholder(value):
    return not value or (
        isinstance(value, str) and value.startswith("<") and value.endswith(">")
    )


def _lookup_token_from_models_yaml(masking_model, models_yaml_path):
    """Look up the OpenRouter token in models.yaml whose
    ``model.model_name_or_path`` matches ``masking_model``. Returns the token
    string, or empty if no entry matches or the matched token is itself a
    placeholder."""
    if not os.path.exists(models_yaml_path):
        return ""
    with open(models_yaml_path, "r", encoding="utf-8") as f:
        models_cfg = yaml.full_load(f) or {}
    for entry in models_cfg.values():
        if not isinstance(entry, dict):
            continue
        model_block = entry.get("model") or {}
        if model_block.get("model_name_or_path") == masking_model:
            token = model_block.get("token", "")
            if not _is_placeholder(token):
                return token
    return ""


class ArtPrompt(SingleBehaviorRedTeamingMethod):
    def __init__(
        self,
        masking_model="openai/gpt-4o-2024-11-20",
        masking_model_key=None,
        masking_model_base_url=None,
        masking_direction="horizontal",
        fontname="cards",
        models_yaml_path=_DEFAULT_MODELS_YAML,
        **model_kwargs,
    ):
        """
        :param masking_model: model used to find the words to be masked
            (OpenRouter slug, e.g. ``openai/gpt-4o-2024-11-20``)
        :param masking_model_key: API key for the masking model. If empty or
            placeholder, falls back to (1) the matching entry's ``token``
            in ``models_yaml_path`` keyed by ``model_name_or_path``, then
            (2) ``OPENROUTER_API_KEY`` env var.
        :param masking_model_base_url: optional OpenAI-compatible base URL
            (e.g. ``https://openrouter.ai/api/v1`` to route via OpenRouter)
        :param models_yaml_path: path to models.yaml for token lookup
        """
        self.masking_model = masking_model

        if _is_placeholder(masking_model_key):
            resolved_key = _lookup_token_from_models_yaml(
                masking_model, models_yaml_path
            ) or os.environ.get("OPENROUTER_API_KEY", "")
        else:
            resolved_key = masking_model_key

        if not resolved_key:
            raise ValueError(
                f"No API key resolvable for masking_model={masking_model!r}. "
                f"Add a matching entry with a real token to {models_yaml_path}, "
                f"or set OPENROUTER_API_KEY in the environment."
            )
        self.masking_model_key = resolved_key
        self.masking_model_base_url = masking_model_base_url
        self.masking_direction = masking_direction
        self.fontname = fontname

    def generate_test_cases_single_behavior(
        self, behavior, num_generate, verbose=False
    ):
        behavior_dict = behavior
        behavior = behavior_dict["Behavior"]
        context_str = behavior_dict["ContextString"]

        behavior += " "
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        model = load_model(
            self.masking_model,
            self.masking_model_key,
            base_url=self.masking_model_base_url,
        )
        PT = GPTMaskPrompt()
        masking_prompt = PT.get_prompt(behavior)
        masking_response = model(masking_prompt, debug=False)
        masking_response = model.resp_parse(masking_response)[0]
        mask_words, masked_instruction = PT.parse(masking_response)

        # Upstream assumed each bracket entry is a single space-separated word.
        # Modern instruction-tuned masking models (GPT-4o etc.) often return a
        # multi-word phrase per entry, e.g. ``[viciously bullies]``. Split each
        # entry into individual tokens so the downstream single-word masker
        # still has a candidate.
        raw_tokens = [tok for entry in mask_words for tok in entry.split() if tok]

        # Filter out adverbs/articles/generic instruction verbs so that the
        # ASCII-art mask actually hides a harm-defining content word rather
        # than a function word. If the filter wipes out everything (model
        # returned only stopwords), fall back to the raw token list to
        # preserve upstream-style behavior.
        filtered_tokens = [t for t in raw_tokens if _is_likely_trigger(t)]
        candidate_tokens = filtered_tokens or raw_tokens

        chosen_word = ""
        for cand in candidate_tokens:
            cand_clean = cand.strip(",.;:!?\"'()[]")
            if cand_clean and cand_clean in behavior:
                chosen_word = cand_clean
                break

        if chosen_word:
            attack_cls = {"vertical": vitc_vertical, "horizontal": vitc_horizontal}[
                self.masking_direction
            ]
            attack_fn = getattr(attack_cls(), self.fontname)
            instruction = behavior
            masked_instruction = instruction.replace(chosen_word, "[MASK]")
            test_case = attack_fn(masked_instruction, chosen_word)
        else:
            test_case = behavior

        logs = [
            dict(
                loss=0,
                test_case=test_case,
                masking_response=masking_response,
                raw_mask_tokens=raw_tokens,
                filtered_mask_tokens=filtered_tokens,
                chosen_mask_word=chosen_word,
            )
        ]
        return [test_case], logs
