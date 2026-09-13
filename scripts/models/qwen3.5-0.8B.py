"""Official Qwen3.5-0.8B text model; dimensions from its pinned HF configuration."""

from miles.utils.external_utils.model_args_utils import load_sibling_model_args


def model_args() -> str:
    tokens = load_sibling_model_args(__file__, "qwen3.5-4B").split()
    dimensions = {
        "--num-attention-heads": "8",
        "--num-query-groups": "2",
        "--num-layers": "24",
        "--hidden-size": "1024",
        "--ffn-hidden-size": "3584",
    }
    for flag, value in dimensions.items():
        tokens[tokens.index(flag) + 1] = value
    return " ".join(tokens)
