from typing import Any

from datasets import IterableDataset, load_dataset
from transformers import AutoTokenizer


ROLES = ("system", "user", "cot", "assistant", "tool")
N_SAMPLES = 250
MAX_SEQ_LEN = 512


def get_c4(seed: int = 42) -> IterableDataset:
    return load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=50_000)


def render_single_role_gptoss(role: str, content: str) -> str:
    if role in ("system", "developer", "user"):
        header = f"{role}<|message|>"
    elif role == "cot":
        header = "assistant<|channel|>analysis<|message|>"
    elif role == "assistant":
        header = "assistant<|channel|>final<|message|>"
    elif role == "tool":
        header = "functions. to=assistant<|channel|>commentary<|message|>"
    else:
        raise ValueError(f"Invalid role: {role}")

    return f"<|start|>{header}{content}<|end|>"


def build_sample_seqs(
    examples: dict[str, list[str]],
    indices: list[int],
    tokenizer: Any,
    max_seq_len: int,
) -> dict[str, list[Any]]:
    input_ids = tokenizer(
        examples["text"],
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_seq_len,
    )["input_ids"]
    truncated_texts = tokenizer.batch_decode(input_ids)

    base_seq_indices = []
    roles = []
    prompts = []
    prompt_indices = []

    for base_seq_ix, text in zip(indices, truncated_texts, strict=True):
        for role_ix, role in enumerate(ROLES):
            base_seq_indices.append(base_seq_ix)
            roles.append(role)
            prompts.append(render_single_role_gptoss(role, text))
            prompt_indices.append(base_seq_ix * len(ROLES) + role_ix)

    return {
        "base_seq_ix": base_seq_indices,
        "role": roles,
        "prompt": prompts,
        "prompt_ix": prompt_indices,
    }


def get_role_probe_dataset(
    tokenizer: Any,
    n_samples: int = N_SAMPLES,
    max_seq_len: int = MAX_SEQ_LEN,
    seed: int = 42,
) -> IterableDataset:
    c4 = get_c4(seed).select_columns(["text"]).take(n_samples)
    return c4.map(
        build_sample_seqs,
        batched=True,
        batch_size=n_samples,
        with_indices=True,
        remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len},
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    dataset = get_role_probe_dataset(tokenizer, n_samples=2)

    for sample in dataset:
        print(sample)


if __name__ == "__main__":
    main()
