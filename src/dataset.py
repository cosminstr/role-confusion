from pathlib import Path
from typing import Any

import torch
from datasets import IterableDataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
N_SAMPLES = 256
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
        header = f"{role}"
    elif role == "cot":
        header = "assistant<|channel|>analysis"
    elif role == "assistant":
        header = "assistant<|channel|>final"
    elif role == "tool":
        header = "functions. to=assistant<|channel|>commentary"
    else:
        raise ValueError(f"Invalid role: {role}")

    return f"<|start|>{header}<|message|>{content}<|end|>"


def is_long_c4_sample(
    sample: dict[str, str],
    tokenizer: Any,
    max_seq_len: int = 512,
) -> bool:
    input_ids = tokenizer(
        sample["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len + 1,
        return_attention_mask=False,
    )["input_ids"]
    return len(input_ids) > max_seq_len


def pad_role_batch(
    batch: list[list[torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    n_roles = len(batch[0])
    sequences = [sequence for sample in batch for sequence in sample]
    masks = [torch.ones_like(sequence) for sequence in sequences]
    input_ids = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=pad_token_id,
    )  # [batch * roles, tagged_length]
    attention_mask = pad_sequence(
        masks,
        batch_first=True,
        padding_value=0,
    )  # [batch * roles, tagged_length]

    return {
        "input_ids": input_ids.view(
            len(batch), n_roles, -1
        ),  # [batch, roles, tagged_length]
        "attention_mask": attention_mask.view(
            len(batch), n_roles, -1
        ),  # [batch, roles, tagged_length]
    }


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
    counter_prompts = []
    prompt_indices = []

    for base_seq_ix, text in zip(indices, truncated_texts, strict=True):
        for role_ix, role in enumerate(ROLES):
            base_seq_indices.append(base_seq_ix)
            roles.append(role)
            prompts.append(render_single_role_gptoss(role, text))
            counter_prompts.append(text)
            prompt_indices.append(base_seq_ix * len(ROLES) + role_ix)

    return {
        "base_seq_ix": base_seq_indices,
        "role": roles,
        "prompt": prompts,
        "counter_prompt": counter_prompts,
        "prompt_ix": prompt_indices,
    }


def get_role_probe_dataset(
    tokenizer: Any,
    n_samples: int = N_SAMPLES,
    max_seq_len: int = MAX_SEQ_LEN,
    seed: int = 42,
) -> IterableDataset:
    c4 = (
        get_c4(seed)
        .select_columns(["text"])
        .filter(
            is_long_c4_sample,
            fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len},
        )
        .take(n_samples)
    )
    return c4.map(
        build_sample_seqs,
        batched=True,
        batch_size=n_samples,
        with_indices=True,
        remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len},
    )


def write_c4_role_datasets(
    tokenizer: Any,
    n_samples: int = N_SAMPLES,
    max_seq_len: int = MAX_SEQ_LEN,
    seed: int = 42,
    data_dir: str | Path = Path(__file__).resolve().parents[1] / "data",
) -> tuple[Path, Path]:
    samples = list(get_role_probe_dataset(tokenizer, n_samples, max_seq_len, seed))
    role_samples = [
        [
            torch.tensor(
                tokenizer(
                    sample["prompt"],
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"],
                dtype=torch.long,
            )
            for sample in samples[start : start + len(ROLES)]
        ]
        for start in range(0, len(samples), len(ROLES))
    ]
    counter_prompts = [
        samples[start]["counter_prompt"] for start in range(0, len(samples), len(ROLES))
    ]

    examples = pad_role_batch(role_samples, tokenizer.pad_token_id)
    counter_examples = tokenizer(
        counter_prompts,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    examples_path = data_dir / "examples.pt"
    counter_examples_path = data_dir / "counter_examples.pt"
    torch.save(examples, examples_path)
    torch.save(
        {
            "input_ids": counter_examples["input_ids"],  # [batch, max_seq_len]
            "attention_mask": counter_examples[
                "attention_mask"
            ],  # [batch, max_seq_len]
        },
        counter_examples_path,
    )

    return examples_path, counter_examples_path


def sanity_check(
    tokenizer: Any,
    examples_path: str | Path,
    counter_examples_path: str | Path,
    output_path: str | Path = Path(__file__).resolve().parents[1] / "tmp.txt",
) -> Path:
    examples = torch.load(examples_path, map_location="cpu")
    print(examples.keys())
    counter_examples = torch.load(counter_examples_path, map_location="cpu")
    print(f"Examples shape: {examples['input_ids'].size()}")
    print(f"Counter-Examples shape: {counter_examples['input_ids'].size()}")
    role_input_ids = examples["input_ids"][1]  # [roles, tagged_length]
    role_attention_mask = examples["attention_mask"][1]  # [roles, tagged_length]
    prompts = [
        tokenizer.decode(input_ids[attention_mask.bool()], skip_special_tokens=False)
        for input_ids, attention_mask in zip(
            role_input_ids,
            role_attention_mask,
            strict=True,
        )
    ]
    counter_input_ids = counter_examples["input_ids"][1]  # [max_seq_len]
    counter_attention_mask = counter_examples["attention_mask"][1]  # [max_seq_len]
    prompts.append(
        tokenizer.decode(
            counter_input_ids[counter_attention_mask.bool()],
            skip_special_tokens=False,
        )
    )

    output_path = Path(output_path)
    output_path.write_text("\n---\n".join(prompts), encoding="utf-8")
    return output_path


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    # write_c4_role_datasets(tokenizer)

    sanity_check(
        tokenizer,
        "data/examples.pt",
        "data/counter_examples.pt",
    )


if __name__ == "__main__":
    main()
