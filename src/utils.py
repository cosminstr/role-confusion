"""
This script contains utilitary methods.

set_seed: Seeds PyTorch and CUDA for reproducible runs.
load_aligned_dataset: Loads tagged and untagged prompts into one aligned dataset.
get_first_content_positions: Identify the first non-role tokens positions for a fixed window size.
cache_layer_inputs: Caches layer input activations at selected token positions.
normalize_candidate_directions: Normalizes each row of the candidate-direction matrix.
split_prompt_indices: Splits shuffled prompt indices into training and held-out sets.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import modal
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from config import DirectionConfig, ProbeConfig, ROLES, SEED

if TYPE_CHECKING:
    from nnsight import LanguageModel


def set_seed(seed: int = SEED) -> None:
    print("setting seed")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("setting gpu seeds")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("finished setting seeds")


@contextmanager
def uploaded_dataset(data_dir: str | Path) -> Iterator[modal.Volume]:
    data_dir = Path(data_dir)
    with modal.Volume.ephemeral() as volume:
        with volume.batch_upload() as upload:
            upload.put_file(data_dir / "examples.pt", "/examples.pt")
            upload.put_file(data_dir / "counter_examples.pt", "/counter_examples.pt")
        yield volume


def load_aligned_dataset(data_dir: Path) -> TensorDataset:
    examples = torch.load(data_dir / "examples.pt", map_location="cpu")
    counter_examples = torch.load(
        data_dir / "counter_examples.pt",
        map_location="cpu",
    )
    return TensorDataset(
        examples["input_ids"],  # [n_samples, roles, tagged_length]
        examples["attention_mask"],  # [n_samples, roles, tagged_length]
        counter_examples["input_ids"],  # [n_samples, content_length]
        counter_examples["attention_mask"],  # [n_samples, content_length]
    )


def get_first_content_positions(
    example_input_ids: torch.Tensor,
    example_attention_mask: torch.Tensor,
    counter_input_ids: torch.Tensor,
    window_size: int,
    token_start: int = ProbeConfig.token_start,
) -> torch.Tensor:
    content_length = counter_input_ids.size(-1)
    assert token_start + window_size <= content_length, (
        f"Token window ends at {token_start + window_size}, beyond content length {content_length}"
    )
    example_windows = example_input_ids.unfold(-1, content_length, 1)
    mask_windows = example_attention_mask.unfold(-1, content_length, 1)
    matches = (example_windows == counter_input_ids[:, None, None]).all(
        dim=-1
    ) & mask_windows.bool().all(dim=-1)  # [batch, roles, windows]
    first_positions = matches.to(dtype=torch.long).argmax(dim=-1)  # [batch, roles]
    offsets = torch.arange(window_size, device=first_positions.device)  # [window_size]
    offsets = offsets + token_start  # [window_size]
    return first_positions.unsqueeze(-1) + offsets  # [batch, roles, window_size]


def cache_layer_inputs(
    model: LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_positions: torch.Tensor,  # [batch, window_size]
    layer_indices: Sequence[int],
) -> torch.Tensor:
    batch_positions = torch.arange(input_ids.size(0))  # [batch]
    batch_positions = batch_positions.unsqueeze(1)  # [batch, 1]
    saved_activations = []
    with model.trace(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    ) as tracer:
        for layer_index in layer_indices:
            layer = model.model.layers[layer_index]
            layer_input = layer.input  # [batch, sequence, hidden]
            selected_tokens = layer_input[
                batch_positions, token_positions
            ]  # [batch, window_size, hidden]
            activation = selected_tokens.float()  # [batch, window_size, hidden]
            saved_activation = activation.save()
            saved_activations.append(saved_activation)
        tracer.stop()

    activations = torch.stack(
        saved_activations, dim=1
    )  # [batch, layers, window_size, hidden]
    activations = activations.detach()
    return activations.cpu()


def normalize_candidate_directions(matrix: torch.Tensor) -> torch.Tensor:
    return F.normalize(matrix.float(), dim=1)  # [candidates, hidden]
    # normalizes the rows.


def split_prompt_indices(
    n_samples: int,
    seed: int,
    train_fraction: float = DirectionConfig.train_fraction,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator)
    split = int(train_fraction * n_samples)
    return permutation[:split], permutation[split:]
