from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from nnsight import LanguageModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_DIR = PROJECT_ROOT / "data" / "L7_17"
ROLES = ("system", "developer", "user", "cot", "assistant", "tool")


def set_seed(seed: int = 42) -> None:
    print("setting seed")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("setting gpu seeds")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("finished setting seeds")


def load_aligned_dataset(data_dir: Path) -> TensorDataset:
    examples = torch.load(data_dir / "examples.pt", map_location="cpu")
    counter_examples = torch.load(
        data_dir / "counter_examples.pt",
        map_location="cpu",
    )
    return TensorDataset(
        examples["input_ids"],
        examples["attention_mask"],
        counter_examples["input_ids"],
        counter_examples["attention_mask"],
    )


def get_first_content_positions(
    example_input_ids: torch.Tensor,
    example_attention_mask: torch.Tensor,
    counter_input_ids: torch.Tensor,
) -> torch.Tensor:
    content_length = counter_input_ids.size(-1)
    example_windows = example_input_ids.unfold(-1, content_length, 1)
    mask_windows = example_attention_mask.unfold(-1, content_length, 1)
    matches = (example_windows == counter_input_ids[:, None, None]).all(
        dim=-1
    ) & mask_windows.bool().all(dim=-1)  # [batch, roles, windows]
    return matches.to(dtype=torch.long).argmax(dim=-1)  # [batch, roles]


def cache_layer_inputs(
    model: LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_positions: torch.Tensor,
    layer_indices: Sequence[int],
) -> torch.Tensor:
    batch_positions = torch.arange(input_ids.size(0))
    saved_activations = []
    with model.trace(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    ) as tracer:
        for layer_index in layer_indices:
            activation = (
                model.model.layers[layer_index]
                .input[batch_positions, token_positions]
                .float()
                .save()
            )  # [batch, hidden]
            saved_activations.append(activation)
        tracer.stop()

    return torch.stack(saved_activations, dim=1).detach().cpu()  # [batch, layers, hidden]


def normalize_candidate_directions(matrix: torch.Tensor) -> torch.Tensor:
    return F.normalize(matrix.float(), dim=1)  # [candidates, hidden]


def split_prompt_indices(
    n_samples: int,
    seed: int,
    train_fraction: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator)
    split = int(train_fraction * n_samples)
    return permutation[:split], permutation[split:]


def project_onto_direction(
    vectors: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    unit_direction = F.normalize(direction.float(), dim=0)  # [hidden]
    return vectors.float() @ unit_direction  # [...]


def plot_role_tag_projections(
    tagged_vectors: torch.Tensor,
    no_tag_vectors: torch.Tensor,
    direction: torch.Tensor,
    layer_indices: Sequence[int] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    import matplotlib.pyplot as plt

    tagged_scores = project_onto_direction(tagged_vectors, direction).cpu()
    no_tag_scores = project_onto_direction(no_tag_vectors, direction).cpu()
    margins = tagged_scores - no_tag_scores  # [layers]

    if layer_indices is None:
        layer_indices = range(tagged_scores.numel())
    layers = list(layer_indices)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    projection_axis, margin_axis = axes

    projection_axis.plot(layers, tagged_scores, marker="o", label="role-tag")
    projection_axis.plot(layers, no_tag_scores, marker="o", label="no-tag")
    projection_axis.set_xlabel("Layer")
    projection_axis.set_ylabel("Projection onto dominant direction")
    projection_axis.set_title("Projected class vectors")
    projection_axis.legend()

    margin_axis.axhline(0, color="black", linewidth=1)
    margin_axis.bar(layers, margins)
    margin_axis.set_xlabel("Layer")
    margin_axis.set_ylabel("role-tag − no-tag projection")
    margin_axis.set_title("Separation along dominant direction")

    figure.tight_layout()
    return figure, (projection_axis, margin_axis)


def main() -> None:
    import matplotlib.pyplot as plt

    candidate_directions = torch.load(
        ACTIVATION_DIR / "pca_matrix.pt", map_location="cpu"
    ).float()
    tagged_vectors = torch.load(
        ACTIVATION_DIR / "pos_act.pt", map_location="cpu"
    )
    no_tag_vectors = torch.load(
        ACTIVATION_DIR / "neg_act.pt", map_location="cpu"
    )

    normalized_directions = normalize_candidate_directions(candidate_directions)
    _, _, components = torch.linalg.svd(normalized_directions, full_matrices=False)
    direction = components[0]  # [hidden]
    if torch.dot(direction, normalized_directions.mean(dim=0)) < 0:
        direction = -direction

    layers = range(7, 18)
    figure, _ = plot_role_tag_projections(
        tagged_vectors,
        no_tag_vectors,
        direction,
        layers,
    )
    output_path = ACTIVATION_DIR / "role_tag_projections.png"
    figure.savefig(output_path, dpi=150)
    print(f"Saved projection plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
