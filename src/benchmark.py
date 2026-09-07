"""
compute_projection_scores: Projects activation vectors onto a candidate direction.
plot_role_tag_projections: Plots tagged and untagged projection clouds per token and layer.
main: Loads cached scores for tokens 0, 8, 32 and 64 and saves and shows the plots.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def compute_projection_scores(
    activations: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    direction = direction.to(device=activations.device, dtype=torch.float32)  # [hidden]
    unit_direction = F.normalize(direction, dim=0)  # [hidden]
    activations = activations.float()  # [..., hidden]
    scores = activations @ unit_direction  # [...]
    return scores


def plot_role_tag_projections(
    tagged_scores: torch.Tensor,
    no_tag_scores: torch.Tensor,
    token_offsets: Sequence[int],
    layer_indices: Sequence[int],
) -> Figure:
    tagged_scores = tagged_scores.cpu()  # [tagged_points, tokens, layers]
    no_tag_scores = no_tag_scores.cpu()  # [untagged_points, tokens, layers]
    layers = list(layer_indices)
    generator = torch.Generator().manual_seed(42)
    figure, axes_grid = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    axes = list(axes_grid.flat)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")

    for token_index, token in enumerate(token_offsets):
        axis = axes[token_index]
        color = colors[token_index]
        groups = (
            (tagged_scores, "Tagged (all six roles)", "black", "o", 9, 0.2),
            (no_tag_scores, "Untagged", color, "x", 20, 0.7),
        )
        for scores, label, point_color, marker, size, alpha in groups:
            for column, layer in enumerate(layers):
                layer_scores = scores[:, token_index, column]  # [points]
                jitter = torch.rand(
                    layer_scores.size(0), generator=generator
                )  # [points]
                jitter = (jitter - 0.5) * 0.56  # [points]
                vertical_positions = layer + jitter  # [points]
                legend_label = label if column == 0 else None
                axis.scatter(
                    layer_scores,
                    vertical_positions,
                    s=size,
                    alpha=alpha,
                    color=point_color,
                    marker=marker,
                    linewidths=0.6,
                    label=legend_label,
                )

        axis.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        axis.set_yticks(layers)
        axis.set_xlabel("Signed projection onto cached direction")
        axis.set_ylabel("Layer")
        axis.set_title(f"Content-token offset {token}", color=color)
        axis.legend(loc="best")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    figure.suptitle(
        "Individual activation projections · all prompts\n"
        "Vertical jitter separates points; each panel has its own x-axis scale"
    )
    figure.tight_layout()
    return figure


def main() -> None:
    scores_path = DATA_DIR / "L7_17" / "linear_separability_2" / "projection_scores.pt"
    scores = torch.load(scores_path, map_location="cpu")
    layers = range(7, 18)
    token_offsets = (0, 8, 32, 64)
    tagged_scores = scores["tagged"][
        :, :, token_offsets, 7:18
    ]  # [samples, roles, tokens, layers]
    tagged_scores = tagged_scores.flatten(0, 1)  # [tagged_points, tokens, layers]
    no_tag_scores = scores["raw"][:, token_offsets, 7:18]  # [samples, tokens, layers]
    figure = plot_role_tag_projections(
        tagged_scores,
        no_tag_scores,
        token_offsets,
        layers,
    )
    # output_path = DATA_DIR / "difference_in_means" / "role_tag_projection_clouds.png"
    # figure.savefig(output_path, dpi=150)
    # print(f"Saved projection plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
