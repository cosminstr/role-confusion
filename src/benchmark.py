import modal
import nnsight
from tqdm import tqdm
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib.figure import Figure
import config as settings
from config import BenchmarkConfig, parse_indices, save_config
from utils import (
    ROLES,
    get_first_content_positions,
    load_aligned_dataset,
    set_seed,
    cache_layer_inputs,
    uploaded_dataset,
)


app = modal.App()
volume = modal.Volume.from_name(settings.VOLUME_NAME)
hf_secret = modal.Secret.from_dotenv(settings.PROJECT_ROOT)
image = (
    modal.Image.debian_slim()
    .uv_pip_install(*settings.IMAGE_PACKAGES, *settings.PLOT_IMAGE_PACKAGES)
    .env(settings.IMAGE_ENV)
    .add_local_python_source("utils", "config")
)


def compute_projection_scores(
    activations: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    direction = direction.to(device=activations.device, dtype=torch.float32)  # [hidden]
    unit_direction = F.normalize(direction, dim=0)  # [hidden]
    activations = activations.float()  # [..., hidden]
    scores = activations @ unit_direction  # [...]
    return scores


@app.function(
    image=image,
    gpu=settings.GPU,
    secrets=[hf_secret],
    timeout=settings.TIMEOUT,
    volumes={str(settings.VOLUME_PATH): volume},
)
def get_sequence_scores(
    direction: torch.Tensor, config: BenchmarkConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(config.seed)
    layer_indices = parse_indices(config.layers)
    model = nnsight.LanguageModel(
        config.model_name,
        device_map=settings.MODEL_DEVICE_MAP,
        dtype=settings.MODEL_DTYPE,
        cache_dir=settings.VOLUME_PATH,
        dispatch=True,
    )
    model.set_experts_implementation(settings.EXPERTS_IMPLEMENTATION)
    model.eval()
    model.config.use_cache = settings.USE_CACHE

    dataloader = DataLoader(
        load_aligned_dataset(settings.REMOTE_DATA_DIR),
        batch_size=config.batch_size,
        shuffle=False,
    )

    role_batches, raw_batches = [], []
    for x_ids, x_mask, y_ids, y_mask in tqdm(dataloader):
        window = get_first_content_positions(
            x_ids, x_mask, y_ids, config.window_size, config.token_start
        )  # [batch, roles, window_size]
        offsets = torch.arange(config.token_start, config.token_start + config.window_size)  # [window_size]
        raw_positions = (
            offsets.unsqueeze(0).expand(y_ids.size(0), -1)
        )  # [B, window_size]
        raw_activations = cache_layer_inputs(
            model, y_ids, y_mask, raw_positions, layer_indices
        )
        raw_batches.append(compute_projection_scores(raw_activations, direction))
        # these activations are huge so delete them
        del raw_activations

        per_role_scores = []
        for role in range(len(ROLES)):
            positions = window[:, role, :]  # [batch, window_size]
            role_input_ids = x_ids[:, role, :]  # [batch, tagged_length]
            role_attention_mask = x_mask[:, role, :]  # [batch, tagged_length]
            role_activations = cache_layer_inputs(
                model, role_input_ids, role_attention_mask, positions, layer_indices
            )  # [batch, layers, window_size, hidden]
            per_role_scores.append(
                compute_projection_scores(role_activations, direction)
            )
            del role_activations  # idem
        role_batches.append(
            torch.stack(per_role_scores, dim=1)
        )  # [batch, roles, layers, window_size]

    tagged_scores = torch.cat(role_batches)  # [samples, roles, layers, tokens]
    raw_scores = torch.cat(raw_batches)  # [samples, layers, tokens]
    return tagged_scores, raw_scores


def plot_role_tag_projections(
    tagged_scores: torch.Tensor,
    no_tag_scores: torch.Tensor,
    token_offsets: Sequence[int],
    layer_indices: Sequence[int],
    seed: int = BenchmarkConfig.seed,
) -> Figure:
    tagged_scores = tagged_scores.cpu()  # [tagged_points, tokens, layers]
    no_tag_scores = no_tag_scores.cpu()  # [untagged_points, tokens, layers]
    layers = list(layer_indices)
    generator = torch.Generator().manual_seed(seed)
    columns = min(2, len(token_offsets))
    rows = (len(token_offsets) + columns - 1) // columns
    figure, axes_grid = plt.subplots(
        rows, columns, figsize=(7.5 * columns, 5.5 * rows), sharey=True, squeeze=False
    )
    axes = list(axes_grid.flat)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")

    for token_index, token in enumerate(token_offsets):
        axis = axes[token_index]
        color = colors[token_index % len(colors)]
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

    for axis in axes[len(token_offsets):]:
        figure.delaxes(axis)

    figure.suptitle(
        "Individual activation projections · all prompts\n"
        "Vertical jitter separates points; each panel has its own x-axis scale"
    )
    figure.tight_layout()
    return figure


@app.local_entrypoint()
def main(
    activations_dir: str,
    seed: int = BenchmarkConfig.seed,
    model_name: str = BenchmarkConfig.model_name,
    batch_size: int = BenchmarkConfig.batch_size,
    layers: str = BenchmarkConfig.layers,
    token_start: int = BenchmarkConfig.token_start,
    window_size: int = BenchmarkConfig.window_size,
    token_offsets: str = BenchmarkConfig.token_offsets,
    data_dir: str = BenchmarkConfig.data_dir,
) -> None:
    config = BenchmarkConfig(
        activations_dir=activations_dir,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        layers=layers,
        token_start=token_start,
        window_size=window_size,
        token_offsets=token_offsets,
        data_dir=str(Path(data_dir).resolve()),
    )
    direction_dir = settings.DATA_DIR / config.activations_dir
    direction = torch.load(direction_dir / "dominant_direction.pt", map_location="cpu")
    with uploaded_dataset(config.data_dir) as inputs:
        remote = get_sequence_scores.with_options(volumes={
            str(settings.VOLUME_PATH): volume,
            str(settings.REMOTE_DATA_DIR): inputs,
        })
        tagged_scores, no_tag_scores = remote.remote(direction, config)
    save_config(config, direction_dir / "benchmark_config.json")

    layer_indices = parse_indices(config.layers)
    plot_offsets = parse_indices(config.token_offsets)
    score_indices = torch.tensor(plot_offsets) - config.token_start  # [plot_tokens]
    tagged_scores = tagged_scores.index_select(-1, score_indices)  # [samples, roles, layers, tokens]
    tagged_scores = tagged_scores.flatten(0, 1)  # [tagged_points, layers, tokens]
    tagged_scores = tagged_scores.transpose(1, 2)  # [tagged_points, tokens, layers]
    no_tag_scores = no_tag_scores.index_select(-1, score_indices)  # [samples, layers, tokens]
    no_tag_scores = no_tag_scores.transpose(1, 2)  # [samples, tokens, layers]
    figure = plot_role_tag_projections(
        tagged_scores,
        no_tag_scores,
        plot_offsets,
        layer_indices,
        seed=config.seed,
    )
    plt.show()
