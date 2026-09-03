from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import modal
import nnsight
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ROLES, get_first_content_positions, load_aligned_dataset, set_seed

if TYPE_CHECKING:
    import pandas as pd


MODEL_NAME = "openai/gpt-oss-20b"
BATCH_SIZE = 32
LAYERS_TO_PROBE = tuple(range(24))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ACTIVATION_DIR = DATA_DIR / "L7_17" / "linear_separability"
REMOTE_DATA_DIR = Path("/data")
VOLUME_PATH = Path("/hf")

app = modal.App()
volume = modal.Volume.from_name("hf")
hf_secret = modal.Secret.from_dotenv(PROJECT_ROOT)
image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "kernels",
        "nnsight==0.7.0",
        "torch==2.13.0",
        "tqdm==4.70.0",
        "transformers==5.15.1",
    )
    .env(
        {
            "HF_HOME": str(VOLUME_PATH),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(
        DATA_DIR / "examples.pt",
        str(REMOTE_DATA_DIR / "examples.pt"),
    )
    .add_local_file(
        DATA_DIR / "counter_examples.pt",
        str(REMOTE_DATA_DIR / "counter_examples.pt"),
    )
    .add_local_file(
        ACTIVATION_DIR / "dominant_direction.pt",
        str(REMOTE_DATA_DIR / "dominant_direction.pt"),
    )
    .add_local_python_source("utils")
)


def cache_projection_scores(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_positions: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    batch_positions = torch.arange(input_ids.size(0))[:, None]
    direction = direction.to(device=model.device, dtype=torch.float32)
    saved_scores = []
    with model.trace(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    ) as tracer:
        for layer_index in LAYERS_TO_PROBE:
            scores = (
                model.model.layers[layer_index]
                .input[batch_positions, token_positions]
                .float()
                @ direction
            ).save()  # [batch, tokens]
            saved_scores.append(scores)
        tracer.stop()

    return torch.stack(saved_scores, dim=2).detach().cpu()  # [batch, tokens, layers]


@app.function(
    image=image,
    gpu="H200",
    secrets=[hf_secret],
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_PATH): volume},
)
def collect_projection_scores(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    model = nnsight.LanguageModel(
        MODEL_NAME,
        device_map="auto",
        dtype="auto",
        cache_dir=VOLUME_PATH,
        dispatch=True,
    )
    model.set_experts_implementation("eager")
    model.eval()
    model.config.use_cache = False

    direction = torch.load(
        REMOTE_DATA_DIR / "dominant_direction.pt",
        map_location="cpu",
    )
    dataloader = DataLoader(
        load_aligned_dataset(REMOTE_DATA_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    tagged_batches = []
    raw_batches = []

    for batch in tqdm(dataloader, desc="Projecting activations"):
        (
            example_input_ids,  # [batch, roles, tagged_length]
            example_attention_mask,
            counter_input_ids,  # [batch, content_length]
            counter_attention_mask,
        ) = batch
        content_starts = get_first_content_positions(
            example_input_ids,
            example_attention_mask,
            counter_input_ids,
        )
        offsets = torch.arange(counter_input_ids.size(1))
        raw_positions = offsets[None].expand(counter_input_ids.size(0), -1)
        raw_batches.append(
            cache_projection_scores(
                model,
                counter_input_ids,
                counter_attention_mask,
                raw_positions,
                direction,
            )
        )

        role_scores = []
        for role_index in range(len(ROLES)):
            example_positions = content_starts[:, role_index, None] + offsets[None]
            role_scores.append(
                cache_projection_scores(
                    model,
                    example_input_ids[:, role_index],
                    example_attention_mask[:, role_index],
                    example_positions,
                    direction,
                )
            )
        tagged_batches.append(torch.stack(role_scores, dim=1))

    tagged_scores = torch.cat(tagged_batches)  # [samples, roles, tokens, layers]
    raw_scores = torch.cat(raw_batches)  # [samples, tokens, layers]
    return tagged_scores, raw_scores


def get_test_indices(n_samples: int, train_indices: torch.Tensor) -> torch.Tensor:
    test_mask = torch.ones(n_samples, dtype=torch.bool)
    test_mask[train_indices] = False
    return torch.arange(n_samples)[test_mask]


def benchmark_scores(
    tagged_scores: torch.Tensor,
    raw_scores: torch.Tensor,
    train_indices: torch.Tensor,
) -> pd.DataFrame:
    import pandas as pd

    test_indices = get_test_indices(tagged_scores.size(0), train_indices)
    test_positive = tagged_scores[test_indices].flatten(0, 1).float()
    test_negative = raw_scores[test_indices].float()
    tagged_high_margin = (
        test_positive.min(dim=0).values - test_negative.max(dim=0).values
    )
    tagged_low_margin = (
        test_negative.min(dim=0).values - test_positive.max(dim=0).values
    )
    separation_margin = torch.maximum(tagged_high_margin, tagged_low_margin)
    token, layer = torch.meshgrid(
        torch.arange(tagged_scores.size(2)),
        torch.tensor(LAYERS_TO_PROBE),
        indexing="ij",
    )

    return pd.DataFrame(
        {
            "token": token.flatten().numpy(),
            "layer": layer.flatten().numpy(),
            "linearly_separable": (separation_margin > 0).flatten().numpy(),
        }
    )


def plot_heatmap(results: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(18, 4))
    values = results.pivot(index="layer", columns="token", values="linearly_separable")
    image = axis.imshow(
        values.astype(float),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        cmap="viridis",
    )
    axis.set_xlabel("Content-token offset")
    axis.set_ylabel("Layer")
    axis.set_title("Exact linear separability on held-out projections")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


@app.local_entrypoint()
def main(seed: int = 42) -> None:
    tagged_scores, raw_scores = collect_projection_scores.remote(seed)
    train_indices = torch.load(
        ACTIVATION_DIR / "train_indices.pt",
        map_location="cpu",
    )
    results = benchmark_scores(tagged_scores, raw_scores, train_indices)

    csv_path = ACTIVATION_DIR / "linear_separability_3.csv"
    heatmap_path = ACTIVATION_DIR / "linear_separability_heatmap_3.png"
    results.to_csv(csv_path, index=False)
    plot_heatmap(results, heatmap_path)

    print(
        f"Exactly separable held-out layer/token pairs: "
        f"{results['linearly_separable'].sum()} / {len(results)}"
    )
    print(f"Saved exact linear separability metrics to {csv_path}")
    print(f"Saved exact linear separability heatmap to {heatmap_path}")
