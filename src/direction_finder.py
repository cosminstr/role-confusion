from pathlib import Path

import modal
import nnsight
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils import (
    cache_layer_inputs,
    get_first_content_positions,
    load_aligned_dataset,
    normalize_candidate_directions,
    set_seed,
    split_prompt_indices,
)


MODEL_NAME = "openai/gpt-oss-20b"
BATCH_SIZE = 32
LAYERS_TO_PROBE = list(range(7, 18, 1))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
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
    .add_local_python_source("utils")
)


def compute_dominant_direction(matrix: torch.Tensor) -> torch.Tensor:
    normalized_directions = normalize_candidate_directions(matrix)
    _, _, components = torch.linalg.svd(normalized_directions, full_matrices=False)
    direction = components[0]  # [hidden]
    reference = normalized_directions.mean(dim=0)  # [hidden]
    if torch.dot(direction, reference) < 0:
        direction = -direction
    return direction


def load_dataloader(seed: int) -> tuple[DataLoader, torch.Tensor]:
    dataset = load_aligned_dataset(REMOTE_DATA_DIR)
    train_indices, _ = split_prompt_indices(len(dataset), seed)
    train_dataset = Subset(dataset, train_indices.tolist())
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False),
        train_indices,
    )


@app.function(
    image=image,
    gpu="H200",
    secrets=[hf_secret],
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_PATH): volume},
)
def build_pca_matrix(
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    model = nnsight.LanguageModel(
        MODEL_NAME,
        device_map="auto",
        dtype="auto",
        cache_dir=VOLUME_PATH,
        dispatch=True,
    )
    model.set_experts_implementation("eager")  # for reproductibility
    model.eval()
    model.config.use_cache = False
    dataloader, train_indices = load_dataloader(seed)
    example_activation_sum = None
    counter_activation_sum = None
    example_prompt_count = 0
    counter_prompt_count = 0

    for batch in tqdm(dataloader, desc="Caching activations"):
        (
            example_input_ids,  # [batch, roles, tagged_length]
            example_attention_mask,
            counter_input_ids,  # [batch, content_length]
            counter_attention_mask,
        ) = batch
        example_positions = get_first_content_positions(
            example_input_ids,
            example_attention_mask,
            counter_input_ids,
        )
        counter_positions = torch.zeros_like(counter_input_ids[:, 0])  # [batch]
        counter_batch_activations = cache_layer_inputs(
            model,
            counter_input_ids,
            counter_attention_mask,
            counter_positions,
            LAYERS_TO_PROBE,
        )  # [batch, layers, hidden]
        counter_batch_sum = counter_batch_activations.sum(dim=0)  # [layers, hidden]
        if counter_activation_sum is None:
            counter_activation_sum = torch.zeros_like(counter_batch_sum)
            example_activation_sum = torch.zeros_like(counter_batch_sum)
        counter_activation_sum += counter_batch_sum

        for role_index in range(example_input_ids.size(1)):
            example_batch_activations = cache_layer_inputs(
                model,
                example_input_ids[:, role_index],
                example_attention_mask[:, role_index],
                example_positions[:, role_index],
                LAYERS_TO_PROBE,
            )  # [batch, layers, hidden]
            example_activation_sum += example_batch_activations.sum(dim=0)

        counter_prompt_count += counter_input_ids.size(0)
        example_prompt_count += example_input_ids.size(0) * example_input_ids.size(1)

    mean_example_activations = example_activation_sum / example_prompt_count
    mean_counter_activations = counter_activation_sum / counter_prompt_count
    pca_matrix = mean_example_activations - mean_counter_activations  # [layers, hidden]
    direction = compute_dominant_direction(pca_matrix.cuda()).cpu()  # [hidden]
    volume.commit()
    return (
        pca_matrix,
        direction,
        mean_example_activations,
        mean_counter_activations,
        train_indices,
    )


@app.local_entrypoint()
def main(seed: int = 42) -> None:
    pca_matrix, direction, positive_activations, negative_activations, train_indices = (
        build_pca_matrix.remote(seed)
    )
    output_path = DATA_DIR / "L7_17" / "pca_matrix.pt"
    torch.save(pca_matrix, output_path)
    direction_path = DATA_DIR / "L7_17" / "dominant_direction.pt"
    torch.save(direction, direction_path)
    train_indices_path = DATA_DIR / "L7_17" / "train_indices.pt"
    torch.save(train_indices, train_indices_path)
    print(f"Saved candidate direction matrix with shape {tuple(pca_matrix.shape)} to {output_path}")
    print(f"Saved dominant direction with shape {tuple(direction.shape)} to {direction_path}")
    print(f"Saved {train_indices.numel()} direction-training indices to {train_indices_path}")

    print("Checking activations")
    torch.save(positive_activations, DATA_DIR / "L7_17" / "pos_act.pt")
    torch.save(negative_activations, DATA_DIR / "L7_17" / "neg_act.pt")
    print(positive_activations.size(), negative_activations.size())
