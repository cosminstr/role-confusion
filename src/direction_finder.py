from pathlib import Path
import modal
import nnsight
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import config as settings
from config import DirectionConfig, parse_indices, save_config
from utils import (
    cache_layer_inputs,
    get_first_content_positions,
    load_aligned_dataset,
    normalize_candidate_directions,
    set_seed,
    split_prompt_indices,
    uploaded_dataset,
)


app = modal.App()
volume = modal.Volume.from_name(settings.VOLUME_NAME)
hf_secret = modal.Secret.from_dotenv(settings.PROJECT_ROOT)
image = (
    modal.Image.debian_slim()
    .uv_pip_install(*settings.IMAGE_PACKAGES)
    .env(settings.IMAGE_ENV)
    .add_local_python_source("utils", "config")
)


def compute_dominant_direction(matrix: torch.Tensor) -> torch.Tensor:
    normalized_directions = normalize_candidate_directions(matrix)
    _, _, components = torch.linalg.svd(normalized_directions, full_matrices=False)
    direction = components[0]  # [hidden]
    reference = normalized_directions.mean(dim=0)  # [hidden]
    if torch.dot(direction, reference) < 0:
        direction = -direction
    return direction


def load_dataloader(config: DirectionConfig) -> tuple[DataLoader, torch.Tensor]:
    dataset = load_aligned_dataset(settings.REMOTE_DATA_DIR)
    train_indices, _ = split_prompt_indices(len(dataset), config.seed, config.train_fraction)
    train_dataset = Subset(dataset, train_indices.tolist())
    return (
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False),
        train_indices,
    )


@app.function(
    image=image,
    gpu=settings.GPU,
    secrets=[hf_secret],
    timeout=settings.TIMEOUT,
    volumes={str(settings.VOLUME_PATH): volume},
)
def build_pca_matrix(
    config: DirectionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(config.seed)
    layer_indices = parse_indices(config.layers)
    model = nnsight.LanguageModel(
        config.model_name,
        device_map=settings.MODEL_DEVICE_MAP,
        dtype=settings.MODEL_DTYPE,
        cache_dir=settings.VOLUME_PATH,
        dispatch=True,
    )
    model.set_experts_implementation(settings.EXPERTS_IMPLEMENTATION)  # for reproductibility
    model.eval()
    model.config.use_cache = settings.USE_CACHE
    dataloader, train_indices = load_dataloader(config)
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
            window_size=config.window_size,
            token_start=config.token_start,
        )
        offsets = torch.arange(config.token_start, config.token_start + config.window_size)  # [window]
        counter_positions = offsets.unsqueeze(0).expand(counter_input_ids.size(0), -1)  # [batch, window]
        counter_batch_activations = cache_layer_inputs(
            model,
            counter_input_ids,
            counter_attention_mask,
            counter_positions,
            layer_indices,
        )  # [batch, layers, window, hidden]
        counter_batch_activations = counter_batch_activations.mean(dim=2)  # [batch, layers, hidden]
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
                layer_indices,
            )  # [batch, layers, window, hidden]
            example_batch_activations = example_batch_activations.mean(dim=2)  # [batch, layers, hidden]
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
def main(
    working_dir: str,
    seed: int = DirectionConfig.seed,
    model_name: str = DirectionConfig.model_name,
    batch_size: int = DirectionConfig.batch_size,
    layers: str = DirectionConfig.layers,
    token_start: int = DirectionConfig.token_start,
    window_size: int = DirectionConfig.window_size,
    train_fraction: float = DirectionConfig.train_fraction,
    data_dir: str = DirectionConfig.data_dir,
) -> None:
    config = DirectionConfig(
        working_dir=working_dir,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        layers=layers,
        token_start=token_start,
        window_size=window_size,
        train_fraction=train_fraction,
        data_dir=str(Path(data_dir).resolve()),
    )
    output_dir = settings.DATA_DIR / config.working_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with uploaded_dataset(config.data_dir) as inputs:
        remote = build_pca_matrix.with_options(volumes={
            str(settings.VOLUME_PATH): volume,
            str(settings.REMOTE_DATA_DIR): inputs,
        })
        pca_matrix, direction, positive_activations, negative_activations, train_indices = remote.remote(config)

    output_path = output_dir / "pca_matrix.pt"
    torch.save(pca_matrix, output_path)
    direction_path = output_dir / "dominant_direction.pt"
    torch.save(direction, direction_path)
    train_indices_path = output_dir / "train_indices.pt"
    torch.save(train_indices, train_indices_path)
    save_config(config, output_dir / "direction_config.json")
    print(
        f"Saved candidate direction matrix with shape {tuple(pca_matrix.shape)} to {output_path}"
    )
    print(
        f"Saved dominant direction with shape {tuple(direction.shape)} to {direction_path}"
    )
    print(
        f"Saved {train_indices.numel()} direction-training indices to {train_indices_path}"
    )

    print("Checking activations")
    torch.save(positive_activations, output_dir / "pos_act.pt")
    torch.save(negative_activations, output_dir / "neg_act.pt")
    print(positive_activations.size(), negative_activations.size())
