import modal
import nnsight
import torch

import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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


MODEL_NAME = "openai/gpt-oss-20b"
BATCH_SIZE = 32
LAYERS_TO_PROBE = list(range(24))
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
)


def compute_pca(matrix: torch.Tensor) -> torch.Tensor:
    float_matrix = matrix.to(dtype=torch.float32)
    _, _, components = torch.linalg.svd(float_matrix, full_matrices=False)
    direction = components[0]  # [hidden]
    reference = float_matrix.mean(dim=0)  # [hidden]
    if torch.dot(direction, reference) < 0:
        direction = -direction
    return direction


def get_content_positions(
    example_input_ids: torch.Tensor,
    example_attention_mask: torch.Tensor,
    counter_input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    content_length = counter_input_ids.size(-1)
    example_windows = example_input_ids.unfold(-1, content_length, 1)
    mask_windows = example_attention_mask.unfold(-1, content_length, 1)
    matches = (example_windows == counter_input_ids[:, None, None]).all(
        dim=-1
    ) & mask_windows.bool().all(dim=-1)  # [batch, roles, windows]
    content_starts = matches.to(dtype=torch.long).argmax(dim=-1)  # [batch, roles]
    counter_positions = torch.arange(content_length).expand(
        counter_input_ids.size(0), -1
    )  # [batch, tokens]
    example_positions = (
        content_starts[:, :, None] + counter_positions[:, None]
    )  # [batch, roles, tokens]
    return example_positions, counter_positions


def cache_layer_inputs(
    model: nnsight.LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    batch_positions = torch.arange(input_ids.size(0))[:, None]
    saved_activations = []
    with model.trace(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    ) as tracer:
        for layer_index in LAYERS_TO_PROBE:
            activation = (
                model.model.layers[layer_index]
                .input[
                    batch_positions,
                    token_positions,
                ]
                .float()
                .save()
            )  # [batch, tokens, hidden]
            saved_activations.append(activation)
        tracer.stop()

    return (
        torch.stack(saved_activations, dim=2).detach().cpu()
    )  # [batch, tokens, layers, hidden]


def load_dataloader() -> DataLoader:
    examples = torch.load(REMOTE_DATA_DIR / "examples.pt", map_location="cpu")
    counter_examples = torch.load(
        REMOTE_DATA_DIR / "counter_examples.pt",
        map_location="cpu",
    )
    dataset = TensorDataset(
        examples["input_ids"],
        examples["attention_mask"],
        counter_examples["input_ids"],
        counter_examples["attention_mask"],
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


@app.function(
    image=image,
    gpu="H200",
    secrets=[hf_secret],
    timeout=60 * 60 * 24,
    volumes={str(VOLUME_PATH): volume},
)
def build_pca_matrix(
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    dataloader = load_dataloader()
    example_activation_sum = None
    counter_activation_sum = None
    example_prompt_count = 0
    counter_prompt_count = 0

    for batch in tqdm(dataloader, desc="Caching activations"):
        (
            example_input_ids,
            example_attention_mask,
            counter_input_ids,
            counter_attention_mask,
        ) = batch
        example_positions, counter_positions = get_content_positions(
            example_input_ids,
            example_attention_mask,
            counter_input_ids,
        )
        counter_batch_activations = cache_layer_inputs(
            model,
            counter_input_ids,
            counter_attention_mask,
            counter_positions,
        )  # [batch, tokens, layers, hidden]
        counter_batch_sum = counter_batch_activations.sum(
            dim=0
        )  # [tokens, layers, hidden]
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
            )  # [batch, tokens, layers, hidden]
            example_activation_sum += example_batch_activations.sum(dim=0)

        counter_prompt_count += counter_input_ids.size(0)
        example_prompt_count += example_input_ids.size(0) * example_input_ids.size(1)

    mean_example_activations = example_activation_sum / example_prompt_count
    mean_counter_activations = counter_activation_sum / counter_prompt_count
    pca_matrix = (mean_example_activations - mean_counter_activations).flatten(
        0, 1
    )  # [tokens * layers, hidden]
    direction = compute_pca(pca_matrix.cuda()).cpu()  # [hidden]
    volume.commit()
    return pca_matrix, direction, mean_example_activations, mean_counter_activations


def compare_act(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    cosine_distance = 1 - F.cosine_similarity(x, y, dim=-1)
    euclidean_distance = torch.linalg.vector_norm(x - y, dim=-1)

    return cosine_distance, euclidean_distance


@app.local_entrypoint()
def main(seed: int = 42) -> None:
    pca_matrix, direction, positive_activations, negative_activations = (
        build_pca_matrix.remote(seed)
    )
    output_path = DATA_DIR / "pca_matrix.pt"
    torch.save(pca_matrix, output_path)
    print(f"Saved PCA matrix with shape {tuple(pca_matrix.shape)} to {output_path}")
    print(f"First principal direction shape: {tuple(direction.shape)}")

    print("Checking activations")
    torch.save(positive_activations, DATA_DIR / "pos_act.pt")
    torch.save(negative_activations, DATA_DIR / "neg_act.pt")
    print(positive_activations.size(), negative_activations.size())
    print(compare_act(positive_activations, negative_activations))
