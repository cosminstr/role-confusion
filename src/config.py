import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REMOTE_DATA_DIR = Path("/data")
VOLUME_PATH = Path("/hf")
VOLUME_NAME = "hf"
GPU = "H200"
TIMEOUT = 60 * 60 * 24
MODEL_NAME = "openai/gpt-oss-20b"
SEED = 42
ROLES = ("system", "developer", "user", "cot", "assistant", "tool")
MODEL_DEVICE_MAP = "auto"
MODEL_DTYPE = "auto"
EXPERTS_IMPLEMENTATION = "eager"
USE_CACHE = False
IMAGE_PACKAGES = (
    "kernels",
    "nnsight==0.7.0",
    "torch==2.13.0",
    "tqdm==4.70.0",
    "transformers==5.15.1",
)
PLOT_IMAGE_PACKAGES = ("matplotlib>=3.11.1",)
IMAGE_ENV = {
    "HF_HOME": str(VOLUME_PATH),
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


@dataclass
class DatasetConfig:
    model_name: str = MODEL_NAME
    seed: int = SEED
    data_dir: str = str(DATA_DIR)
    n_samples: int = 256
    max_seq_len: int = 512
    dataset_name: str = "allenai/c4"
    dataset_subset: str = "en"
    dataset_split: str = "validation"
    shuffle_buffer: int = 50_000


@dataclass
class ProbeConfig:
    model_name: str = MODEL_NAME
    seed: int = SEED
    data_dir: str = str(DATA_DIR)
    batch_size: int = 32
    layers: str = "7:18"
    token_start: int = 0


@dataclass(kw_only=True)
class DirectionConfig(ProbeConfig):
    working_dir: str
    window_size: int = 1
    train_fraction: float = 0.8


@dataclass(kw_only=True)
class BenchmarkConfig(ProbeConfig):
    activations_dir: str
    window_size: int = 512
    token_offsets: str = "0,8,32,64"


def parse_indices(value: str) -> list[int]:
    indices = []
    for part in value.split(","):
        if ":" in part:
            bounds = [int(bound) for bound in part.split(":")]
            indices.extend(range(*bounds))
        else:
            indices.append(int(part))
    return indices


def save_config(
    config: DatasetConfig | DirectionConfig | BenchmarkConfig, path: Path
) -> None:
    path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
