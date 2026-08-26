import os
import modal
import nnsight

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2])


app = modal.App("r-confusion")
image = (
    modal.Image.debian_slim()
    .uv_pip_install("datasets>=5.0.1")
    .env({"HF_TOKEN": os.environ["HF_TOKEN"], "HF_XET_HIGH_PERFORMANCE": "1"})
)
volume = modal.Volume.from_name("hf")
VOLUME_PATH = Path("/hf")
