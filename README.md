# Role confusion as prompt injection follow-up

Finding a role feature in GPT-OSS-20b, following <https://role-confusion.github.io/>.

## Benchmark a candidate direction

Run from `role-confusion/`. Requires `uv sync`, Modal authentication, the `hf`
Modal Volume, and `.env`. Both Modal scripts use an H200; plotting runs locally.
All optional arguments default to `src/config.py`; only the run directories are required.

### 1. Generate datasets (skip if already present)

```bash
uv run python src/dataset.py --n-samples 256 --max-seq-len 512 --seed 42
```

- `--n-samples`: aligned prompts; `--max-seq-len`: content tokens; `--seed`: shuffle seed.
- `--data-dir`: output directory (default `data/`); overwrites both dataset `.pt` files.
- `--model-name`: tokenizer (default `openai/gpt-oss-20b`); role formatting remains GPT-OSS-specific.
- C4 options: `--dataset-name`, `--dataset-subset`, `--dataset-split`, `--shuffle-buffer`.

### 2. Find the direction (skip if already available)

```bash
uv run modal run src/direction_finder.py --working-dir L7_17
```

- `--working-dir`: output folder under `data/` (created automatically), or an absolute path.
- `--layers`: `7:18` by default; lists (`7,10,17`) and ranges (`7:18:2`) work; stop is exclusive.
- `--token-start`: first content offset (0); `--window-size`: token count (1).
- Multiple tokens are averaged per layer before computing the direction.
- `--train-fraction`: training prompt fraction (0.8); `--seed`: model/split seed (42).
- Saves matrix, direction, training indices, class means, and `direction_config.json` together.

### 3. Benchmark the direction

```bash
uv run modal run src/benchmark.py --activations-dir L7_17
```

- `--activations-dir`: folder under `data/` (or absolute path) containing `dominant_direction.pt`.
- `--layers`, `--token-start`, `--window-size`: scoring window (defaults `7:18`, 0, 512).
- `--token-offsets`: plotted content offsets (`0,8,32,64`); list/range syntax as above.
- Offsets are relative to content start, not window start; they must lie inside the scoring window.
- Both Modal scripts accept `--model-name`, `--batch-size` (32), `--seed` (42), `--data-dir` (`data/`).
- `--data-dir` selects local input files uploaded to a temporary Modal Volume; windows must fit the content.
- Defaults score all prompts, including training data; scores/plots are not saved. Close the window to exit.
- Dataset/benchmark settings are saved as `dataset_config.json` / `benchmark_config.json` (overwritten on rerun).
- Keep datasets/model fixed when comparing directions; override only the arguments you want to change.
