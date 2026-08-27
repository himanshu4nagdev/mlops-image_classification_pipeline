# Image Classification Training Pipeline

End-to-end MobileNetV2 / CIFAR-10 training pipeline with MLflow experiment tracking,
designed to be trained on multiple machines of very different GPU sizes without
touching code — only `config/env_config.yaml`.

## Architecture decisions

**Why MobileNetV2 (`google/mobilenet_v2_1.0_224`)**
Trains on a 2GB laptop GPU (MX450) without exotic tricks. It's ~3.5M params vs.
~25M+ for a ResNet50, so activations and optimizer state fit in VRAM at a usable
batch size. It's also a realistic edge/mobile-deployment target, which matters
once the model moves to serving.

**Why environment profiles in one YAML file**
The same codebase needs to run unchanged on a 2GB laptop, a 3GB home server, and a
24GB GPU box (and CPU as a last resort). Rather than branching on `if hostname ==`
logic in the training code, every environment-sensitive knob (batch size, workers,
fp16, gradient accumulation, epochs) lives in a named profile. Switching machines
is a one-line edit (`active_profile: "..."`) or a `--profile` CLI flag — the
training/eval/inference code never changes. This also makes the config a single
artifact you can diff, review, and log as MLflow params (see `active_profile` logged
per run), so every run is traceable to the exact hardware/setting combination that
produced it.

**Why gradient accumulation**
On the MX450 (2GB), a batch size that fits in VRAM (8) is too small/noisy for
stable convergence on its own. Accumulating gradients over 4 micro-batches before
each optimizer step simulates an effective batch size of 32 without holding 32
samples' activations in memory at once — the standard trick for training real
batch sizes on constrained GPUs.

**Why mixed precision (fp16)**
Halves activation memory and roughly doubles throughput on the MX450's Turing
cores, which is what makes batch_size=8 + gradient accumulation practical at all
in 2GB. Disabled on the 24GB profile where memory isn't the constraint and full
precision is simpler to reason about.

## Project layout

```
config/env_config.yaml   # environment profiles + MLflow settings
src/
  config.py               # profile loading, device resolution, GPU memory logging
  mlflow_utils.py          # tracking-server setup with local-file fallback
  data_loader.py           # CIFAR-10 -> torchvision transforms -> DataLoaders
  train.py                 # training loop, MLflow run + model logging
  evaluate.py               # standalone eval (accuracy, per-class P/R/F1, confusion matrix)
  inference.py              # single-image inference from the MLflow model registry
run_pipeline.py            # orchestrator: load -> train -> evaluate -> register
```

## Switching environments

Change one line in `config/env_config.yaml`:

```yaml
active_profile: "laptop_mx450"   # -> "home_server_3gb" / "gpu_server_24gb" / "cpu_fallback"
```

or override per-run without touching the file:

```bash
python run_pipeline.py --profile gpu_server_24gb
```

Each profile fully controls batch size, worker count, epochs, LR, fp16, gradient
accumulation, and image size — everything that needs to change when the hardware
changes.

## How MLflow tracking works across machines

All profiles point at the same remote tracking server
(`mlflow.tracking_uri` in `env_config.yaml`, currently
`http://192.168.26.172:5000`). Every machine — laptop, home server, GPU box —
logs into the **same experiment** (`image-clf-mobilenetv2`), so runs from
different hardware are directly comparable in one MLflow UI, and the model
registry (`mobilenetv2-cifar10`) is shared: whichever machine trains the best
run registers a new model version that any other machine (including the
serving box) can pull by name/stage.

If the remote server is unreachable, `src/mlflow_utils.setup_mlflow()` catches
the connection failure, warns, and falls back to a local `./mlruns` file store
so training never crashes — you just lose cross-machine visibility for that
run until the network issue is fixed.

Each run logs:
- every config value from the active profile, plus `active_profile` itself, as params
- per-epoch `train_loss`, `train_accuracy`, `learning_rate` as metrics
- the trained model as an artifact (`mlflow.pytorch.log_model`)
- (from evaluation) `test_accuracy` and per-class precision/recall/F1

## Environment setup

This machine's only Python is a Miniconda install. Plain `python -m venv` on top
of it re-uses Miniconda's base interpreter DLL, which pulls in an **older bundled
`msvcp140.dll`** — new CUDA wheels (built with a newer MSVC toolset) crash on
import with a generic `WinError 1114` (`DLL initialization routine failed`).
A real conda environment doesn't have this problem: each one ships its own
`python3xx.dll` and matching VC++ runtime, fully isolated from the base env.
So: use `conda create`, not `venv`, on this machine.

```bash
conda create -n mlops-image-clf python=3.12 -y
conda activate mlops-image-clf

# CUDA-enabled torch/torchvision matched to the driver (CUDA 12.8 here)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# everything else
pip install -r requirements.txt
```

Verify the GPU is visible before training anything:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Commands

Run the full pipeline (train -> eval -> register) on the active profile:

```bash
python run_pipeline.py
```

Run on a different profile without editing the YAML:

```bash
python run_pipeline.py --profile cpu_fallback
```

Skip stages (e.g. re-evaluate without retraining is not wired to load an
existing checkpoint by default — extend `run_pipeline.py` if you need that,
or call `evaluate.py` directly, see below):

```bash
python run_pipeline.py --skip-eval
```

Train only, standalone:

```bash
python src/train.py
```

Evaluate a specific model (local path or MLflow URI) standalone:

```bash
python src/evaluate.py --model-uri runs:/<run_id>/model
python src/evaluate.py --model-path ./some/local/model/dir
```

Run inference on a single image using the latest registered model
(Production stage if set, otherwise the latest version):

```bash
python src/inference.py path/to/image.jpg
```

## Notes
Training - On Laptop
<img width="3840" height="1080" alt="image" src="https://github.com/user-attachments/assets/b0dd0d5b-66b2-4d9e-9893-ba0f00b26c0c" />

mlflow - port 5000
<img width="1892" height="970" alt="image" src="https://github.com/user-attachments/assets/f1957810-718d-4858-b8e3-4b190c5025a7" />
<img width="1901" height="973" alt="image" src="https://github.com/user-attachments/assets/790b6dee-8b9e-4c1b-bb8a-a577bdec8080" />
<img width="1892" height="966" alt="image" src="https://github.com/user-attachments/assets/017a083b-c060-4aa4-bb96-25bdad008cf2" />
<img width="1901" height="944" alt="image" src="https://github.com/user-attachments/assets/85112dab-b09d-476f-97d5-176d13e74b81" />
<img width="1894" height="965" alt="image" src="https://github.com/user-attachments/assets/a9b8eb0b-e16b-4f96-9ab4-64e55dbf1888" />

Minio - port 9001
<img width="1899" height="972" alt="image" src="https://github.com/user-attachments/assets/4b8576d6-6042-4491-809d-7581e700e4f2" />
<img width="1907" height="969" alt="image" src="https://github.com/user-attachments/assets/7e8a8d87-5ccf-4682-abc7-025c592b25e4" />
<img width="1899" height="874" alt="image" src="https://github.com/user-attachments/assets/412266e5-5fa1-438d-90e1-5ca2119f7937" />
<img width="1886" height="874" alt="image" src="https://github.com/user-attachments/assets/51a33906-3db1-4df6-abc9-4011db2f8e37" />

graphana - port 3001

<img width="1896" height="940" alt="image" src="https://github.com/user-attachments/assets/7d0e03b6-9a65-4b7d-b0b2-8d216b08a6ab" />
<img width="1893" height="969" alt="image" src="https://github.com/user-attachments/assets/b7b9d133-3ace-44ad-9dab-f5b5a6a253c5" />


Infrencing - On Browser (via Platform server)
<img width="1903" height="968" alt="image" src="https://github.com/user-attachments/assets/0a2b3546-5330-45f4-b832-11664d153e58" />


- This machine (laptop, MX450, 2GB VRAM) is for **training only**. Serving
  happens on the remote server, which is why `fastapi`/`uvicorn` are in
  `requirements.txt` but unused here.
- GPU memory is printed before and after model load (`torch.cuda.mem_get_info`)
  so VRAM headroom on the MX450 is verifiable at a glance.
