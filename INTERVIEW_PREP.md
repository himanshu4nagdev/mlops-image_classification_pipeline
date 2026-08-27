# The MLOps Field Manual

Everything you need to talk confidently about this image-classification
pipeline — starting from "what even is MLOps," with zero assumed background.
Every concept below is tied to a real file, a real number, or a real bug you
personally fixed while building it.

> An interactive, better-formatted version of this same content (sidebar nav,
> collapsible Q&A, live-styled cheat sheet) also exists as a published page —
> ask for the link if you want it back. This file is the durable, in-repo copy.

**Project:** `mobilenetv2-cifar10` &middot; **Stack:** PyTorch, Transformers, MLflow, CUDA 12.8 &middot; **Your hardware:** MX450 laptop GPU, 2GB VRAM

---

## Level 00 — What is MLOps, actually?

Skip this if you've heard it a hundred times. If it's a brand-new phrase,
start here — everything else in this manual is just this idea, applied.

**Ordinary software engineering ships code.** You write a function, test it,
deploy it. If it works on Monday, it works on Tuesday — the code doesn't
change itself.

**Machine learning ships a moving target.** A trained model isn't just code —
it's code *plus* a specific dataset *plus* a specific set of hyperparameters
*plus* the exact hardware/software environment it was trained in. Change any
one of those and you get a different model. And models degrade over time as
the real world drifts away from the data they were trained on.

**MLOps is the discipline of making that moving target manageable** —
borrowing the "automate it, track it, make it reproducible" instincts of
DevOps and applying them to the ML-specific parts: datasets, training runs,
and model versions.

### The analogy that makes it click

A one-off `train.py` you run on your laptop is **home cooking**: you know
what you did, but nobody else can reproduce your dinner from memory, and if
it's great, there's no way to serve it to a hundred people tomorrow.

MLOps is **running a restaurant kitchen**: every dish (model) is cooked from
a written recipe (config), every ingredient batch (dataset/run) is logged,
there's a pass-through window between the kitchen (training) and the dining
room (serving), and any chef on shift can reproduce last Tuesday's special
exactly.

### The five pillars

- **Experiment tracking** — record every run's settings and results so you can compare them later.
- **Reproducibility** — same config, same data, same result, on any machine.
- **Model registry & versioning** — a librarian for trained models, so "which model is live?" always has an answer.
- **Environment management** — training happens on wildly different hardware; the code shouldn't have to change.
- **Deployment / serving** — kept deliberately separate from training.

> **Say this in the interview:** "MLOps is DevOps applied to the parts unique
> to ML — the data, the training run, and the model artifact — on top of the
> usual code discipline."

---

## Level 01 — The 30-second pitch

Memorize this paragraph. It's the answer to "walk me through what you built."

> **Say this in the interview:** "I built an end-to-end training pipeline for
> a 10-class image classifier — CIFAR-10 — using a MobileNetV2 backbone
> pretrained on ImageNet and fine-tuned for the new classes. It's
> config-driven, so the exact same code trains on a 2GB laptop GPU or a 24GB
> GPU server without any code changes — just swapping which hardware profile
> is active. Every run is tracked centrally with MLflow — params, per-epoch
> metrics, the model artifact — and successful runs get pushed into an MLflow
> model registry. The architecture splits two roles: one shared *platform
> server* that only runs the MLflow tracking service, and any number of
> *compute servers* that only train — so a laptop, a home desktop, and a
> rented GPU box can all train against the same registry."

### The architecture, visually

```
                 Platform server (one, shared)
                 192.168.26.172:5000
                 MLflow tracking + model registry
                              ▲
              runs are logged here over HTTP
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                       │
 Compute server         Compute server          Compute server
 This laptop (MX450)    Home desktop            Rented GPU box
 profile:               profile:                profile:
 laptop_mx450           home_server_3gb          gpu_server_24gb
```

### What actually happens, in order

1. **Load** — CIFAR-10 pulled from Hugging Face, resized to 224×224, normalized to ImageNet statistics.
2. **Train** — MobileNetV2's head is swapped from 1,000 ImageNet classes to 10 CIFAR classes, fine-tuned with mixed precision + gradient accumulation to fit 2GB VRAM.
3. **Evaluate** — accuracy, per-class precision/recall/F1, confusion matrix on the held-out test set.
4. **Register** — the trained model is versioned into the MLflow model registry, retrievable by name from any machine.

---

## Level 02 — Core concepts, tied to your code

### Experiment tracking — `src/train.py`

**Plain English:** every time you train, you write down what settings you
used and what happened — automatically, to a shared place, instead of in
your head or a spreadsheet.

**Why it matters:** without it, "which run got 91% accuracy and what
learning rate did it use?" is unanswerable a week later.

**Where:** `mlflow.start_run()` opens a tracked run; `mlflow.log_params(config)`
records every hyperparameter (batch size, LR, fp16, profile name…);
`mlflow.log_metric("train_loss", value, step=epoch)` records per-epoch results.

> **Say this:** "Every run logs its full config as MLflow params and
> per-epoch loss/accuracy as metrics, so any two runs are directly
> comparable in the MLflow UI — including which hardware profile produced them."

### The model registry — `run_pipeline.py`, `src/inference.py`

**Plain English:** experiment tracking answers "what did we try?" — the
registry answers "which trained model is *the* one we serve?" It's a
versioned, named pointer to a specific model artifact.

**Why it matters:** a serving machine shouldn't need to know a run ID. It
asks the registry for `mobilenetv2-cifar10`, latest version (or the one
flagged `Production`), and gets a model back.

**Where:** `mlflow.register_model(model_uri, name)` promotes a run's
artifact into the registry; `inference.py`'s `resolve_model_uri()` looks up
the latest `Production` version, falling back to the latest untagged one.

> **Say this:** "Training and serving never talk about run IDs directly —
> they talk about a registered model name and a stage. That indirection is
> what lets me swap in a better model without touching the serving code."

### Config-driven environments — `config/env_config.yaml`, `src/config.py`

**Plain English:** instead of writing `if hostname == "laptop": batch_size = 8`
inside the training code, every hardware-sensitive knob lives in one YAML
file, grouped into named *profiles*.

**Why it matters:** this project has to train, unmodified, on a 2GB laptop
and a 24GB server. Branching logic inside the code doesn't scale past two
machines and is untestable; a named profile does — and it's also a piece of
data you can log (`active_profile` is logged as an MLflow param on every
run), so every result is traceable to the exact hardware that produced it.

> **Say this:** "Switching machines is a one-line YAML edit or a `--profile`
> flag — the training code itself never branches on hardware."

### Separating training from serving — `DEPLOYMENT.md`

**Plain English:** the machine that trains a model and the machine that
serves predictions don't have to be — and usually shouldn't be — the same box.

**Why it matters:** training wants short, expensive bursts of raw GPU
compute; serving wants steady uptime, low latency, and predictable cost.
Coupling them means a training job can starve the very server answering live
requests. This project's "platform server / compute server" split is that
separation made concrete: the platform server is a small, always-on box;
compute servers are disposable, can be scaled out independently, and never
serve traffic.

### Mixed precision (fp16) — `src/train.py`, `run_epoch()`

**Plain English:** numbers inside the model are normally stored as 32-bit
floats. Mixed precision does most of the math in 16-bit floats instead —
half the memory, and roughly double the throughput on a modern GPU's tensor
cores — while keeping a 32-bit master copy of the weights for
numerically-sensitive steps.

**Why it matters:** this is one of the two things that makes training
possible at all on 2GB of VRAM.

**Where:** `torch.autocast(device_type=..., enabled=fp16)` wraps the forward
pass; `torch.amp.GradScaler` keeps gradients from underflowing to zero in 16-bit.

### Gradient accumulation — `src/train.py`, `run_epoch()`

**Plain English:** you want the training stability of a batch of 32 images,
but 32 images' worth of activations don't fit in 2GB. So you process 4
micro-batches of 8, add up (accumulate) their gradients *without* updating
the model yet, and only step the optimizer once every 4th micro-batch. The
model sees the statistical effect of a batch of 32 while only ever holding 8
images in memory at once.

**Where:** `gradient_accumulation_steps: 4` in the laptop profile; the
optimizer only calls `.step()` when `(step + 1) % grad_accum_steps == 0`.

> **Say this:** "Batch size 8 physically fits in 2GB; accumulating over 4
> steps gives an effective batch size of 32 without ever materializing 32
> images' activations at once."

### Transfer learning — `src/train.py`, `build_model()`

**Plain English:** instead of training a network from random weights, you
start from a network already trained on a huge, general dataset (ImageNet,
1,000 classes) and only retrain the parts that need to change for your new task.

**Why it matters:** the early layers of an image model learn generic things
— edges, textures, shapes — that transfer to almost any visual task. Reusing
them means you need far less data and far less compute to reach good
accuracy on a new 10-class problem than training from scratch would.

**Where:** `AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224",
num_labels=10, ignore_mismatched_sizes=True)` — that last flag is doing real
work: the checkpoint's final classifier layer is shaped for 1,001 classes
(1,000 ImageNet + background); `ignore_mismatched_sizes` tells Transformers
to discard just that layer and reinitialize a fresh 10-class head, while
keeping every earlier, pretrained layer intact.

### Graceful degradation — `src/mlflow_utils.py`

**Plain English:** if the tracking server is unreachable, training should
still work — just with a warning — instead of crashing an hour into a run.

**Where:** `setup_mlflow()` pings the remote server's `/health` endpoint with
a 5-second timeout; on failure it warns and points MLflow at a local SQLite
file (`sqlite:///mlruns.db`) instead, so the run is still fully tracked,
just not centrally — you get the data back later instead of losing the run.

---

## Level 03 — Code tour, file by file

If someone says "share your screen and show me," this is the order to walk
through it in.

**`config/env_config.yaml`** — *the single source of truth for "how"*
Four named hardware profiles (`laptop_mx450`, `home_server_3gb`,
`gpu_server_24gb`, `cpu_fallback`) plus one `mlflow:` section with the
tracking URI, experiment name, and registry model name. Changing
`active_profile` is the only edit needed to retarget the whole pipeline at
different hardware.

**`src/config.py`** — *loads the YAML, resolves the device*
`load_config()` reads the YAML and flattens the active profile into one
dict. `resolve_device()` falls back to CPU with a warning if CUDA was
requested but isn't available, instead of crashing. `print_gpu_memory()`
prints free/used/total VRAM — used before and after model load to prove
things fit in budget.

**`src/data_loader.py`** — *CIFAR-10 → tensors*
Pulls `uoft-cs/cifar10` from Hugging Face, wraps it in a thin
`torch.utils.data.Dataset`, and applies `Resize(224) → RandomHorizontalFlip
(train only) → ToTensor → Normalize(ImageNet mean/std)`.
`get_dataloaders(config)` builds train/test loaders using whatever batch
size, worker count, and pin-memory setting the active profile specifies.

**`src/train.py`** — *the training loop*
`build_model()` loads and re-heads MobileNetV2. `run_epoch()` is one pass
over the training set with mixed precision + gradient accumulation.
`train()` wraps all of it in an MLflow run: logs params, logs metrics every
epoch, and logs the final model artifact. Has its own
`if __name__ == "__main__"` block so it can be run standalone.

**`src/evaluate.py`** — *how good is it, really*
Loads a model either from a local path or an MLflow URI. Runs it over the
test set with `@torch.no_grad()`, computes accuracy plus a full per-class
precision/recall/F1 report (`sklearn.metrics.classification_report`) and a
confusion matrix, then logs those metrics back into the same MLflow run.

**`src/inference.py`** — *single-image prediction, for local testing*
Pulls the latest `Production` (or, failing that, latest untagged) version
straight from the model registry by name — no run ID needed. Preprocesses
one image with the same eval transform used in training/eval, and prints
the top-3 predicted classes with confidence scores.

**`src/mlflow_utils.py`** — *the safety net*
One function, `setup_mlflow()`: health-checks the remote tracking server
and points MLflow at it if reachable, or at a local SQLite fallback if not.
Every other file calls this before touching MLflow.

**`run_pipeline.py`** — *the orchestrator*
Ties it together: load config → train → evaluate → register. Accepts
`--profile`, `--skip-train`, `--skip-eval`, and prints a final summary with
the MLflow run URL.

**`setup_and_run.bat` / `run_training_only.bat`** — *idempotent Windows setup*
Checks what already exists (conda env, `.venv`, torch+CUDA, the rest of
`requirements.txt`) and only does the steps that are missing, printing
`[SKIP]` or `[INSTALL]` for each one — safe to re-run any time, e.g. after a
`git pull`.

---

## Level 04 — War stories: real bugs you actually fixed

This is your strongest interview material. These aren't hypotheticals —
every one of these happened, verbatim, while building this project. "Tell me
about a hard bug you debugged" has four real answers now.

### War story 01 — The CUDA import that crashed with a meaningless Windows error

**Symptom:** A brand-new virtual environment installed torch cleanly, but
`import torch` crashed instantly with `OSError: [WinError 1114] A dynamic
link library (DLL) initialization routine failed` — a generic Windows error
with no obvious cause.

**Investigation:** Isolated it by loading each of torch's ~50 bundled DLLs
individually with `ctypes.WinDLL()` to find which one actually failed
(`c10.dll` and an unrelated `shm.dll` — both, which ruled out a
CUDA-specific cause). Confirmed the driver and CUDA runtime themselves
loaded fine. The real clue came from Windows' own crash log (Event Viewer →
Application, event ID 1000/1001): the faulting module was `msvcp140.dll`,
loaded from `C:\ProgramData\miniconda3\` at version `14.29.30153.0` — while
System32 had a much newer `14.50.35719.0`.

**Root cause:** A plain `python -m venv` built on top of the base Miniconda
interpreter doesn't get its own copy of the Python DLL or C++ runtime — it
loads them from the base conda install, which injects its own (older,
pinned) runtime directory early into the process's DLL search path. Torch's
newer, differently-compiled DLLs need CRT symbols that old runtime doesn't have.

**Fix:** Stopped layering a venv on the base conda install. Instead: create
a real `conda create -n <env> python=3.12` environment first (each conda
environment ships its own modern runtime DLLs, fully isolated from base),
then build `.venv` *from that environment's own interpreter* — still gives a
completely normal `.venv\Scripts\python.exe` layout, but seeded from a
modern runtime instead of an ancient one.

**What this shows:** comfortable reading raw OS-level crash evidence
(Windows Event Viewer, DLL load order) instead of stopping at "reinstall and
hope," and picking a fix that solves the root cause rather than papering
over the symptom.

### War story 02 — The safe fallback that would have crashed the moment it was needed

**Symptom:** The graceful-degradation path — falling back to
`file:./mlruns` when the remote MLflow server is unreachable — looked
correct and matched long-standing MLflow convention. Deliberately testing it
(by pointing at an unreachable address) surfaced an immediate crash:
`MlflowException: The filesystem tracking backend… is in maintenance mode`.

**Root cause:** MLflow 3.x quietly deprecated the classic file-based
tracking store; it now refuses to start unless you opt out with an
environment variable or switch to a database-backed store. The
currently-installed MLflow version (3.15.1) enforces this — code that was
correct for MLflow 1.x/2.x silently became wrong.

**Fix:** Switched the local fallback to a SQLite-backed URI
(`sqlite:///mlruns.db`) — MLflow's own recommended replacement — and added
it to `.gitignore`.

**What this shows:** a habit of actually exercising error/fallback paths
instead of trusting that they're fine because the code "looks right" — this
exact bug would have hidden silently until the one moment it mattered: the
network being down during a real training run.

### War story 03 — The crash that only shows up after hours of training

**Symptom:** End-to-end testing of the registry round-trip (log → register →
resolve → load) surfaced: `MlflowException: If serialization_format is set
to 'pt2', then input_example is required` — on a call,
`mlflow.pytorch.log_model(model, artifact_path="model")`, that had no
input_example argument at all.

**Root cause:** This MLflow version changed `log_model()`'s default
`serialization_format` to `'pt2'` — PyTorch's dynamo trace-export format —
which needs a concrete example input to trace the graph. This call sits at
the very *end* of the training function, after all epochs finish — on real
hardware this would have meant losing a multi-hour run's model artifact
after training had already succeeded.

**Fix:** Explicit `serialization_format="pickle"` — a deliberate, documented
choice, since the model's `forward()` takes keyword arguments
(`pixel_values`, `labels`) that don't suit trace-based export cleanly.

**What this shows:** caught by testing the *full path* end-to-end before
trusting it, not just the parts that are easy to unit-test in isolation —
the kind of bug that's invisible until it costs you the most.

### War story 04 — The script that lies about succeeding

**Symptom:** Caught during review (before ever being run): the setup script
ended with `endlocal` immediately followed by `exit /b %PIPELINE_EXIT%`.

**Root cause:** `endlocal` discards every variable set since the matching
`setlocal` — including `PIPELINE_EXIT`, captured right after the pipeline
ran. By the time `exit /b` read it, the variable was empty, so the script
would exit with code 0 (success) no matter what the actual pipeline did — a
CI/automation false-positive waiting to happen.

**Fix:** Dropped the stray `endlocal` — the process exiting already tears
down its local scope, so it wasn't needed at the very end of the script anyway.

**What this shows:** exit codes are part of the contract of any automation
script, and a script that "looks like" it works can still lie about whether
it actually did — worth reviewing even code nobody has run yet.

---

## Level 05 — Likely questions, and how to answer them

### Conceptual

<details>
<summary><b>What's the difference between experiment tracking and a model registry?</b></summary>

Tracking is the lab notebook — every run you tried, with its settings and
results, mostly useful for comparison and debugging. The registry is the
release shelf — a small set of named, versioned models that are actually fit
to be served. Most runs never make it to the registry; only the ones you
promote with `mlflow.register_model()` do.
</details>

<details>
<summary><b>What is model drift, and does this project handle it?</b></summary>

Drift is when the live data a deployed model sees gradually stops looking
like its training data, so accuracy degrades silently over time. This
project doesn't monitor for it today — that's an honest gap, and a good
"what would you add next" answer (see below).
</details>

<details>
<summary><b>What's CI/CD for machine learning?</b></summary>

The same idea as regular CI/CD — automatically test and deploy on every
change — extended to also automatically retrain and re-evaluate when the
data or code changes, gate promotion on evaluation metrics crossing a
threshold, and version the data alongside the code. This project doesn't
have that automation layer yet; today, `run_pipeline.py` is triggered manually.
</details>

### This project, specifically

<details>
<summary><b>Why MobileNetV2 instead of a bigger model like ResNet50?</b></summary>

VRAM. MobileNetV2 is a mobile/edge-oriented architecture with a far smaller
memory footprint than a classic ResNet, which is what makes training at a
usable batch size possible at all on a 2GB card. It's also a realistic
target for eventual edge/mobile deployment, which matters once serving is in scope.
</details>

<details>
<summary><b>Why gradient accumulation instead of just using a smaller batch size?</b></summary>

Batch size 8 alone is small enough to be noisy and unstable to train with
directly. Gradient accumulation gets the statistical stability of a batch of
32 — better gradient estimates, smoother convergence — while the peak memory
footprint at any instant still only reflects 8 images. It buys stability
without buying memory.
</details>

<details>
<summary><b>Why split the platform server and compute servers instead of one box that does both?</b></summary>

They have opposite resource profiles: training wants short, expensive GPU
bursts and can tolerate downtime between runs; the tracking/registry service
wants to be small, cheap, and always up so any machine can log to it any
time. Coupling them means a training job could starve the service everyone
else depends on. It also lets you add compute capacity (more laptops, a
rented GPU box) without touching the platform server at all.
</details>

<details>
<summary><b>What happens if the MLflow server goes down mid-training?</b></summary>

Nothing crashes. `setup_mlflow()` health-checks the server before the run
starts and falls back to a local SQLite store if it's unreachable, with a
warning. If the server drops *during* a run, individual
`log_metric`/`log_param` calls are wrapped in try/except and warn rather
than raise — training keeps going either way.
</details>

<details>
<summary><b>How would you scale this to more classes or a bigger dataset?</b></summary>

The data loader and profile system barely change — swap the dataset in
`data_loader.py` and bump `num_labels`. The real scaling lever is hardware:
point `active_profile` at `gpu_server_24gb` (bigger batches, fp16 off since
memory isn't the constraint anymore, more workers) and the exact same code
trains a larger dataset without modification — that's the entire reason the
profile system exists.
</details>

### Honest gaps — own these, don't hide them

<details>
<summary><b>What's missing from this pipeline that a production system would need?</b></summary>

Being specific here reads far better than pretending nothing's missing:

- **No automated tests** — no unit tests on the data pipeline or model-building logic.
- **No CI/CD** — `run_pipeline.py` is triggered manually, not on a schedule or a data/code change.
- **No data versioning** — CIFAR-10 is a fixed public dataset here; a real project needs something like DVC or lakeFS once the data itself changes over time.
- **No drift/monitoring** — nothing watches live prediction distributions once a model is serving.
- **Stage-based registry** — `Production`/`None` stages are the API used here; MLflow's newer versions are moving toward aliases/tags instead, and that's a known, deliberate compatibility choice, not an oversight.
</details>

<details>
<summary><b>Why didn't you just use TensorFlow Serving / TorchServe / a full CI pipeline?</b></summary>

Scope. The brief was a training pipeline for compute-constrained hardware
with central tracking — serving infrastructure was explicitly out of scope
and lives on a separate server. Adding a serving framework before the
training side was solid would have been solving a problem I didn't have yet.
</details>

---

## Level 06 — Cheat sheet

### VRAM budget, verified end-to-end

`865 MiB used / 1182 MiB free` — after one real forward+backward+optimizer
step, out of 2048 MiB total on the MX450 (42% utilization).

| Setting | `laptop_mx450` |
|---|---|
| Batch size | 8 |
| Gradient accumulation | 4 steps → effective batch 32 |
| Mixed precision | fp16 on |
| Image size | 224 × 224 |
| Learning rate | 0.001 (AdamW + cosine annealing) |
| Epochs | 5 |
| Train batches / epoch | 6,250 (50,000 images ÷ 8) |
| Test batches | 625 (10,000 images ÷ 16) |

| Component | Version / value |
|---|---|
| GPU | NVIDIA GeForce MX450, 2048 MiB VRAM |
| Driver / CUDA | 573.91 / CUDA 12.8 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| transformers | 5.15.1 |
| mlflow | 3.15.1 (client) · server at 2.18.0 |
| Backbone checkpoint | `google/mobilenet_v2_1.0_224` |
| Dataset | `uoft-cs/cifar10` · 10 classes, 50k/10k split |
| MLflow experiment | `image-clf-mobilenetv2` |
| Registered model name | `mobilenetv2-cifar10` |

### Commands worth having memorized

```bash
# full pipeline, laptop profile
python run_pipeline.py --profile laptop_mx450

# train only, no eval/registration
python src/train.py --profile laptop_mx450

# evaluate a specific registered/run model
python src/evaluate.py --model-uri runs:/<run_id>/model

# single-image inference from the registry
python src/inference.py path/to/image.jpg

# is the GPU visible right now?
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Level 07 — Glossary

- **Artifact** — Any file MLflow stores alongside a run — usually the trained model itself, but can be a plot, a report, anything.
- **Autocast** — PyTorch's context manager that automatically runs eligible operations in 16-bit precision for mixed-precision training.
- **Backbone** — The main body of a neural network (everything before the final task-specific layer) — here, MobileNetV2's convolutional layers.
- **Checkpoint** — A saved snapshot of a model's weights, usually with a name identifying what it was trained on.
- **Compute server** — In this project's architecture: any machine that trains models. Talks to the platform server but never serves traffic itself.
- **Fine-tuning** — Continuing to train an already-pretrained model on a new, usually smaller, task-specific dataset.
- **Gradient accumulation** — Summing gradients across several small forward/backward passes before updating weights once, to simulate a larger batch size under memory constraints.
- **GradScaler** — Scales loss values up before the backward pass in mixed-precision training so small gradients don't underflow to zero in 16-bit floats, then unscales before the optimizer step.
- **Model registry** — A versioned, named catalog of trained models, separate from the raw experiment-tracking history, meant to answer "which model should be served?"
- **Platform server** — In this project's architecture: the one shared machine running the MLflow tracking service and model registry.
- **Run** — One tracked execution of a training (or evaluation) script in MLflow — has its own ID, params, metrics, and artifacts.
- **Stage (MLflow)** — A label on a specific model version — e.g. `Production` or `None` — used to signal which version should currently be served.
- **Transfer learning** — Reusing a model trained on one (usually large, general) task as the starting point for a different, related task.
- **VRAM** — Video RAM — the GPU's own dedicated memory, where model weights, activations, and gradients all have to fit during training.

---

*Built alongside the pipeline itself — every number and story above came
from actually running this code, not from a template.*
