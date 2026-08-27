# Deployment Guide

This project has two distinct server roles. Understanding the split is the
whole trick to deploying it anywhere:

| Role | What it does | How many | Example in this project |
|---|---|---|---|
| **Platform server** | Runs the MLflow tracking server: experiment metadata, metrics, the model registry. Nothing trains here. | Exactly **one**, shared | `192.168.26.172` |
| **Compute server** | Has a GPU (or CPU) and does the actual training. Talks to the platform server over the network. | **Any number** | This laptop (`laptop_mx450` profile) |

Every compute server points at the same platform server, so runs from a
laptop, a home desktop, and a beefy GPU box all land in one place and are
directly comparable. **The platform server must be set up and reachable
before any compute server is set up**, since every compute server's config
needs its IP.

```
                 ┌─────────────────────────┐
                 │     Platform server      │
                 │   (e.g. 192.168.26.172)  │
                 │                           │
                 │   mlflow server           │
                 │   :5000  ── tracking DB   │
                 │        └── artifact store │
                 └────────────▲──────────────┘
                               │  HTTP (tracking_uri)
              ┌────────────────┼────────────────┐
              │                │                 │
     ┌────────┴───────┐ ┌──────┴────────┐ ┌──────┴────────┐
     │ Compute server  │ │ Compute server │ │ Compute server │
     │  (laptop, MX450)│ │ (home, 3GB)    │ │ (GPU box, 24GB)│
     │ profile:        │ │ profile:       │ │ profile:       │
     │ laptop_mx450     │ │ home_server_3gb│ │ gpu_server_24gb│
     └─────────────────┘ └────────────────┘ └────────────────┘
```

---

## Part 1 — Platform server setup (do this once)

Example target: `192.168.26.172`. Any Linux or Windows box with Python
reachable on your network works the same way.

### 1.1 Prerequisites

```bash
python3 --version   # 3.9+
pip3 --version
```

### 1.2 Install MLflow

```bash
pip3 install mlflow
```

### 1.3 Choose a backend store and artifact store

For anything beyond a quick test, use a real database as the backend store
(the default local-file backend is deprecated/limited — see
[Troubleshooting](#mlflow-file-store-is-in-maintenance-mode)) and a fixed
directory (or object storage) as the artifact store:

```bash
mkdir -p /srv/mlflow/artifacts
```

### 1.4 Start the tracking server

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////srv/mlflow/mlflow.db \
  --default-artifact-root /srv/mlflow/artifacts
```

`--host 0.0.0.0` is required — binding to `127.0.0.1` (the default) makes it
unreachable from any other machine on the network, which defeats the whole
point of a shared platform server.

### 1.5 Keep it running (recommended)

The command above dies when your shell closes. For a real deployment, run it
as a service so it survives reboots and SSH disconnects.

**Linux (systemd):**

```ini
# /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=mlflow
WorkingDirectory=/srv/mlflow
ExecStart=/usr/local/bin/mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////srv/mlflow/mlflow.db \
  --default-artifact-root /srv/mlflow/artifacts
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now mlflow
```

**Windows:** run it via NSSM, Task Scheduler ("run at startup"), or inside a
persistent terminal (e.g. `tmux`/Windows Terminal tab) — any mechanism that
keeps the process alive.

### 1.6 Open the firewall port

```bash
# Linux (ufw)
sudo ufw allow 5000/tcp

# Windows
netsh advfirewall firewall add rule name="MLflow" dir=in action=allow protocol=TCP localport=5000
```

### 1.7 Verify it's reachable from another machine

From any compute server (not the platform server itself):

```bash
curl http://192.168.26.172:5000/health
# expected: 200 (empty body is fine)
```

If this fails, nothing downstream will work — fix networking/firewall before
proceeding to Part 2.

---

## Part 2 — Compute server setup (do this on every training machine)

Example target: this laptop, `laptop_mx450` profile. Repeat this entire part
on every additional machine you want to train on.

### 2.1 Prerequisites

- Python 3.10+ (Windows: a Miniconda/Anaconda install is fine and is what
  the setup scripts assume; Linux/Mac: plain `python3` is fine)
- `git`
- A GPU with drivers installed, if you intend to use a CUDA profile
  (`nvidia-smi` should list the card)

### 2.2 Clone the repo

```bash
git clone https://github.com/himanshu4nagdev/mlops-image_classification_pipeline.git
cd mlops-image_classification_pipeline
```

### 2.3 Point the config at your platform server

Open `config/env_config.yaml` and set the platform server's address once:

```yaml
mlflow:
  tracking_uri: "http://192.168.26.172:5000"   # <- your platform server IP
```

This one line is shared by every compute server — change it here, not per-machine.

### 2.4 Pick (or add) a hardware profile

Still in `config/env_config.yaml`, set `active_profile` to whichever of the
four built-in profiles matches this machine's GPU:

| Profile | Fits |
|---|---|
| `laptop_mx450` | ~2GB VRAM laptop GPUs |
| `home_server_3gb` | ~3GB VRAM |
| `gpu_server_24gb` | Large GPU servers (A100/4090-class) |
| `cpu_fallback` | No usable GPU |

```yaml
active_profile: "laptop_mx450"
```

If none fit, copy the closest block under `profiles:` and adjust
`batch_size`/`fp16`/`gradient_accumulation_steps` for the new card's VRAM —
no code changes needed anywhere else.

### 2.5a Windows compute server

Windows machines whose only Python is an Anaconda/Miniconda base install hit
a known issue: a plain `venv` inherits the base interpreter's *older*
bundled `msvcp140.dll`, which crashes torch's CUDA DLLs on import
(`WinError 1114`). `setup_and_run.bat` works around this by seeding `.venv`
from a dedicated conda environment (which carries its own modern runtime)
instead of the base install. You don't need to do anything manually — just
run it:

```bat
setup_and_run.bat
```

This is idempotent — safe to re-run any time (e.g. after a `git pull`); it
skips steps that are already done and prints `[SKIP]`/`[INSTALL]` so you can
see what happened.

To train only (skip eval/registration) on an already-set-up machine:

```bat
run_training_only.bat
```

### 2.5b Linux / Mac compute server

No conda-DLL issue exists on these platforms, so a plain venv is fine:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # adjust cu-version to the machine's CUDA driver, or omit --index-url for CPU-only
pip install -r requirements.txt

python run_pipeline.py --profile laptop_mx450   # use this machine's profile
```

### 2.6 Verify GPU + connectivity before a long run

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
curl http://192.168.26.172:5000/health
```

---

## Part 3 — Adding more compute servers

Repeat **Part 2** on each new machine. They all share the same platform
server (Part 1 is not repeated), so:

- Every machine's runs land in the same MLflow experiment (`image-clf-mobilenetv2`) — directly comparable across hardware.
- Any machine can pull the latest registered model (`mobilenetv2-cifar10`) that any other machine produced.
- Scaling out is just: clone repo → point config at the same platform server IP → pick a profile → run setup script.

---

## Command reference (what runs where)

| Step | Server | Command |
|---|---|---|
| 1 | Platform | `pip3 install mlflow` |
| 2 | Platform | `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////srv/mlflow/mlflow.db --default-artifact-root /srv/mlflow/artifacts` |
| 3 | Platform | `sudo systemctl enable --now mlflow` (or equivalent) |
| 4 | Any compute server | `git clone <repo>` |
| 5 | Any compute server | edit `config/env_config.yaml` (`tracking_uri`, `active_profile`) |
| 6 | Compute (Windows) | `setup_and_run.bat` |
| 6 | Compute (Linux/Mac) | `python3 -m venv .venv && pip install -r requirements.txt` then `python run_pipeline.py --profile <name>` |
| 7 | Any compute server | `curl http://<platform-ip>:5000/health` to sanity-check before a long run |

---

## Troubleshooting

### MLflow unreachable from a compute server
- `curl http://<platform-ip>:5000/health` should return `200`. If it times out, check the platform server's firewall (§1.6) and that `mlflow server` was started with `--host 0.0.0.0`, not `127.0.0.1`.
- The pipeline doesn't crash if this fails — `src/mlflow_utils.py` falls back to a local SQLite store (`mlruns.db`) and warns — but you lose cross-machine visibility for that run.

### `mlflow file store is in maintenance mode`
MLflow 3.x deprecated the old `file:./mlruns` backend. If you see this
exception on the platform server, use a database backend (as shown in §1.4)
instead of a bare file path.

### Windows: `WinError 1114` / torch import crash
Only happens if `.venv` was created with a plain `python -m venv` on top of
a base Anaconda/Miniconda install. Use `setup_and_run.bat`, which seeds
`.venv` from a dedicated conda environment instead — see §2.5a.

### GPU not detected (`torch.cuda.is_available()` is `False`)
- Run `nvidia-smi` — if that fails, the driver itself isn't installed/working.
- Confirm the torch build matches the driver's CUDA support: `python -c "import torch; print(torch.version.cuda)"`. Reinstall with the matching `--index-url` from [pytorch.org](https://pytorch.org/get-started/locally/) if it doesn't.
