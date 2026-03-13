# Local Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all hardcoded absolute paths with a `local_config.yaml` system so both fitten and ts3 servers can share the same git branch.

**Architecture:** A `local_config.py` module loads `local_config.yaml` from the project root at import time and exposes a `get_paths()` function. YAML configs use placeholder values that get overridden by `arguments.py` at startup. Standalone scripts (`get_metric.py`, `generate_test_json.py`) import `local_config` directly.

**Tech Stack:** Python, PyYAML (already available), OmegaConf (already used)

---

### Task 1: Create local_config module

**Files:**
- Create: `local_config.py`
- Create: `local_config.example.yaml`
- Modify: `.gitignore`

**Step 1: Create `local_config.example.yaml`**

```yaml
# Copy this file to local_config.yaml and edit paths for your server.
# local_config.yaml is in .gitignore and will not be committed.

paths:
  # CogVideoX-5B SAT format transformer checkpoint
  transformer: /path/to/CogVideoX-5b-sat/transformer
  # 3D VAE checkpoint
  vae: /path/to/CogVideoX-5b-sat/vae/3d-vae.pt
  # SigLIP2 vision encoder
  siglip2: /path/to/models/google/siglip2-so400m-patch14-384
  # Directory containing video clips (mp4 files)
  video_dir: /path/to/datasets/videos
  # Dataset root (contains sub-XXXX/, JSON files, captions)
  dataset_root: /path/to/datasets
```

**Step 2: Create `local_config.py`**

```python
import os
import yaml

_config = None
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def _load_config():
    global _config
    if _config is not None:
        return _config
    config_path = os.path.join(_PROJECT_ROOT, "local_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"local_config.yaml not found at {config_path}. "
            f"Copy local_config.example.yaml to local_config.yaml and edit paths for your server."
        )
    with open(config_path) as f:
        _config = yaml.safe_load(f)
    return _config

def get_paths():
    """Return the paths dict from local_config.yaml."""
    return _load_config()["paths"]
```

**Step 3: Add `local_config.yaml` to `.gitignore`**

Append to existing `.gitignore`:
```
local_config.yaml
```

**Step 4: Create fitten's `local_config.yaml`**

```yaml
paths:
  transformer: /data/lilehui/cinebrain/CogVideoX-5b-sat/transformer
  vae: /data/lilehui/cinebrain/CogVideoX-5b-sat/vae/3d-vae.pt
  siglip2: /data/lilehui/cinebrain/models/google/siglip2-so400m-patch14-384
  video_dir: /data/lilehui/cinebrain/datasets/videos
  dataset_root: /data/lilehui/cinebrain/datasets
```

**Step 5: Commit**

```bash
git add local_config.py local_config.example.yaml .gitignore
git commit -m "feat: add local_config module for server-specific paths"
```

---

### Task 2: Wire local_config into YAML config loading

**Files:**
- Modify: `arguments.py:258-283` (process_config_to_args)
- Modify: `configs/cogvideox_5b_lora_brain_va.yaml:92`
- Modify: `configs/sft_5b_brain_va_clip.yaml:6,15-16`

**Step 1: Replace hardcoded paths in YAML configs with placeholder `__LOCAL_CONFIG__` prefix**

In `configs/cogvideox_5b_lora_brain_va.yaml` line 92, change:
```yaml
      ckpt_path: "/data/lilehui/cinebrain/CogVideoX-5b-sat/vae/3d-vae.pt"
```
to:
```yaml
      ckpt_path: __LOCAL_CONFIG_VAE__
```

In `configs/sft_5b_brain_va_clip.yaml`, change lines 6, 15, 16:
```yaml
  load: __LOCAL_CONFIG_TRANSFORMER__
  ...
  train_data: [ "__LOCAL_CONFIG_DATASET_ROOT__/sub-0005_train_va.json" ]
  valid_data: [ "__LOCAL_CONFIG_DATASET_ROOT__/sub-0005_test_va.json" ]
```

**Step 2: Add path injection in `arguments.py` `process_config_to_args()`**

After line 262 (`config = OmegaConf.merge(*configs)`), add local_config injection:

```python
    # Inject local_config paths into merged config
    try:
        from local_config import get_paths
        lp = get_paths()

        # Replace __LOCAL_CONFIG__ placeholders in args section
        if "args" in config:
            args_cfg = config["args"]
            if args_cfg.get("load") == "__LOCAL_CONFIG_TRANSFORMER__":
                args_cfg["load"] = lp["transformer"]
            # Replace dataset_root in train_data / valid_data lists
            for key in ("train_data", "valid_data"):
                if key in args_cfg and isinstance(args_cfg[key], (list, ListConfig)):
                    args_cfg[key] = [
                        p.replace("__LOCAL_CONFIG_DATASET_ROOT__", lp["dataset_root"])
                        for p in args_cfg[key]
                    ]

        # Replace in model config (VAE ckpt_path)
        if "model" in config:
            _replace_placeholder(config["model"], "__LOCAL_CONFIG_VAE__", lp["vae"])
    except (ImportError, FileNotFoundError):
        pass  # No local_config, use paths as-is in YAML
```

Add helper before `process_config_to_args`:

```python
def _replace_placeholder(d, placeholder, value):
    """Recursively replace placeholder string in a nested dict/DictConfig."""
    if isinstance(d, (dict, omegaconf.DictConfig)):
        for k in d:
            if d[k] == placeholder:
                d[k] = value
            elif isinstance(d[k], (dict, omegaconf.DictConfig, list, omegaconf.ListConfig)):
                _replace_placeholder(d[k], placeholder, value)
    elif isinstance(d, (list, omegaconf.ListConfig)):
        for i, item in enumerate(d):
            if item == placeholder:
                d[i] = value
            elif isinstance(item, (dict, omegaconf.DictConfig, list, omegaconf.ListConfig)):
                _replace_placeholder(item, placeholder, value)
```

**Step 3: Commit**

```bash
git add arguments.py configs/cogvideox_5b_lora_brain_va.yaml configs/sft_5b_brain_va_clip.yaml
git commit -m "feat: inject local_config paths into YAML configs at startup"
```

---

### Task 3: Wire local_config into Python modules

**Files:**
- Modify: `sgm/modules/encoders/modules.py:117`
- Modify: `data_video.py:515`

**Step 1: Update `sgm/modules/encoders/modules.py` line 117**

Change:
```python
        self.siglip_model = AutoModel.from_pretrained("/data/lilehui/cinebrain/models/google/siglip2-so400m-patch14-384")
```
to:
```python
        from local_config import get_paths
        self.siglip_model = AutoModel.from_pretrained(get_paths()["siglip2"])
```

**Step 2: Update `data_video.py` line 515**

Change:
```python
        self.text_processor = AutoProcessor.from_pretrained("/data/lilehui/cinebrain/models/google/siglip2-so400m-patch14-384")
```
to:
```python
        from local_config import get_paths
        self.text_processor = AutoProcessor.from_pretrained(get_paths()["siglip2"])
```

**Step 3: Commit**

```bash
git add sgm/modules/encoders/modules.py data_video.py
git commit -m "feat: use local_config for SigLIP2 model path"
```

---

### Task 4: Wire local_config into standalone scripts

**Files:**
- Modify: `get_metric.py:25`
- Modify: `tools/generate_test_json.py:3,29`

**Step 1: Update `get_metric.py`**

At top of file, after imports, add:
```python
from local_config import get_paths
```

Change line 25:
```python
            os.path.join("/data/lilehui/cinebrain/datasets/videos", f'{str(i).zfill(6)}.mp4'),
```
to:
```python
            os.path.join(get_paths()["video_dir"], f'{str(i).zfill(6)}.mp4'),
```

**Step 2: Update `tools/generate_test_json.py`**

Replace lines 1-3:
```python
import json, os

data_root = "/data/lilehui/cinebrain/datasets"
```
with:
```python
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from local_config import get_paths
_paths = get_paths()
data_root = _paths["dataset_root"]
video_dir = os.path.basename(_paths["video_dir"])  # "videos" or "clips"
```

Change line 29 from:
```python
        video_path = os.path.join(data_root, "videos", f"{str(video_id).zfill(6)}.mp4")
```
to:
```python
        video_path = os.path.join(data_root, video_dir, f"{str(video_id).zfill(6)}.mp4")
```

**Step 3: Commit**

```bash
git add get_metric.py tools/generate_test_json.py
git commit -m "feat: use local_config for paths in standalone scripts"
```

---

### Task 5: Create ts3's local_config.yaml and sync

**Step 1: Create ts3's `local_config.yaml`**

On ts3 at `/public/home/maoyaoxin/xxt/CineBrain/local_config.yaml`:
```yaml
paths:
  transformer: /public/home/maoyaoxin/xxt/CogVideoX-5b-sat/transformer
  vae: /public/home/maoyaoxin/xxt/CogVideoX-5b-sat/vae/3d-vae.pt
  siglip2: /public/home/maoyaoxin/xxt/models/google/siglip2-so400m-patch14-384
  video_dir: /public/home/maoyaoxin/xxt/datasets/clips
  dataset_root: /public/home/maoyaoxin/xxt/datasets
```

**Step 2: Sync code to ts3**

Since hardcoded paths are removed, ts3 can now use the same main branch:
```bash
# On ts3
git checkout main
git pull origin main
# local_config.yaml is already in place (gitignored)
```

**Step 3: Delete the ts3 branch (no longer needed)**

```bash
git branch -d ts3
git push origin --delete ts3  # if it was pushed
```

---

### Summary of changes

| File | Change |
|------|--------|
| `local_config.py` | New: config loader module |
| `local_config.example.yaml` | New: template for users |
| `.gitignore` | Add `local_config.yaml` |
| `arguments.py` | Add `_replace_placeholder()` + injection in `process_config_to_args()` |
| `configs/cogvideox_5b_lora_brain_va.yaml` | VAE path → placeholder |
| `configs/sft_5b_brain_va_clip.yaml` | transformer/data paths → placeholders |
| `sgm/modules/encoders/modules.py` | SigLIP2 path → `get_paths()["siglip2"]` |
| `data_video.py` | SigLIP2 path → `get_paths()["siglip2"]` |
| `get_metric.py` | video path → `get_paths()["video_dir"]` |
| `tools/generate_test_json.py` | data_root/video_dir → `get_paths()` |
