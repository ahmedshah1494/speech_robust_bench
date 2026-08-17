# Running newer ASR models (transformers >= 5)

`ibm-granite/granite-speech-4.1-2b`, `Qwen/Qwen3-ASR-1.7B-hf`, and
`CohereLabs/cohere-transcribe-03-2026` need a much newer `transformers`
(5.15.0) than this repo's pinned main environment (`transformers==4.34.0`,
see `requirements.txt`). Bumping the main venv in place would risk breaking
every other model already working there (whisper, wav2vec2, canary,
parakeet, mms, speechbrain, the `robust_speech` adversarial pipeline).
Instead, these 3 models run out of a **second, isolated venv**, left
side-by-side with the main one. `openai/whisper-large-v3` also works fine
in this second venv (it's the generic HF pipeline path), so all 4 newly
requested models can be run from it.

This doc records the exact setup for reproducing (or recreating from
scratch) that second environment, since it lives partly outside this git
repo (see "Where this lives" below) and involved a few non-obvious fixes.

## Where this lives

- **New venv**: `/workspace/srb-venv-new` — not committed anywhere (large
  binary venv directory), rebuild with the steps below if lost.
- **Portable FFmpeg 7 build**: `/workspace/.local/ffmpeg7-shared` — also
  not committed, rebuild with the steps below.
- **Two shell env files** in `/workspace/pod-setup/` (a plain persistent
  directory on the pod's `/workspace` volume, **not a git repo** — nothing
  under it is version-controlled, which is why its content is transcribed
  here):
  - `env.sh` — the repo's existing base env (venv path, HF cache dirs,
    `HF_TOKEN`). Unchanged by this work except adding `HF_TOKEN` (see
    "HF Hub rate limiting" below).
  - `env_new_models.sh` — new file, full contents reproduced verbatim
    below. Sets `PATH`/`LD_LIBRARY_PATH` for the second venv.
- **This repo** (`speech_robust_bench`): the 3 new model modules
  (`models/granite_speech.py`, `models/qwen3_asr.py`,
  `models/cohere_transcribe.py`) plus the dispatcher/orchestrator wiring —
  committed normally, see the rest of this repo's history.

## Activating the environment

```bash
source /workspace/pod-setup/env.sh
source /workspace/pod-setup/env_new_models.sh
cd /workspace/speech_robust_bench
python evaluate_single.py --model_name ibm-granite/granite-speech-4.1-2b ...
```

`env_new_models.sh` puts `srb-venv-new`'s `bin/` ahead of the main venv's
on `PATH`, so `python`/`pip` resolve to the new venv after sourcing both
files in that order.

## Rebuilding the venv from scratch

```bash
# 1. Create the venv with the system python (not the main srb-venv's python)
/usr/bin/python3 -m venv /workspace/srb-venv-new
source /workspace/srb-venv-new/bin/activate
pip install --upgrade pip

# 2. torch + torchaudio (CUDA 12.8 wheels — matches the pod's driver, which
#    supports newer CUDA than the main venv's pinned cu121 build)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. torchcodec — MUST be the cu128-tagged build matching the exact torch
#    version above, not latest. See "torchcodec / FFmpeg / CUDA" below for
#    why this is finicky.
pip install "torchcodec==0.11.1+cu128" --index-url https://download.pytorch.org/whl/cu128

# 4. transformers + the rest of the stack (versions as actually installed;
#    "transformers" just needs to be new enough for granite_speech/
#    qwen3_asr/cohere_asr's config types — 5.15.0 is what was verified)
pip install "transformers==5.15.0" datasets evaluate jiwer soundfile scipy \
    numpy pandas tqdm accelerate sentencepiece librosa nvidia-npp-cu12
```

Versions actually verified working together (`pip list` in `srb-venv-new`):

```
torch            2.11.0+cu128
torchaudio       2.11.0+cu128
torchcodec       0.11.1+cu128
transformers     5.15.0
datasets         5.0.1
evaluate         0.4.6
jiwer            4.0.0
accelerate       1.14.0
soundfile        0.14.0
scipy            1.15.3
numpy            2.2.6
pandas           2.3.3
sentencepiece    0.2.2
librosa          0.11.0
nvidia-npp-cu12  12.4.1.87
```

## torchcodec / FFmpeg / CUDA — the fiddly part

`datasets>=5` requires `torchcodec` to decode the `Audio` feature column
(the old `soundfile`-only decode path is gone). Getting `torchcodec` to
actually import cleanly took 3 separate fixes, in this order:

1. **FFmpeg version.** The pod's system FFmpeg is 4.4
   (`libavutil.so.56`), but `torchcodec` needs FFmpeg 5, 6, or 7's shared
   libs (`libavutil.so.57/58/59`) and there's no newer FFmpeg in the
   Ubuntu 22.04 (jammy) apt repos, even backports. Fix: download a
   portable prebuilt shared FFmpeg 7 build and point `LD_LIBRARY_PATH` at
   it, without touching the system FFmpeg (which other things, e.g.
   `torchaudio` sox effects in the main venv, depend on):
   ```bash
   mkdir -p /workspace/.local
   cd /workspace/.local
   curl -sL -o ffmpeg7.tar.xz \
     https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1.tar.xz
   tar -xf ffmpeg7.tar.xz
   mv ffmpeg-n7.1-latest-linux64-gpl-shared-7.1 ffmpeg7-shared
   rm -f ffmpeg7.tar.xz
   ```
2. **torchcodec/torch ABI matching.** `pip install torchcodec` with no
   version pin resolves to the newest release (0.16.0 at the time of this
   work), which is built against **CUDA 13** (`libnvrtc.so.13`) — not
   present with a `torch+cu128` install (CUDA 12.8), so it fails to import
   with a missing-`.so` error. Downgrading torchcodec without also
   matching it to the installed torch version fails differently (an ABI
   mismatch: `undefined symbol: ..c10_cuda..` — torchcodec 0.9.x wants
   torch 2.9, not 2.11). The fix is to install the specific `+cu128`
   tagged torchcodec release matching the installed torch minor version —
   here, `torchcodec==0.11.1+cu128` for `torch==2.11.0+cu128` — from
   PyTorch's own wheel index (`--index-url
   https://download.pytorch.org/whl/cu128`), not plain PyPI. See
   https://github.com/pytorch/torchcodec#installing-torchcodec for the
   torchcodec-to-torch version compatibility table if upgrading either in
   the future.
3. **Runtime library search order + a missing NPP package.** Even with
   matching versions, plain `import torchcodec` still failed twice more:
   - `libcudart.so.12: undefined symbol: cudaGetDriverEntryPointByVersion`
     — a *different*, older CUDA 12.1 runtime install at
     `/usr/local/cuda-12.1` (unrelated to this venv, used by the main
     venv/system) was being found first via the default library search
     path. Fix: put the venv's own `torch/lib` directory at the *front* of
     `LD_LIBRARY_PATH` so torch's bundled CUDA libs win.
   - `libnppicc.so.12: cannot open shared object file` — torchcodec's
     video/color-conversion ops need NVIDIA's NPP library, which isn't a
     `torch`/`torchcodec` dependency and has to be installed separately:
     `pip install nvidia-npp-cu12`. Its lib dir also needs to be on
     `LD_LIBRARY_PATH`.

All three fixes are captured together in `env_new_models.sh` (full
contents below) so sourcing it once sets up `LD_LIBRARY_PATH` correctly in
one step; you shouldn't need to reason through this again unless the venv
is rebuilt with different versions.

`env_new_models.sh` (verbatim, lives at `/workspace/pod-setup/env_new_models.sh`):

```bash
# Env for the second venv (srb-venv-new) used to evaluate models requiring
# transformers >= 5.x (granite-speech, Qwen3-ASR, cohere-transcribe, etc.)
# that the main srb-venv (transformers==4.34.0) can't load.
# Usage: source /workspace/pod-setup/env.sh && source /workspace/pod-setup/env_new_models.sh

export SRB_VENV_NEW=/workspace/srb-venv-new

# torchcodec (needed by datasets>=4 for Audio decoding) requires:
#  - FFmpeg >=5 shared libs (system ffmpeg is only 4.4) -> portable ffmpeg7 build
#  - torch's own bundled CUDA libs to take priority over /usr/local/cuda-12.1
#  - nvidia-npp-cu12 (pip installed into srb-venv-new) for video color-convert ops
_NV_LIBS=$(find "$SRB_VENV_NEW/lib/python3.10/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="$SRB_VENV_NEW/lib/python3.10/site-packages/torch/lib:$_NV_LIBS:/workspace/.local/ffmpeg7-shared/lib:$LD_LIBRARY_PATH"
unset _NV_LIBS

case ":$PATH:" in
  *":$SRB_VENV_NEW/bin:"*) ;;
  *) export PATH="$SRB_VENV_NEW/bin:$PATH" ;;
esac
```

## HF Hub rate limiting

Unrelated to the venv itself, but hit during this same work: with no
`HF_TOKEN` configured, anonymous HuggingFace Hub API calls get rate
limited (`429 Too Many Requests`) under moderate load (e.g. downloading
several not-yet-cached perturbation configs back to back). Fixed by adding
an `HF_TOKEN` to `/workspace/pod-setup/env.sh` (this file already existed
and is sourced by every shell via `bootstrap.sh` → `.bashrc`, so it's
persisted across pod restarts, unlike a token merely exported in
`~/.bashrc` directly, which lives on the pod's ephemeral overlay). Not
reproduced here since it's a credential — see `env.sh` on the pod
directly.

## Model-specific notes

- **`ibm-granite/granite-speech-4.1-2b`** and **`Qwen/Qwen3-ASR-1.7B-hf`**
  load natively in transformers 5.15 — no `trust_remote_code` needed.
- **`CohereLabs/cohere-transcribe-03-2026`** requires
  `trust_remote_code=True` on both `AutoProcessor.from_pretrained` and
  `AutoModelForSpeechSeq2Seq.from_pretrained` — transformers 5.15 has a
  *native* `cohere_asr` model type registered too, but it's incompatible
  with this specific checkpoint's config schema; only `trust_remote_code`
  routes to the repo's own bundled (correct) modeling code via its
  `config.json`'s `auto_map`.
  - That bundled remote code also hits a real `transformers` internals
    bug on load: it defines `_keys_to_ignore_on_load_unexpected` as a
    `list`, but `transformers==5.15.0`'s
    `PreTrainedModel._adjust_missing_and_unexpected_keys` does
    `(self._keys_to_ignore_on_load_unexpected or set()) | additional_unexpected_patterns`,
    which raises `TypeError: unsupported operand type(s) for |: 'list' and 'set'`.
    Worked around with a small scoped monkeypatch at the top of
    `models/cohere_transcribe.py` that coerces the attribute to a `set`
    before calling the original method — see that file for the exact
    patch (~10 lines, self-contained, only active while
    `cohere_transcribe.py` is imported).
- **`evaluate_single.py --language` default is lowercase `'english'`**,
  but `cohere_transcribe.py`'s (and `canary.py`'s, pre-existing) language
  map is keyed by title-case `'English'`. Always pass `--language English`
  explicitly for these models, otherwise it raises (cohere) or is silently
  ignored (canary already had this bug).
- Recommended `--batch_size` on a single 46GB A40: `granite-speech` → 4,
  `qwen3-asr` → 8 (both do raw autoregressive `generate()` batching),
  `cohere-transcribe` → 16-32 (its `batch_size` is an internal
  micro-batching hint inside its own `model.transcribe(...)`, not a raw
  `generate()` batch, so it tolerates more). The repo's own
  `evaluate_single.py` argparse default of 128 would OOM granite/qwen3-asr.

## Verification performed

Full `gnoise:1` (LibriSpeech `test.clean`, 2620/2620 samples) runs via
`evaluate_single.py` under this venv, all completed cleanly:

| Model | WER | CER |
|---|---|---|
| `ibm-granite/granite-speech-4.1-2b` | 1.76% | 0.50% |
| `Qwen/Qwen3-ASR-1.7B-hf` | 2.56% | 0.74% |
| `CohereLabs/cohere-transcribe-03-2026` | 1.79% | 0.49% |
| `openai/whisper-large-v3` | confirmed working via smoke test |
