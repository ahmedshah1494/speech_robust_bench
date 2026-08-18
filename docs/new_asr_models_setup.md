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
- **Two shell env files** normally live at `/workspace/pod-setup/` (a plain
  persistent directory on the pod's `/workspace` volume, **not a git
  repo**, and lost entirely if the pod is *terminated*, not just
  stopped/restarted). Git-tracked backups of both, plus `bootstrap.sh`, now
  live in this repo's [`pod-setup/`](../pod-setup/README.md) — copy them
  back to `/workspace/pod-setup/` on a fresh pod (see that dir's README):
  - `env.sh` — the repo's existing base env (venv path, HF cache dirs).
    The committed copy does **not** include a real `HF_TOKEN` (that would
    be a leaked credential) — see "HF Hub rate limiting" below for how to
    set your own.
  - `env_new_models.sh` — sets `PATH`/`LD_LIBRARY_PATH` for the second
    venv; full contents also reproduced inline below.
- **This repo** (`speech_robust_bench`): the 3 new model modules
  (`models/granite_speech.py`, `models/qwen3_asr.py`,
  `models/cohere_transcribe.py`) plus the dispatcher/orchestrator wiring —
  committed normally, see the rest of this repo's history. The `robust_speech`
  submodule's adversarial-support additions (see below) need their own
  commit inside that submodule — check `git -C robust_speech status`.

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

# 1b. setuptools: the venv ships an old setuptools (59.6.0) that lacks both
#     PEP 660 editable-install support and a bundled pkg_resources inside
#     pip's isolated build env - needed later for `pip install -e
#     robust_speech`. Upgrade now; keep <82 since torch pins setuptools<82.
pip install "setuptools>=68,<82" wheel

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

`env_new_models.sh` (verbatim; canonical copy is
[`pod-setup/env_new_models.sh`](../pod-setup/env_new_models.sh) in this
repo — copy it to `/workspace/pod-setup/env_new_models.sh` on a fresh pod,
this inline copy is for readability and should match):

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
an `HF_TOKEN` to `/workspace/pod-setup/env.sh` (sourced by every shell via
`bootstrap.sh` → `.bashrc`, so it persists across pod stop/start, unlike a
token merely exported in `~/.bashrc` directly, which lives on the pod's
ephemeral overlay and doesn't survive even a restart).

The git-tracked `pod-setup/env.sh` in this repo does **not** carry a real
token (would be a leaked credential). It sources
`/workspace/pod-setup/env.local.sh` if present — put your token there
(this file is intentionally *not* committed):
```bash
echo 'export HF_TOKEN=hf_...your_token...' > /workspace/pod-setup/env.local.sh
```

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

## Adversarial (PGD / universal) support

The `robust_speech` submodule's attacks (`SNRPGDAttack`, `UniversalAttack`)
are model-agnostic — they only need a `BaseASR` subclass exposing a
differentiable path from raw waveform → features → teacher-forced loss.
The generic wrapper (`robust_speech/models/hf.py` / `sb_hf_binding.py`)
assumes a plain `model(input_values, labels=...)` call, which doesn't fit
granite/Qwen3-ASR (chat/multimodal-LLM interface: audio scattered into a
text prompt) or cohere (bespoke decoder prompt + non-differentiable shipped
feature extractor) — so each got a dedicated wrapper, following the
existing `models/canary.py` precedent for non-generic models.

### `openai/whisper-large-v3`
Small fix only — runs through the *existing* generic wrapper under the
**main `srb-venv`** (not `srb-venv-new`; whisper-large-v3 loads fine under
`transformers==4.34.0`, same as v2). `robust_speech/models/modules/whisper_feat.py`'s
`mel_filters()` hardcoded `assert n_mels == 80`; v3 uses 128 mel bins. Fixed
to accept both, plus a new `model_configs/hf_models/whisper-large-v3.yaml`
(clone of the v2 config) and an `en_models` entry in
`run_speech_robust_bench_adv.py`.

### `granite-speech-4.1-2b`, `Qwen3-ASR-1.7B-hf`, `cohere-transcribe-03-2026`
Each needs: a differentiable feature-extraction module (the shipped
extractors wrap their real, otherwise-differentiable math in `torch.no_grad()`
or force a numpy round-trip — bypassed with hand-written torch-only
equivalents in `robust_speech/models/modules/{granite_feat,qwen3_asr_feat}.py`,
or for cohere, a fresh instance of the checkpoint's own `FilterbankFeatures`
class with `use_grads=True`, grabbed dynamically via
`sys.modules[type(feature_extractor).__module__]` so it's guaranteed
numerically identical to the shipped one), a `BaseASR` subclass
(`robust_speech/models/{granite_speech,qwen3_asr,cohere_transcribe}.py`),
a `model_configs/hf_models/*.yaml` hparams file, and an `attack_configs/LibriSpeech/{pgd,universal}/*.yaml`
pair (same wrapper class referenced by both, unlike the generic path's
pgd/universal class split). All three run under **`srb-venv-new`**.

Key implementation notes:
- `robust_speech` attack configs always use `batch_size: 1` — all three
  wrappers are written single-utterance-only (no padding/masking logic),
  matching `canary.py`'s existing convention.
- Granite/Qwen3-ASR's `forward(labels=...)` shifts internally (pass full
  unshifted `input_ids`/`labels`); cohere's does **not** shift internally —
  pre-shift by hand like `canary.py` does.
- Granite's `input_features_mask` masks the model's *post-encoder* audio-token
  embeddings, not raw feature frames — at batch_size=1 there's no padding,
  so it's safe to omit entirely. Qwen3-ASR's `input_features_mask` is the
  opposite: a required raw-frame-level mask matching the *un-transposed*
  `(batch, num_mel_bins, padded_frames)` feature shape the encoder expects.
- Model dtype isn't always fp32 by default (`AutoModelForSpeechSeq2Seq.from_pretrained`
  loaded Qwen3-ASR in bf16) — cast features to `next(model.parameters()).dtype`
  before the forward call.
- Cohere's `model.generate()` (its own remote-code override) calls
  `super().generate(...)`, which errors under transformers 5.15's
  `PreTrainedModel`/`GenerationMixin` MRO — a bug in the checkpoint's bundled
  code, not fixable here. `eval_forward` does plain greedy decoding by hand
  instead (repeated `forward()` calls, no KV-cache reuse) purely for WER/CER
  reporting text; the attack itself only needs `train_attack_forward`, which
  doesn't call `.generate()` at all.
- Cohere's `decoder_start_token_id`/`eos_token_id` aren't reliably present on
  `model.generation_config`/`model.config` in all loading contexts for this
  checkpoint — load a fresh `GenerationConfig.from_pretrained(repo_id, trust_remote_code=True)`
  instead of trusting the loaded model instance's own attributes.

### Additional torchaudio/speechbrain version-drift fixes (`robust_speech/__init__.py`)
`speechbrain==1.0.0` was written against an older `torchaudio` API surface
than what's installed in `srb-venv-new` (`torchaudio==2.11.0+cu128`).
`robust_speech/robust_speech/__init__.py` now shims, at import time, three
functions `torchaudio>=2.9` removed: `list_audio_backends()` (used only for
a diagnostic log message), the `torchaudio.io` module (only its
`StreamReader`/`StreamWriter` names are referenced, in a type annotation on
a class robust_speech never uses), and `info()` (used by `robust_speech`'s
own `data/dataio.py` for sample-rate/channel-count — shimmed via
`soundfile.info()`, already a dependency). All three are no-ops wherever the
real API still exists (e.g. the main `srb-venv`'s older `torchaudio`), so
this is safe in both environments.

One import-order fix was also needed: `recipes/evaluate.py` and
`recipes/fit_attacker.py` used to `import speechbrain` before
`import robust_speech` — since `robust_speech`'s `__init__.py` is what
applies the above patches, that ordering let `speechbrain` crash first.
Both scripts now `import robust_speech` first.

### `speechbrain`/`robust_speech` installed into `srb-venv-new`
Needed since granite/Qwen3-ASR/cohere require `srb-venv-new`'s newer
`transformers`, but `speechbrain`/`robust_speech` were previously only
installed in the main `srb-venv`. Installed cleanly with no dependency
conflicts (confirmed via `pip check` and a full non-adversarial regression
run of all 4 models):
```bash
source /workspace/pod-setup/env.sh && source /workspace/pod-setup/env_new_models.sh
pip install speechbrain==1.0.0
pip install -e /workspace/speech_robust_bench/robust_speech --no-build-isolation
```
`--no-build-isolation` is required — `robust_speech`'s `audlib` git
dependency's legacy `setup.py` needs `pkg_resources`, which pip's isolated
build environment doesn't provide by default in this setuptools version.

### Verification performed
Real (not synthetic) PGD attacks via `evaluate.py`, 5 iterations (not the
full 100) at SNR=10, on 3 real utterances, using the model's own hand-built
CSV — both granite/qwen3-asr/cohere all completed and showed dramatic,
real accuracy degradation vs. their (already-known-good) clean baselines:

| Model | Clean CER | Adversarial CER (5/100 PGD iters, SNR=10) |
|---|---|---|
| `granite-speech-4.1-2b` | 0.00% | 52.60% |
| `Qwen3-ASR-1.7B-hf` | 0.00% | 66.47% |
| `cohere-transcribe-03-2026` | 3.47% | 57.23% |

A universal-attack smoke test (`fit_attacker.py`, 1 epoch / 3 iterations
instead of the full 10/20, on granite) also completed end-to-end without
error and wrote a checkpoint — too few iterations to show measurable
degradation (expected), but confirms the training/checkpointing pipeline
works.

Exact commands used (granite shown; swap the yaml filename and
`--model_name`-equivalent for qwen3-asr-1.7b-hf / cohere-transcribe-03-2026):
```bash
source /workspace/pod-setup/env.sh && source /workspace/pod-setup/env_new_models.sh
cd /workspace/speech_robust_bench/robust_speech/recipes

# small sanity run (5 iters instead of 100)
python evaluate.py attack_configs/LibriSpeech/pgd/granite-speech-4.1-2b.yaml \
  --root=/workspace/srb_root/robust_speech_data_root \
  --snr=10 --nb_iter=5 --dataset LibriSpeech --data_csv_name test-clean

# full run, one SNR level (repeat for snr=40,30,20,10, or use
# run_speech_robust_bench_adv.py to sweep all of them automatically)
python evaluate.py attack_configs/LibriSpeech/pgd/granite-speech-4.1-2b.yaml \
  --root=/workspace/srb_root/robust_speech_data_root \
  --snr=10 --dataset LibriSpeech --data_csv_name test-clean   # nb_iter defaults to 100

# universal attack (small sanity run: 1 epoch / 3 iters instead of 10/20)
python fit_attacker.py attack_configs/LibriSpeech/universal/granite-speech-4.1-2b.yaml \
  --root=/workspace/srb_root/robust_speech_data_root \
  --snr=10 --nb_epochs=1 --nb_iter=3 --dataset LibriSpeech --data_csv_name dev-clean
```
`run_speech_robust_bench_adv.py --models ibm-granite/granite-speech-4.1-2b Qwen/Qwen3-ASR-1.7B-hf CohereLabs/cohere-transcribe-03-2026 --dataset LibriSpeech --attack_type pgd` (or `--attack_type universal`) sweeps all 4 SNR levels for the given models automatically — run under `env_new_models.sh` for these 3; `openai/whisper-large-v3` needs a **separate** invocation under plain `srb-venv` (no `env_new_models.sh`), since `run_speech_robust_bench_adv.py` itself doesn't switch venvs mid-run.

**Before trusting a full-scale run**, do the same per-model gradient-flow
sanity check used during development: build one real utterance batch,
`requires_grad_()` a `delta` tensor added to the (detached) waveform
(mirroring `pgd_loop`'s actual construction — not `requires_grad_()` on the
raw waveform directly, which trips leaf-tensor restrictions some feature
extractors' internals hit but real attacks never do), run
`text_to_tokens`→`wav_to_feats`→`train_attack_forward`, call
`loss.mean().backward()`, and confirm `delta.grad` is non-`None` and
non-all-zero.

## Adversarial eval data: subsampled LibriSpeech set

Full-scale PGD (100 iterations × 4 SNR levels) and universal (10 epochs ×
500 training utterances × 4 SNR levels) attacks over the complete
2620-utterance LibriSpeech `test-clean` set are expensive — order of
multiple GPU-days per model. The working set actually used for these 4
models is a **subsample**: 200 utterances (`test-clean`, for PGD) and 500
utterances (`dev-clean`, for universal-attack training), built with
`make_subsample_csv.py` (repo root):

```bash
source /workspace/pod-setup/env.sh   # main srb-venv, NOT env_new_models.sh — see below
cd /workspace/speech_robust_bench

mkdir -p "$SRB_ROOT/robust_speech_data_root/data/LibriSpeech/test-clean" \
         "$SRB_ROOT/robust_speech_data_root/data/LibriSpeech/dev-clean"

python make_subsample_csv.py --split test.clean --dirname test-clean --n 200
python make_subsample_csv.py --split validation.clean --dirname dev-clean --n 500
```

This streams the requested number of examples directly from the
`librispeech_asr` HF dataset (no full-corpus download — see the script's
docstring for why plain `load_dataset(...)` without `streaming=True` is
much more expensive here: it resolves to the "all" config and pulls every
split, tens of GB, before you can slice anything out of it) and writes real
flac files + a matching CSV directly into
`$SRB_ROOT/robust_speech_data_root/data/LibriSpeech/{test-clean,dev-clean}/`
— exactly the layout/schema `attack_configs/LibriSpeech/*/*.yaml`'s
`test_csv`/`train_csv` (with `skip_prep: True`) already expect, so no
further wiring is needed; just run `evaluate.py`/`fit_attacker.py` as
normal against `--data_csv_name test-clean` / `dev-clean`.

**Must run under the main `srb-venv`** (`source env.sh` only, not
`env_new_models.sh`) — under `srb-venv-new`'s newer `huggingface_hub`,
streaming `librispeech_asr` (a root-level, non-namespaced dataset id) fails
with `HfUriError: Repository id must be 'namespace/name'` during internal
URI resolution. This only affects *building* the subsample; running the
actual attacks against granite/Qwen3-ASR/cohere afterward still needs
`env_new_models.sh` as usual.

If you hit `FileNotFoundError: ... .incomplete/dataset_info.json` from
`datasets`, a previous interrupted (non-streaming) `librispeech_asr`
download left a stale cache entry — remove
`$HF_HOME/datasets/librispeech_asr/` and retry.
