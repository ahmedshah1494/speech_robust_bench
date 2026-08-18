# Speech Robust Bench pod environment.
# Sourced by ~/.bashrc (see bootstrap.sh). Safe to source repeatedly.

# --- persistent locations (survive pod stop/start; lost only on terminate) ---
export SRB_REPO=/workspace/speech_robust_bench
export SRB_VENV=/workspace/srb-venv
export SRB_ROOT=/workspace/srb_root                 # deepspeech_ckps/, MS-SNSD/, ...
export SRB_OUTPUTS=/workspace/srb_outputs           # eval result .tsv files
export SRB_DATA_ROOT=/workspace/srb_data            # robust_speech data root (adversarial evals)

# --- caches: keep everything off the ephemeral 128G overlay on / ---
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME=/workspace/.cache/torch
export PIP_CACHE_DIR=/workspace/.cache/pip
export NEMO_CACHE_DIR=/workspace/.cache/nemo
export SPEECHBRAIN_CACHE_DIR=/workspace/.cache/speechbrain

# --- misc ---
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

# HF Hub auth (raises anonymous-IP rate limits from HF API on dataset downloads).
# NOT committed with a real value - this file is git-tracked, a live token here
# would be a leaked credential. Set your own token before sourcing, e.g. in a
# separate untracked file:
#   echo 'export HF_TOKEN=hf_...' > /workspace/pod-setup/env.local.sh
# and source that after this file, or just export it directly in your shell.
: "${HF_TOKEN:=}"
if [ -z "$HF_TOKEN" ] && [ -f /workspace/pod-setup/env.local.sh ]; then
  # shellcheck disable=SC1091
  source /workspace/pod-setup/env.local.sh
fi

# venv first on PATH: `python`, `pip` are the SRB venv's
case ":$PATH:" in
  *":$SRB_VENV/bin:"*) ;;
  *) export PATH="$SRB_VENV/bin:$PATH" ;;
esac
