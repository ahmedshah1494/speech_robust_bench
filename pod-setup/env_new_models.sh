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
