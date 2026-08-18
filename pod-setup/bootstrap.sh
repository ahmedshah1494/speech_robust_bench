#!/usr/bin/env bash
# Re-create everything that lives on the ephemeral overlay (/) after a pod stop/start.
# Idempotent -- safe to run any number of times.
#
#   bash /workspace/pod-setup/bootstrap.sh
#
# What is ephemeral on this pod: /root (home), /usr (apt packages), /tmp.
# What persists: /workspace (network volume).

set -uo pipefail

SETUP_DIR=/workspace/pod-setup
PERSIST_HOME=/workspace/persist/home
log() { printf '  %s\n' "$*"; }

echo "==> speech_robust_bench pod bootstrap"

# ---------------------------------------------------------------- 1. system deps
# torchaudio's sox effects need libsox.so; datasets/ffmpeg decoding needs ffmpeg.
if command -v ffmpeg >/dev/null 2>&1 && [ -e /usr/lib/x86_64-linux-gnu/libsox.so ]; then
  log "system deps: ffmpeg + libsox already present"
else
  log "system deps: installing cached debs from $SETUP_DIR/debs"
  if ls "$SETUP_DIR"/debs/*.deb >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive dpkg -i "$SETUP_DIR"/debs/*.deb >/tmp/bootstrap-dpkg.log 2>&1 \
      || log "  dpkg reported issues, see /tmp/bootstrap-dpkg.log"
  else
    log "  no cached debs; falling back to apt-get (needs network)"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg sox libsox-dev libsox-fmt-all
  fi
  command -v ffmpeg >/dev/null 2>&1 && log "  ffmpeg ok" || log "  WARNING: ffmpeg missing"
  [ -e /usr/lib/x86_64-linux-gnu/libsox.so ] && log "  libsox ok" || log "  WARNING: libsox missing"
fi

# ------------------------------------------------------- 2. claude code (persisted)
mkdir -p "$PERSIST_HOME/.local/share" /root/.local/share /root/.local/bin

link_into_root() {   # $1 = path under $PERSIST_HOME, $2 = path under /root
  local src="$PERSIST_HOME/$1" dst="$2"
  [ -e "$src" ] || return 0
  if [ -L "$dst" ] && [ "$(readlink -f "$dst")" = "$(readlink -f "$src")" ]; then
    return 0
  fi
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then
    # a fresh container may ship its own copy -- keep it out of the way
    mv "$dst" "$dst.container-$(date +%s)"
  fi
  rm -f "$dst"
  ln -s "$src" "$dst"
  log "linked $dst -> $src"
}

link_into_root .claude            /root/.claude
link_into_root .claude.json       /root/.claude.json
link_into_root .local/share/claude /root/.local/share/claude

# claude launcher: point at the newest installed version binary
latest_claude=$(ls -1t "$PERSIST_HOME"/.local/share/claude/versions/* 2>/dev/null | head -1)
if [ -n "${latest_claude:-}" ]; then
  ln -sfn "$latest_claude" /root/.local/bin/claude
  log "claude cli: $(basename "$latest_claude")"
else
  log "WARNING: no claude binary in $PERSIST_HOME/.local/share/claude/versions"
fi

# ------------------------------------------------------------------- 3. shell env
if ! grep -q 'pod-setup/env.sh' /root/.bashrc 2>/dev/null; then
  cat >> /root/.bashrc <<'EOF'

# --- speech_robust_bench pod setup ---
export PATH="$HOME/.local/bin:$PATH"
[ -f /workspace/pod-setup/env.sh ] && source /workspace/pod-setup/env.sh
EOF
  log "added env.sh + ~/.local/bin to /root/.bashrc"
else
  log "/root/.bashrc already sources env.sh"
fi
# shellcheck disable=SC1091
source "$SETUP_DIR/env.sh"

# ---------------------------------------------------------------- 4. sanity checks
echo "==> checks"
log "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd'; ')"
log "/ free:   $(df -h / | awk 'NR==2{print $4}')  (ephemeral)"
log "venv:     $("$SRB_VENV/bin/python" -V 2>&1)"
"$SRB_VENV/bin/python" - <<'PY' 2>/dev/null | sed 's/^/  /'
import torch, torchaudio
print(f"torch:    {torch.__version__}  cuda={torch.cuda.is_available()}")
try:
    import torchaudio.sox_effects as se
    se.apply_effects_tensor(torch.zeros(1, 16000), 16000, [["speed", "1.1"], ["rate", "16000"]])
    print("sox:      ok")
except Exception as e:
    print(f"sox:      FAILED ({e})")
PY
log "HF_HOME:  $HF_HOME ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1) cached)"
log "SRB_ROOT: $SRB_ROOT"

cat <<EOF

==> ready.  Resume a claude session with:
      cd /workspace && claude --continue     # most recent session
      cd /workspace && claude --resume       # pick from list
    Run evals with:
      cd \$SRB_REPO && python evaluate_single.py --help          # single model/dataset/augmentation
      cd \$SRB_REPO && python run_speech_robust_bench.py --help  # multi-model/perturbation sweep
EOF
