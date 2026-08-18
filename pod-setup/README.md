# Pod setup (persisted copy)

This directory is a git-tracked backup of the pod-level bootstrap scripts
that live at `/workspace/pod-setup/` on the dev pod. That location is a
plain directory on the pod's persistent `/workspace` volume — **it is not a
git repo**, so anything there is lost if the pod is *terminated* (not just
stopped/restarted). These copies are what let a fresh pod (or a fresh clone
of this repo on any machine) reconstruct the same environment.

## Setting up a fresh pod

```bash
mkdir -p /workspace/pod-setup
cp pod-setup/bootstrap.sh pod-setup/env.sh pod-setup/env_new_models.sh /workspace/pod-setup/
echo 'export HF_TOKEN=hf_...'"'"'your real token'"'"'' > /workspace/pod-setup/env.local.sh   # not committed, see env.sh
bash /workspace/pod-setup/bootstrap.sh
```

`bootstrap.sh` installs ffmpeg/libsox (via cached `.deb`s if present under
`pod-setup/debs/`, else `apt-get` — the `.deb` cache itself isn't committed
here, ~100MB of binaries, apt-get is a fine fallback), wires `env.sh` into
`~/.bashrc`, and runs sanity checks. It also links in a persisted Claude
Code install (`/workspace/persist/home/.claude/...`) — that part is
specific to this pod's Claude Code session persistence, not required for
the ASR eval work itself; harmless to leave in, safe to ignore/remove if
setting up a pod without that persistence layer.

`env.sh` still needs the **main venv** (`$SRB_VENV`, i.e.
`/workspace/srb-venv`) to actually exist — that's not something bootstrap.sh
creates; follow the main `README.md`'s Installation section
(`pip install -r requirements.txt`, `pip install -e robust_speech`, `pip
install -e deepspeech.pytorch`, all inside a venv created at that path) to
build it.

For the second venv needed by `granite-speech`/`Qwen3-ASR`/`cohere-transcribe`
(`$SRB_VENV_NEW`, `/workspace/srb-venv-new`) and its adversarial-attack
support, see [`docs/new_asr_models_setup.md`](../docs/new_asr_models_setup.md)
— that doc has the full rebuild-from-scratch steps (torch/transformers
versions, the FFmpeg7/torchcodec fixes, the `speechbrain`/`robust_speech`
install into the second venv).
