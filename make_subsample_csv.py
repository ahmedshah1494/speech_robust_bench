"""
Build a small, real (not synthetic) LibriSpeech-format CSV + flac directory
for robust_speech adversarial attack runs, by streaming N examples directly
from the librispeech_asr HF dataset - no full-corpus download needed.

`load_dataset("librispeech_asr", split=..., streaming=True)` reads only the
requested examples over the network rather than downloading whole shard
archives, which matters here: the plain (non-streaming) `load_dataset` call
resolves to the "all" config and pulls the ENTIRE LibriSpeech corpus (all
train/dev/test splits, tens of GB) before you can slice out just what you
need - streaming avoids that entirely.

Output matches the CSV schema robust_speech.data.librispeech.create_csv
produces (ID,duration,wav,spk_id,wrd), so it drops straight into the
`test_csv`/`train_csv` paths any attack_configs/LibriSpeech/*/*.yaml
already expects (with `skip_prep: True`, prepare_librispeech is never
invoked - this script's output IS the csv it would have built).

Usage:
    python make_subsample_csv.py --split test.clean --dirname test-clean --n 200
    python make_subsample_csv.py --split validation.clean --dirname dev-clean --n 500

Run under the main srb-venv (not srb-venv-new) - streaming librispeech_asr
via a script-based HF dataset needs the older datasets/huggingface_hub
versions pinned there; srb-venv-new's newer huggingface_hub errors on
librispeech_asr's root-level (non-namespaced) repo id during URI parsing.
"""
import argparse
import csv
import os

import soundfile
from datasets import Audio, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--split", required=True, help="HF split name, e.g. test.clean or validation.clean")
parser.add_argument("--dirname", required=True, help="LibriSpeech-style dir name, e.g. test-clean or dev-clean")
parser.add_argument("--n", type=int, required=True, help="Number of utterances to sample")
parser.add_argument(
    "--root",
    default=os.path.join(os.environ.get("SRB_ROOT", "/workspace/srb_root"), "robust_speech_data_root"),
    help="robust_speech data root, i.e. the --root passed to evaluate.py/fit_attacker.py. "
         "Default: $SRB_ROOT/robust_speech_data_root, matching README.md's adversarial-eval instructions.",
)
args = parser.parse_args()

data_dir = os.path.join(args.root, "data", "LibriSpeech", args.dirname)
csv_dir = os.path.join(args.root, "data", "LibriSpeech", "csv")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

print(f"Streaming librispeech_asr split={args.split}, n={args.n}...")
ds = load_dataset("librispeech_asr", split=args.split, streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

rows = []
for i, row in enumerate(ds):
    if i >= args.n:
        break
    wav = row["audio"]["array"]
    text = row["text"]
    uid = row["id"].replace("_", "-")
    spk_id = "-".join(uid.split("-")[0:2])
    odir = os.path.join(data_dir, *spk_id.split("-"))
    os.makedirs(odir, exist_ok=True)
    flac_path = os.path.join(odir, f"{uid}.flac")
    soundfile.write(flac_path, wav, 16000)
    duration = len(wav) / 16000
    rows.append([uid, str(duration), flac_path, spk_id, text])
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{args.n} written")

csv_path = os.path.join(csv_dir, f"{args.dirname}.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w.writerow(["ID", "duration", "wav", "spk_id", "wrd"])
    for r in rows:
        w.writerow(r)

print(f"DONE: {len(rows)} rows -> {csv_path}")
