from datasets import load_dataset,Audio
import os
import tqdm
import soundfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="librispeech_asr", help='Name for dataset to load from huggingface hub. default: librispeech_asr.')
parser.add_argument('--subset', default=None, help='Subset of the dataset to use. default: None')
parser.add_argument('--split', default='test.clean', help='Split of the dataset to use. default: test.clean')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the data. default: outputs')
args = parser.parse_args()
if args.subset is not None:
    ds = load_dataset(args.dataset, args.subset, split=args.split)
else:
    ds = load_dataset(args.dataset, split=args.split)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

for x in tqdm.tqdm(ds):
    uid = x['id']
    uid = uid.replace("_","-")
    wav = x['audio']['array']

    spkid, chid, _ = uid.split('-')
    odir = f'{args.output_dir}/{spkid}/{chid}/'
    os.makedirs(odir, exist_ok=True)
    soundfile.write(f'{odir}/{uid}.flac', wav, 16000)
    with open(f'{odir}/{uid}.trans.txt', 'w') as f:
        f.write(f"{uid} {x['text']}")
