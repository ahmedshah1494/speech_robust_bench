from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC, pipeline, WhisperProcessor
from transformers.pipelines.pt_utils import KeyDataset
import torch
import torchaudio.transforms as audio_transforms
import torchaudio
from torchaudio import functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import os
from copy import deepcopy

class UniformNoise(torch.nn.Module):
    def __init__(self, snr) -> None:
        super().__init__()
        self.snr = snr
    
    def __repr__(self):
        return f"UniformNoise({self.snr} dB)"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        # if not isinstance(xlen, torch.Tensor):
        #     xlen = torch.LongTensor()
        d = torch.empty_like(x).uniform_(-1, 1)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        return F.add_noise(x, d, snr).numpy()

class GaussianNoise(torch.nn.Module):
    def __init__(self, snr) -> None:
        super().__init__()
        self.snr = snr
    
    def __repr__(self):
        return f"GaussianNoise({self.snr} dB)"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        # if not isinstance(xlen, torch.Tensor):
        #     xlen = torch.LongTensor()
        d = torch.empty_like(x).normal_(0, 1)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        return F.add_noise(x, d, snr).numpy()

class EnvNoise(torch.nn.Module):
    seeds = [4117371, 7124264, 1832224, 8042969, 4454604, 5347561, 7059465,
                3774329, 1412644, 1519183, 6969162, 7885564, 3707167, 5816443,
                9477077, 9822365, 7482569, 7792808, 9120101, 5467473]
    
    def __init__(self, snr, noise_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/MS-SNSD/noise_test') -> None:
        super().__init__()
        self.snr = snr
        self.noise_dir = noise_dir
        self.noise_files = [x for x in os.listdir(noise_dir) if x.endswith('.wav')]
        seed = self.seeds[int(snr % len(self.seeds))]
        self.rng = np.random.default_rng(seed)
    
    def __repr__(self):
        return f"EnvNoise({self.snr} dB)"

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        noise_file = os.path.join(self.noise_dir, self.noise_files[self.rng.choice(len(self.noise_files))])
        noise_raw, sample_rate = torchaudio.load(noise_file)
        noise = noise_raw[..., :xlen]
        while noise.shape[-1] < xlen:
            noise = torch.cat([noise, noise], -1)
            noise = noise[..., :xlen]
        noise = noise[0].reshape(-1).to(x.device)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        x_ = F.add_noise(x, noise, snr)
        return x_

class RIR(torch.nn.Module):
    seed = 9983137
    def __init__(self, *args, rir_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/RIRS_NOISES/simulated_rirs') -> None:
        super().__init__()
        self.rir_dir = rir_dir
        # self.rir_files = [x for x in os.listdir(rir_dir) if x.endswith('.wav')]
        self.rir_files = []
        for root, dirs, files in os.walk(rir_dir):
            for name in files:
                if name.endswith('wav'):
                    self.rir_files.append(os.path.join(root, name))
        self.rng = np.random.default_rng(self.seed)

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        def get_random_rir():
            rir_file = os.path.join(self.rir_dir, self.rir_files[self.rng.choice(len(self.rir_files))])
            rir_raw, sample_rate = torchaudio.load(rir_file)
            rir = rir_raw[:, int(sample_rate * .01) : ]
            return rir
        rir = get_random_rir()
        rir = rir / torch.norm(rir, p=2)
        rir = rir[0].reshape(-1).to(x.device)
        x_ = torchaudio.functional.fftconvolve(x, rir)
        return x_

class Speed(torch.nn.Module):
    def __init__(self, factor, orig_freq=16000) -> None:
        super().__init__()
        self.factor = factor
        self.transform = audio_transforms.Speed(orig_freq, factor)
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        x_, new_lens =  self.transform.to(x.device)(x)
        return x_
    
class Pitch(torch.nn.Module):
    def __init__(self, shift, orig_freq=16000) -> None:
        super().__init__()
        self.shift = shift
        self.transform = audio_transforms.PitchShift(orig_freq, shift)
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        x_ =  self.transform.to(x.device)(x)
        return x_

NOISE_SNRS = [30, 10, 5, 1, -10]
SPEEDUP_FACTORS = [1, 1.25, 1.5, 1.75, 2]
SLOWDOWN_FACTORS = [1, 0.875, 0.75, 0.625, 0.5]
PITCH_UP_STEPS = [0, 3, 6, 9, 12]
PITCH_DOWN_STEPS = [0, -3, -6, -9, -12]
AUGMENTATIONS = {
    'unoise': (UniformNoise, NOISE_SNRS),
    'gnoise': (GaussianNoise, NOISE_SNRS),
    'env_noise': (EnvNoise, NOISE_SNRS),
    'speedup': (Speed, SPEEDUP_FACTORS),
    'slowdown': (Speed, SLOWDOWN_FACTORS),
    'pitch_up': (Pitch, PITCH_UP_STEPS),
    'pitch_down': (Pitch, PITCH_DOWN_STEPS),
    'rir': (RIR, [None]),
}

parser = ArgumentParser()
parser.add_argument('--model_name', default="openai/whisper-small")
parser.add_argument('--dataset', default="librispeech_asr")
parser.add_argument('--split', default='test.clean')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--augmentation')
parser.add_argument('--severity')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--model_parallelism', action='store_true')
args = parser.parse_args()

if args.augmentation:
    aug, sev = args.augmentation.split(':', 1)
    sev = int(sev)
    assert sev <= 4
else:
    aug = None
    sev = 0
if aug in AUGMENTATIONS:
    fn, sev_args = AUGMENTATIONS[aug]
    transform = fn(sev_args[sev])
else:
    transform = lambda x: x
print(transform)
def transform_(batch):
    for audio in batch['audio']:
        audio['array'] = transform(audio['array'])
    return batch

dataset = load_dataset(args.dataset, split=args.split)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
print(dataset[0])
dataset = dataset.map(transform_, batched=True, batch_size=128, num_proc=4, load_from_cache_file=False)
dataset = dataset.with_format('np')

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

if args.model_parallelism: 
    device_kwargs = {'device_map': 'auto'}
else:
    device_kwargs = {'device': 'cuda:0'}
pipe = pipeline("automatic-speech-recognition", model=args.model_name, batch_size=args.batch_size, torch_dtype=torch.float16, **device_kwargs)

output_rows = []
t = tqdm(zip(pipe(KeyDataset(dataset, "audio")), dataset))
for out, inp in t:
    hyp = out['text'].upper()
    ref = inp['text'].upper()
    wer = wer_metric.compute(references=[ref], predictions=[hyp])
    cer = cer_metric.compute(references=[ref], predictions=[hyp])
    r = {
        'reference': ref,
        'prediction': hyp,
        'wer': wer,
        'cer': cer
    }
    output_rows.append(r)
    t.set_postfix(wer=wer, cer=cer)

odir = f'{args.output_dir}/{args.model_name.split("/")[-1]}/{args.dataset}'
if not os.path.exists(odir):
    os.makedirs(odir)

df = pd.DataFrame(output_rows)
df.to_csv(f'{odir}/{aug}-{sev}.tsv', sep='\t')