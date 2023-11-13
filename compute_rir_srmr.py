from datasets import load_dataset, Audio
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
import os
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed  
from srmrpy.srmr import srmr


def apply_rir(args):
    x, (rir_raw, sample_rate) = args
    rir = rir_raw[:, int(sample_rate * .01) : ]
    rir = rir / torch.norm(rir, p=2)
    rir = rir[0].reshape(-1).to(x.device)
    x_ = torchaudio.functional.fftconvolve(x, rir)
    return x_

def apply_rirs(x, rirs):
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    # x_out = Pool(cpu_count()).map(apply_rir, [(i, x, rir) for i,rir in enumerate(rirs)])
    x_out = Parallel(n_jobs=5)(delayed(apply_rir)((x, rir)) for rir in tqdm(rirs))
    # x_out = torch.stack(x_out, 0)
    return x_out

def compute_srmr(audio):
    def _srmr(x):
        return srmr(x, 16000)[0]
    audio = [a.cpu().detach().numpy() for a in audio]
    srmrs = Parallel(n_jobs=5)(delayed(_srmr)(a) for a in tqdm(audio))
    return np.array(srmrs)

rir_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/RIRS_NOISES/simulated_rirs'
rir_files = []
print('listing rir files')
for root, dirs, files in tqdm(os.walk(rir_dir)):
    for name in files:
        if name.endswith('wav'):
            rir_file = os.path.join(rir_dir, root, name)
            rir_files.append(rir_file)
# rir_files = rir_files[:4]
print(f'found {len(rir_files)} rir files')
# rirs = Pool(cpu_count()).map(torchaudio.load, rir_files)
rirs = Parallel(n_jobs=4)(delayed(torchaudio.load)(rir_file) for rir_file in tqdm(rir_files))
print(f'loaded {len(rirs)} rirs')

print('loading dataset...')
dataset = load_dataset("librispeech_asr", split='test.clean')
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
rir_srmrs = []

for i in tqdm(np.random.choice(len(dataset), 10)):
    i = int(i)
    audio, text = dataset[i]['audio']['array'], dataset[i]['text']
    print(f'generating RIRs for index {i}: {text}')
    new_audio = apply_rirs(audio, rirs)
    if isinstance(new_audio, Tensor):
        new_audio = new_audio.cpu().detach().numpy()

    # if not os.path.exists('examples/rirs'):
    #     os.makedirs('examples/rirs')
    # for j, na in enumerate(new_audio[:5]):
    #     sf.write(f"examples/rirs/{dataset[i]['id']}-{j}.wav", na, samplerate=16000)

    srmrs = compute_srmr(new_audio)
    rir_srmrs.append(srmrs)

rir_srmrs = np.stack(rir_srmrs, 0).mean(0)
data = []
for fn, srmr in zip(rir_files, rir_srmrs):
    data.append({
        'filename': fn,
        'srmr': srmr
    })
df = pd.DataFrame(data)
df.to_csv('rir_srmr.csv')