from datasets import load_dataset, Audio
import soundfile as sf
import torch
import torchaudio
import os
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from joblib import Parallel, delayed
import speechmetrics

def apply_rir(args):
    x, (rir_raw, sample_rate) = args
    rir = rir_raw[:, int(sample_rate * .01) : ]
    rir = rir / torch.norm(rir, p=2)
    rir = rir[0].reshape(-1).to(x.device)
    x_ = torchaudio.functional.fftconvolve(x, rir)
    x_ = x_[:x.shape[0]]
    return x_

def apply_rirs(x, rirs):
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    # x_out = Pool(cpu_count()).map(apply_rir, [(i, x, rir) for i,rir in enumerate(rirs)])
    x_out = Parallel(n_jobs=5)(delayed(apply_rir)((x, rir)) for rir in tqdm(rirs))
    # pad x_out to have same length as x
    x_out = [torch.nn.functional.pad(x_, (0, x.shape[0] - x_.shape[0])) for x_ in x_out]
    x_out = torch.stack(x_out, 0)
    return x_out

def compute_snr(clean, noisy):
    snrs = []
    for _noisy in noisy:
        noise = _noisy - clean
        snr = 10 * np.log10((clean**2).sum() / (noise**2).sum())
        snrs.append(snr)
    snrs = np.array(snrs)
    return snrs

def load_audio(path):
    audio, sr = sf.read(path)
    audio = torch.FloatTensor(audio)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio[:, 0].unsqueeze(0)
    else:
        raise ValueError(f'Invalid audio shape {audio.shape}')
    return audio, sr

rir_dir=f'{os.environ["SRB_ROOT"]}/RIRS_NOISES/real_rirs_isotropic_noises'
rir_files = []
print('listing rir files')
for root, dirs, files in tqdm(os.walk(rir_dir)):
    for name in files:
        if name.endswith('wav') and ('simroom' not in name):
            rir_file = os.path.join(rir_dir, root, name)
            rir_files.append(rir_file)
# rir_files = rir_files[:10]
print(f'found {len(rir_files)} rir files')
# rirs = Pool(cpu_count()).map(torchaudio.load, rir_files)
rirs = Parallel(n_jobs=4)(delayed(load_audio)(rir_file) for rir_file in tqdm(rir_files))
print(f'loaded {len(rirs)} rirs')

print('loading dataset...')
dataset = load_dataset("librispeech_asr", split='test.clean')
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
rir_snrs = []
rir_srmrs = []

window_length = 5 # seconds
metric = speechmetrics.load(['srmr'], window_length)
for i in tqdm(np.random.choice(len(dataset), 10)):
    i = int(i)
    audio, text = dataset[i]['audio']['array'], dataset[i]['text']
    print(f'generating RIRs for index {i}: {text}')
    new_audio = apply_rirs(audio, rirs)
    if isinstance(new_audio, torch.Tensor):
        new_audio = new_audio.cpu().detach().numpy()

    # if not os.path.exists('examples/rirs'):
    #     os.makedirs('examples/rirs')
    # for j, na in enumerate(new_audio[:5]):
    #     sf.write(f"examples/rirs/{dataset[i]['id']}-{j}.wav", na, samplerate=16000)

    snrs = compute_snr(audio, new_audio)
    srmr = Parallel(n_jobs=4)(delayed(metric)(a, rate=16000) for a in tqdm(new_audio))
    srmr = [s['srmr'][0] for s in srmr]
    rir_snrs.append(snrs)
    rir_srmrs.extend(srmr)

rir_snrs = np.stack(rir_snrs, 0).mean(0)
data = []
for fn, snr, srmr in zip(rir_files, rir_snrs, rir_srmrs):
    data.append({
        'filename': fn,
        'snr': snr,
        'srmr': srmr
    })
df = pd.DataFrame(data)
df.to_csv('rir_snr.csv')