import time
import torch
import torchaudio.transforms as audio_transforms
from datasets import load_dataset
import torchaudio
from torchaudio import functional as F
import numpy as np
import pandas as pd
import os

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
        return F.add_noise(x, d, snr)

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
        rng = torch.Generator(x.device)
        rng = rng.manual_seed(rng.seed())
        d = torch.empty_like(x).normal_(0, 1, generator=rng)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        return F.add_noise(x, d, snr)

class EnvNoise(torch.nn.Module):
    seeds = [4117371, 7124264, 1832224, 8042969, 4454604, 5347561, 7059465,
                3774329, 1412644, 1519183, 6969162, 7885564, 3707167, 5816443,
                9477077, 9822365, 7482569, 7792808, 9120101, 5467473]
    
    def __init__(self, snr, noise_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/MS-SNSD/noise_test') -> None:
        super().__init__()
        self.snr = snr
        self.noise_dir = noise_dir
        self.noise_files = [x for x in os.listdir(noise_dir) if x.endswith('.wav')]
        # seed = self.seeds[int(snr % len(self.seeds))]
        seed = time.time_ns()
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
    
class EnvNoiseDeterministic(EnvNoise):
    def __init__(self, snr, noise_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/MS-SNSD/noise_test') -> None:
        super().__init__(snr, noise_dir)
        self.noise_files = self.rng.choice(self.noise_files, 1)

class UniversalAdversarialPerturbation(torch.nn.Module):
    def __init__(self, snr, path_to_noise='/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/deepspeech-1/1002/CKPT+2023-11-21+04-27-45+00/delta.ckpt') -> None:
        super().__init__()
        self.perturbation = torch.load(path_to_noise)['tensor']
        self.snr = snr
    
    def __repr__(self):
        return f'UniversalAdversarialPerturbation(snr={self.snr})'
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        noise_raw = self.perturbation
        noise = noise_raw[..., :xlen]
        while noise.shape[-1] < xlen:
            noise = torch.cat([noise, noise], -1)
            noise = noise[..., :xlen]
        noise = noise.reshape(-1).to(x.device)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        x_ = F.add_noise(x, noise, snr)
        return x_

class RIR(torch.nn.Module):
    seed = 9983137
    def __init__(self, sev, rir_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/RIRS_NOISES/simulated_rirs', rir_snr_file='rir_snr.csv') -> None:
        super().__init__()
        assert sev <= 4
        self.rir_dir = rir_dir
        # self.rir_files = [x for x in os.listdir(rir_dir) if x.endswith('.wav')]
        rir_files = []
        for root, dirs, files in os.walk(rir_dir):
            for name in files:
                if name.endswith('wav'):
                    rir_files.append(os.path.join(root, name))
        rir_snr_df = pd.read_csv(rir_snr_file)
        unique_snrs = rir_snr_df['snr'].unique()
        snr_sevs = np.linspace(unique_snrs.max(), unique_snrs.min(), 5)
        print(snr_sevs, snr_sevs[sev])
        if sev == 0:
            filtered_rows = rir_snr_df[rir_snr_df['snr'] >= snr_sevs[sev]]
        elif sev == 1:
            filtered_rows = rir_snr_df[(snr_sevs[sev] <= rir_snr_df['snr']) & (rir_snr_df['snr'] <= snr_sevs[sev-1])]
        else:
            filtered_rows = rir_snr_df[(snr_sevs[sev] <= rir_snr_df['snr']) & (rir_snr_df['snr'] < snr_sevs[sev-1])]
        self.rir_files = filtered_rows['filename'].values
        print(filtered_rows["snr"].min(), filtered_rows["snr"].max())
        print(f'using {len(self.rir_files)} rirs with average SNR={filtered_rows["snr"].mean()}')
        # {x['filename']: x['snr'] for x in pd.read_csv(rir_snr_file).to_dict('records')}
        # self.rng = np.random.default_rng(self.seed)
        seed = time.time_ns()
        self.rng = np.random.default_rng(seed)

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
    
class ResamplingNoise(torch.nn.Module):
    def __init__(self, factor, orig_freq=16000) -> None:
        super().__init__()
        print(f'factor={factor}', f'orig_freq={orig_freq}', f'new_freq={int(factor*orig_freq)}')
        self.ds = torchaudio.transforms.Resample(int(orig_freq*factor), orig_freq)
        self.us = torchaudio.transforms.Resample(orig_freq, int(orig_freq*factor))
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x_ = self.ds(self.us(x))
        return x_


class VoiceConversion(torch.nn.Module):
    seed = 9983137
    def __init__(self, accents) -> None:
        super().__init__()
        self.accents = accents
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.ds = ds.filter(lambda x: x['filename'].split('_')[2] in accents)
        self.rng = np.random.default_rng(self.seed)
    
    def __repr__(self):
        return f"VoiceConversion({self.accents})"
    
    def forward(self, speech, text, *args, **kwargs):
        if len(self.ds) == 0:
            return speech
        voice_idxs = self.rng.choice(len(self.ds))
        device = self.parameters().__next__().device
        speaker_embeddings = torch.tensor(self.ds[voice_idxs]["xvector"]).to(device).unsqueeze(0)
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings, vocoder=self.vocoder)
        return speech
    
class Compose(torch.nn.Module):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
    
    def __repr__(self):
        return f"Compose({self.transforms})"
    
    def forward(self, speech, *args, **kwargs):
        for t in self.transforms:
            if isinstance(t, VoiceConversion):
                speech = t(speech, args[0])
            else:
                x = t(speech)
        return x