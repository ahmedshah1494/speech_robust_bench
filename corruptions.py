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
    
    def __repr__(self):
        return f"EnvNoise({self.snr} dB)"

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        noise_file = os.path.join(self.noise_dir, self.noise_files[rng.choice(len(self.noise_files))])
        noise_raw, sample_rate = torchaudio.load(noise_file)
        noise = noise_raw[..., :xlen]
        while noise.shape[-1] < xlen:
            noise = torch.cat([noise, noise], -1)
            noise = noise[..., :xlen]
        noise = noise[0].reshape(-1).to(x.device)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        x_ = F.add_noise(x, noise, snr)
        return x_
    
class EnvNoiseESC50(EnvNoise):
    def __init__(self, snr) -> None:
        super().__init__(snr, '/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/ESC-50-master/audio')
    
    def __repr__(self):
        return f"EnvNoiseESC50({self.snr} dB)"
    
class EnvNoiseDeterministic(EnvNoise):
    def __init__(self, snr, noise_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/MS-SNSD/noise_test') -> None:
        super().__init__(snr, noise_dir)
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        self.noise_files = rng.choice(self.noise_files, 1)

class UniversalAdversarialPerturbation(torch.nn.Module):
    def __init__(self, snr, path_to_noise='/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/deepspeech-1/1002/CKPT+2023-11-21+04-27-45+00/delta.ckpt') -> None:
        super().__init__()
        self.perturbation = torch.load(path_to_noise)['tensor'].cpu()
        self.snr = snr
    
    def __repr__(self):
        return f'UniversalAdversarialPerturbation(snr={self.snr})'
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        noise_raw = self.perturbation.to(x.device)
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

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        def get_random_rir():
            rir_file = os.path.join(self.rir_dir, self.rir_files[rng.choice(len(self.rir_files))])
            rir_raw, sample_rate = torchaudio.load(rir_file)
            rir = rir_raw[:, int(sample_rate * .01) : ]
            return rir
        rir = get_random_rir()
        rir = rir / torch.norm(rir, p=2)
        rir = rir[0].reshape(-1).to(x.device)
        x_ = torchaudio.functional.fftconvolve(x, rir)
        return x_

class RIR_RoomSize(torch.nn.Module):
    def __init__(self, room_type, rir_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/RIRS_NOISES/simulated_rirs') -> None:
        super().__init__()
        self.rir_dir = rir_dir
        rir_files = []
        for root, dirs, files in os.walk(rir_dir):
            for name in files:
                if name.endswith('wav'):
                    if room_type in root:
                        rir_files.append(os.path.join(root, name))
        self.rir_files = rir_files
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        def get_random_rir():
            rir_file = os.path.join(self.rir_dir, self.rir_files[rng.choice(len(self.rir_files))])
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


class AbsVoiceConversion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, speech, text, *args, **kwargs):
        raise NotImplementedError
class VoiceConversion(AbsVoiceConversion):
    seed = 9983137
    def __init__(self, accents) -> None:
        super().__init__()
        self.accents = accents
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        if len(accents) > 0:
            spkid = np.random.default_rng(self.seed).choice(accents)
            self.ds = self.ds.filter(lambda x: x['filename'].split('_')[2] == spkid)
            print(f'using {len(self.ds)} samples from {spkid} voice conversion')
            if len(self.ds) > 0:
                self.avg_xvec = np.mean([x['xvector'] for x in self.ds], axis=0)
            else:
                self.avg_xvec = None
        else:
            self.avg_xvec = None
    
    def __repr__(self):
        return f"VoiceConversion({self.accents})"
    
    def forward(self, speech, text, *args, **kwargs):
        if self.avg_xvec is None:
            return speech
        device = self.parameters().__next__().device
        # speaker_embeddings = torch.tensor(self.ds[voice_idxs]["xvector"]).to(device).unsqueeze(0)
        speaker_embeddings = torch.FloatTensor(self.avg_xvec).to(device).unsqueeze(0)
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings, vocoder=self.vocoder)
        return speech

class VoiceConversionVCTK(AbsVoiceConversion):
    def __init__(self, accents, lang='en', vctk_dir='/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/VCTK') -> None:
        super().__init__()
        self.accents = accents
        self.vctk_dir = vctk_dir
        self.lang = lang
        meta = pd.read_csv(os.path.join(self.vctk_dir, 'speaker-info.txt'), sep=r'\s+', usecols=[0,1,2,3])
        accent_spks = meta[meta['ACCENTS'].isin(accents)]['ID'].values
        self.spk_utts = []
        for spk in accent_spks:
            self.spk_utts += [os.path.join(self.vctk_dir, 'wav48_silence_trimmed', spk, x) for x in os.listdir(os.path.join(self.vctk_dir, 'wav48_silence_trimmed', spk))]
        # from flacduration import get_flac_duration
        # self.spk_utts = [x for x in spk_utts if get_flac_duration(x) > 4.0]
    
    def _maybe_init_tts(self):
        if not hasattr(self, 'tts'):
            print(f'initializing TTS model in PID {os.getpid()}')
            from TTS.api import TTS
            device_id = os.getpid() % torch.cuda.device_count() #np.random.default_rng(time.time_ns() + os.getpid()).choice(torch.cuda.device_count())
            self.tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(f'cuda:{device_id}')
    
    def __repr__(self):
        return f"VoiceConversionVCTK({self.accents})"
    
    def _get_spk_utt(self):
        rng = np.random.default_rng(time.time_ns()+os.getpid())
        # spk_id = rng.choice(self.accent_spks)
        # spk_utt = rng.choice(os.listdir(os.path.join(self.vctk_dir, 'wav48_silence_trimmed', spk_id)))
        # spk_utts = [os.path.join(self.vctk_dir, 'wav48_silence_trimmed', spk_id, x) for x in os.listdir(os.path.join(self.vctk_dir, 'wav48_silence_trimmed', spk_id))]
        # spk_utt_path = os.path.join(self.vctk_dir, 'wav48_silence_trimmed', spk_id, spk_utt)
        spk_utt_path = rng.choice(self.spk_utts)
        return spk_utt_path
    
    def forward(self, speech, text, *args, **kwargs):
        self._maybe_init_tts()
        spk_utt_path = self._get_spk_utt()
        print(spk_utt_path)
        wav = self.tts.tts(text=text, speaker_wav=spk_utt_path, language=self.lang)
        return wav       
    
class Gain(torchaudio.transforms.Vol):
    def __init__(self, gain) -> None:
        super().__init__(gain)
    
    def __repr__(self):
        return f"Gain({self.gain})"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        return super().forward(x)

class SoxEffect(torch.nn.Module):
    def __init__(self, effect, *args, sample_rate=16000) -> None:
        super().__init__()
        self.effect = effect
        self.args = args
        self.sample_rate = sample_rate
    
    def __repr__(self):
        return f"SoxEffect({self.effect}, {self.args})"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate, [[self.effect] + [str(a) for a in self.args]])[0].squeeze(0)


class Echo(torch.nn.Module):
    def __init__(self, delay, decay=0.3, sample_rate=16000) -> None:
        super().__init__()
        self.delay = delay
        self.decay = decay
        self.sample_rate = sample_rate
    
    def __repr__(self):
        return f"Echo({self.delay}, {self.decay})"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate, [['echo', '0.8', '0.9', str(self.delay), '0.3']])[0].squeeze(0)
    
class Phaser(SoxEffect):
    def __init__(self, decay, sample_rate=16000) -> None:
        args = [0.6, 0.8, 3, decay, 2, '-t']
        super().__init__('phaser', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Phaser({self.args})"

class Tempo(SoxEffect):
    def __init__(self, factor, sample_rate=16000) -> None:
        args = [factor, 30]
        super().__init__('tempo', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Tempo({self.args})"
    
class HighPassFilter(SoxEffect):
    def __init__(self, freq, sample_rate=16000) -> None:
        args = [freq]
        super().__init__('sinc', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"HighPassFilter({self.args})"

class LowPassFilter(SoxEffect):
    def __init__(self, freq, sample_rate=16000) -> None:
        args = [f'0-{freq}']
        super().__init__('sinc', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"LowPassFilter({self.args})"
    
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