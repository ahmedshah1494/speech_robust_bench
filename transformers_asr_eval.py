from datasets import load_dataset, Audio
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import torchaudio.transforms as audio_transforms
import torchaudio
from torchaudio import functional as F
import evaluate
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import os
from copy import deepcopy
import string
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
        d = torch.empty_like(x).normal_(0, 1)
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
        speaker_embeddings = torch.tensor(self.ds[voice_idxs]["xvector"]).cuda().unsqueeze(0)
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"].cuda(), speaker_embeddings, vocoder=self.vocoder)
        return speech
    
class Compose(torch.nn.Module):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
    
    def __repr__(self):
        return f"Compose({self.transforms})"
    
    def forward(self, speech, text, *args, **kwargs):
        for t in self.transforms:
            if isinstance(t, VoiceConversion):
                speech = t(speech, text)
            else:
                x = t(speech)
        return x

def normalize_transcript(txt):
    txt = txt.lower()
    puncs = list(string.punctuation)
    for pnc in puncs:
        txt = txt.replace(pnc, '')
    return txt

NOISE_SNRS = [30, 10, 5, 1, -10]
SPEEDUP_FACTORS = [1, 1.25, 1.5, 1.75, 2]
SLOWDOWN_FACTORS = [1, 0.875, 0.75, 0.625, 0.5]
PITCH_UP_STEPS = [0, 3, 6, 9, 12]
PITCH_DOWN_STEPS = [0, -3, -6, -9, -12]
VC_ACCENTS = [[], ['bdl', 'slt', 'rms', 'clb'], ['jmk'], ['ksp'], ['awb']]
AUGMENTATIONS = {
    'unoise': (UniformNoise, NOISE_SNRS),
    'gnoise': (GaussianNoise, NOISE_SNRS),
    'env_noise': (EnvNoise, NOISE_SNRS),
    'speedup': (Speed, SPEEDUP_FACTORS),
    'slowdown': (Speed, SLOWDOWN_FACTORS),
    'pitch_up': (Pitch, PITCH_UP_STEPS),
    'pitch_down': (Pitch, PITCH_DOWN_STEPS),
    'rir': (RIR, [0,1,2,3,4]),
    'voice_conversion': (VoiceConversion, VC_ACCENTS)
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', default="openai/whisper-small")
    parser.add_argument('--dataset', default="librispeech_asr")
    parser.add_argument('--subset', default=None)
    parser.add_argument('--split', default='test.clean')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--augmentation', type=str)
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
    
    if aug is None:
        transform = lambda x: x
    elif "+" in aug:
        augs = aug.split('+')
        augfns = []
        for a in augs:
            fn, sev_args = AUGMENTATIONS[a]
            augfns.append(fn(sev_args[min(sev, len(sev_args)-1)]))
        transform = Compose(augfns)
    elif aug in AUGMENTATIONS:
        fn, sev_args = AUGMENTATIONS[aug]
        transform = fn(sev_args[sev])

    print(transform, aug, sev)
    def transform_(batch):
        # if isinstance(transform, VoiceConversion):
        #     new_audios = transform([x['array'] for x in batch['audio']], batch['text'])
        #     for audio, na in zip(batch['audio'], new_audios):
        #         audio['array'] = na
        # else:
        if isinstance(transform, (VoiceConversion, Compose)):
            T = deepcopy(transform).cuda()
        else:
            T = transform
        for audio, text in zip(batch['audio'], batch['text']):
            if isinstance(transform, (VoiceConversion, Compose)):
                audio['array'] = T(audio['array'], text)
            else:
                audio['array'] = T(audio['array'])
        if isinstance(transform, VoiceConversion):
            del T
        return batch

    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    nproc = 8 if isinstance(transform, VoiceConversion) else 4
    print(dataset[0])
    if aug is not None:
        dataset = dataset.map(transform_, batched=True, batch_size=128, num_proc=nproc, load_from_cache_file=isinstance(transform, VoiceConversion))
    dataset = dataset.with_format('np')

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    if args.model_name == 'deepspeech':
        from deepspeech_pytorch.model import DeepSpeech
        from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
        from deepspeech_pytorch.decoder import GreedyDecoder

        model = DeepSpeech.load_from_checkpoint('/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/deepspeech_ckps/librispeech_pretrained_v3.ckpt')
        parser = ChunkSpectrogramParser(audio_conf=model.spect_cfg)
        def extract_features(x):
            waveform = x['audio']['array']
            spec = list(parser.parse_audio(waveform))[0]
            x['spec'] = spec
            x['lengths'] = spec.shape[1]
            return x
        dataset = dataset.map(extract_features, batched=False, num_proc=4)
        
        def collate_fn(batch):
            specs = [torch.FloatTensor(batch[i]['spec']).transpose(0,1) for i in range(len(batch))]
            lengths = torch.LongTensor([(batch[i]['lengths']) for i in range(len(batch))])
            specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True).unsqueeze(1).transpose(2,3)
            return {'spec': specs, 'lengths': lengths}
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False)
        decoder = GreedyDecoder(model.labels)
        def transcribe(dataloader):
            for batch in dataloader:
                length_order = torch.argsort(batch['lengths'], descending=True)
                reverse_length_order = torch.argsort(length_order)
                batch['spec'] = batch['spec'][length_order]
                batch['lengths'] = batch['lengths'][length_order]
                out, lens, _ = model(batch['spec'].cuda(), batch['lengths'].cuda())                
                decoded_output, decoded_offsets = decoder.decode(out, lens)
                for i in reverse_length_order:
                    yield {'text': decoded_output[i][0]}
        pipe = transcribe(dataloader)
    elif args.model_name == 'rnnt':
        import torchaudio
        from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH

        feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
        decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder().cuda()
        token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()

        def extract_features(x):
            waveform = torch.FloatTensor(x['audio']['array'])
            spec, length = feature_extractor(waveform)
            x['spec'] = spec
            x['lengths'] = length[0]
            return x
        dataset = dataset.map(extract_features, batched=False, num_proc=4)

        def collate_fn(batch):
            specs = [torch.FloatTensor(batch[i]['spec']) for i in range(len(batch))]
            lengths = torch.LongTensor([(batch[i]['lengths']) for i in range(len(batch))])
            specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
            return {'spec': specs, 'lengths': lengths}      
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False)
        
        def transcribe(dataloader):
            for batch in dataloader:
                with torch.no_grad():
                    for feature, length in zip(batch['spec'].cuda(), batch['lengths'].cuda()):
                        hypotheses = decoder(feature.unsqueeze(0), length.unsqueeze(0), 1)
                        text = token_processor(hypotheses[0][0])
                        yield {'text': text}
        pipe = transcribe(dataloader)
    else:
        if args.model_parallelism: 
            device_kwargs = {'device_map': 'auto'}
        else:
            device_kwargs = {'device': 'cuda:0'}
        pipe = pipeline("automatic-speech-recognition", model=args.model_name, batch_size=args.batch_size, torch_dtype=torch.float16, **device_kwargs)
        pipe = pipe(KeyDataset(dataset, "audio"))
    
    output_rows = []
    t = tqdm(zip(pipe, dataset))
    for out, inp in t:
        hyp = out['text'].upper()
        ref = inp['text'].upper()
        
        ref = normalize_transcript(ref)
        hyp = normalize_transcript(hyp)

        wer = wer_metric.compute(references=[ref], predictions=[hyp])
        cer = cer_metric.compute(references=[ref], predictions=[hyp])
        r = {
            'id': inp['id'],
            'reference': ref,
            'prediction': hyp,
            'wer': wer,
            'cer': cer
        }
        output_rows.append(r)
        t.set_postfix(wer=wer, cer=cer)

    odir = f'{args.output_dir}/{args.model_name.split("/")[-1]}/{args.dataset.split("/")[-1]}'
    if not os.path.exists(odir):
        os.makedirs(odir)

    df = pd.DataFrame(output_rows)
    df.to_csv(f'{odir}/{aug}-{sev}.tsv', sep='\t')