from datasets import load_dataset, Audio, concatenate_datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import evaluate
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import os
from copy import deepcopy
import string
from corruptions import *

def normalize_transcript(txt):
    txt = txt.lower()
    puncs = list(string.punctuation)
    for pnc in puncs:
        txt = txt.replace(pnc, '')
    return txt

def load_augmentation(args):
    if args.augmentation:
        aug, sev = args.augmentation.split(':', 1)
        sev = int(sev)
        assert sev <= 4
    else:
        aug = None
        sev = 0
    
    if aug is None:
        transform = None
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
    return transform, aug, sev

def transform_dataset(dataset, transform):
    def transform_(batch):
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

    nproc = 8 if isinstance(transform, VoiceConversion) else 4
    print(dataset[0])
    if transform is not None:
        dataset = dataset.map(transform_, batched=True, batch_size=128, num_proc=nproc, load_from_cache_file=isinstance(transform, VoiceConversion))
    dataset = dataset.with_format('np')
    return dataset

def transform_dataset_for_ptest(dataset, transform, num_samples, num_perturb_per_sample, subset_seed=9999):
    def update_pert_idx(batch):
        batch['pert_idx'] = [pert_idx] * len(batch['audio'])
        return batch
    
    def transform_(batch):
        T = transform
        for audio in batch['audio']:
            audio['array'] = T(audio['array'])
        return batch

    nproc =  4
    rng = np.random.default_rng(subset_seed)
    subset = rng.choice(len(dataset), num_samples, replace=False)
    dataset = dataset.select(subset)
    print(dataset[0])
    datasets = []
    for pert_idx in range(num_perturb_per_sample):
        dataset = dataset.map(update_pert_idx, batched=True, batch_size=128, num_proc=nproc, load_from_cache_file=False)
        if pert_idx == 0:            
            datasets.append(dataset)
        else:
            dataset_ = dataset.map(transform_, batched=True, batch_size=128, num_proc=nproc, load_from_cache_file=False,)
            if not isinstance(transform, (GaussianNoise, UniformNoise)):
                dataset = dataset_
            datasets.append(dataset_)
        # print(datasets[0][0]['audio']['array'], datasets[-1][0]['audio']['array'])
        # print(pert_idx, datasets[0][0]['audio']['array'] - datasets[-1][0]['audio']['array'])
    dataset = concatenate_datasets(datasets)
    dataset = dataset.with_format('np')
    print(dataset[0])
    return dataset

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
    parser.add_argument('--language', default='english')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--model_parallelism', action='store_true')
    parser.add_argument('--run_perturb_robustness_eval', action='store_true')
    parser.add_argument('--n_perturb_per_sample', type=int, default=30)
    parser.add_argument('--n_samples', type=int, default=500)
    args = parser.parse_args()

    transform, aug, sev = load_augmentation(args)
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    if args.run_perturb_robustness_eval:
        dataset = transform_dataset_for_ptest(dataset, transform, args.n_samples, args.n_perturb_per_sample)
    else:
        dataset = transform_dataset(dataset, transform)

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
        if ('whisper' in args.model_name) and (args.language != 'english'):
            from transformers import WhisperProcessor
            processor = WhisperProcessor.from_pretrained(args.model_name)
            gen_kwargs = {'forced_decoder_ids': processor.get_decoder_prompt_ids(language=args.language, task="transcribe")}
            print(gen_kwargs)
        else:
            gen_kwargs = {}
        pipe = pipeline("automatic-speech-recognition", model=args.model_name, batch_size=args.batch_size, torch_dtype=torch.float16, **device_kwargs, generate_kwargs=gen_kwargs)
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
        if 'pert_idx' in inp:
            r['pert_idx'] = inp['pert_idx']
        output_rows.append(r)
        t.set_postfix(wer=wer, cer=cer)

    odir = f'{args.output_dir}/{args.model_name.split("/")[-1]}/{args.dataset.split("/")[-1]}'
    if not os.path.exists(odir):
        os.makedirs(odir)

    df = pd.DataFrame(output_rows)
    ofn = f'{aug}-{sev}'
    if args.run_perturb_robustness_eval:
        ofn = f'{ofn}-pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
    df.to_csv(f'{odir}/{ofn}.tsv', sep='\t')