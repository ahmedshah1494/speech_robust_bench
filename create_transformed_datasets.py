from datasets import load_dataset, Audio, concatenate_datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import os
from copy import deepcopy
import string
from corruptions import *
from multiprocessing import cpu_count

N_CPUS = cpu_count()
N_GPUS = torch.cuda.device_count()

def normalize_transcript(txt):
    txt = txt.lower()
    puncs = list(string.punctuation)
    for pnc in puncs:
        txt = txt.replace(pnc, '')
    return txt

def load_augmentation(args):
    if args.augmentation:
        if ':' in args.augmentation:
            aug, sev = args.augmentation.split(':', 1)
            sev = int(sev)
            assert sev <= 4
        else:
            aug = args.augmentation
            sev = None
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
        if sev is None:
            transform = fn(args.severity)
            sev = args.severity
        else:
            if issubclass(fn, UniversalAdversarialPerturbation):
                transform = fn(sev_args[sev], args.universal_delta_path)
            else:    
                transform = fn(sev_args[sev])
    return transform, aug, sev

def trim_text_to_charcount(text, charcount):
    if len(text) > charcount:
        new_text = text[:charcount]
        if text[charcount] in string.ascii_letters:
            new_text = new_text[:new_text.rfind(' ')]
        text = new_text
    return text

def transform_dataset(dataset, transform):
    def transform_(batch):
        if isinstance(transform, (AbsVoiceConversion, Compose)):
            device_id = np.random.default_rng(time.time_ns() + os.getpid()).choice(torch.cuda.device_count())
            T = deepcopy(transform).to(f'cuda:{device_id}')
        else:
            T = transform
        for i, (audio, text) in enumerate(zip(batch['audio'], batch['text'])):
            if isinstance(transform, (AbsVoiceConversion, Compose)):
                text = trim_text_to_charcount(text, 250)
                batch['text'][i] = text
                audio['array'] = T(audio['array'], text)
            else:
                audio['array'] = T(audio['array'])
        if isinstance(transform, AbsVoiceConversion):
            del T
        print('done', os.getpid())
        return batch

    nproc = 4 if isinstance(transform, (AbsVoiceConversion)) else 8
    print(dataset[0])
    if transform is not None:
        dataset = dataset.map(transform_, batched=True, batch_size=128, num_proc=nproc, load_from_cache_file=isinstance(transform, AbsVoiceConversion))
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

    nproc = cpu_count()
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
            if not isinstance(transform, (GaussianNoise, UniformNoise, EnvNoise)):
                dataset = dataset_
            datasets.append(dataset_)
        # print(datasets[0][0]['audio']['array'], datasets[-1][0]['audio']['array'])
        # print(pert_idx, datasets[0][0]['audio']['array'] - datasets[-1][0]['audio']['array'])
    dataset = concatenate_datasets(datasets)
    dataset = dataset.with_format('np')
    print(dataset[0])
    return dataset

UNIV_ADV_DELTAS = [
    'robust_speech/advattack_data_and_results/attacks/universal/whisper-tiny.en-10/1002/CKPT+2023-11-27+19-04-38+00/delta.ckpt',
    'robust_speech/advattack_data_and_results/attacks/universal/deepspeech-10/1002/CKPT+2023-11-27+17-27-39+00/delta.ckpt',
    'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-base-960h-10/1002/CKPT+2023-11-27+15-58-14+00/delta.ckpt',
    'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-large-960h-lv60-self-10/1002/CKPT+2023-11-27+16-06-02+00/delta.ckpt',
    'robust_speech/advattack_data_and_results/attacks/universal/hubert-large-ls960-ft-10/1002/CKPT+2023-11-27+15-56-27+00/delta.ckpt',
    'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-large-robust-ft-libri-960h-10/1002/CKPT+2023-11-27+16-40-39+00/delta.ckpt',
    'robust_speech/advattack_data_and_results/attacks/universal/whisper-large-v2-10/1002/CKPT+2023-11-27+18-55-29+00/delta.ckpt',
]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True, help='Model name or path compatible with HuggingFace Transformers library.')
    parser.add_argument('--dataset', default="librispeech_asr", help='Name for dataset to load from huggingface hub. default: librispeech_asr.')
    parser.add_argument('--hf_repo', required=True, help='HuggingFace repo to push the transformed dataset to.')
    parser.add_argument('--subset', default=None, help='Subset of the dataset to use. default: None')
    parser.add_argument('--split', default='test.clean', help='Split of the dataset to use. default: test.clean')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--augmentation', type=str, help='Augmentation to apply to the dataset. Should be of the form <aug>:<sev>, where <aug> is a key in corruptions.AUGMENTATIONS, and <sev> is the severity in range 1-4 (except for voice_conversion_vctk for which it should be 1). default: None')
    parser.add_argument('--universal_delta_path', type=str, help='Path to the universal adversarial perturbation. default: None')
    parser.add_argument('--run_perturb_robustness_eval', action='store_true', help='Run prediction stability analysis. default: False')
    parser.add_argument('--n_perturb_per_sample', type=int, default=30, help='Number of perturbations to generate per sample for stability analysis. default: 30')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples to use for stability analysis. default: 500')
    args = parser.parse_args()

    transform, aug, sev = load_augmentation(args)
    print(aug, sev)
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    if args.perturb_robustness_eval:
        dataset = transform_dataset_for_ptest(dataset, transform, args.n_samples, args.n_perturb_per_sample)
    else:
        dataset = transform_dataset(dataset, transform)

    subset = f'{args.subset}_{args.split}' if args.subset else args.split
    if aug == 'universal_adv':
        tgt_model = args.universal_delta_path.split('/')[-4]
        aug = f'universal_adv_{tgt_model}'
    if args.perturb_robustness_eval:
        subset = f'{subset}_pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
    dataset.push_to_hub(args.hf_repo, f'{args.dataset.split("/")[-1]}-{subset}', split=f'{aug}.{sev}')