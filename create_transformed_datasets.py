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
import scipy

N_CPUS = cpu_count()
N_GPUS = torch.cuda.device_count()

def normalize_transcript(txt):
    txt = txt.lower()
    puncs = list(string.punctuation)
    for pnc in puncs:
        txt = txt.replace(pnc, '')
    return txt

def parse_augmentation(args):
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
    if aug == 'accent' and args.language != 'English':
        aug = f'accent_{args.language}'
    return aug, sev

def load_augmentation(aug, sev, universal_delta_path=None):
    if aug is None:
        transform = None
    elif "+" in aug:
        augs = aug.split('+')
        augfns = []
        for a in augs:
            fn, sev_args = AUGMENTATIONS_2_FN_SEV[a]
            augfns.append(fn(sev_args[min(sev, len(sev_args)-1)]))
        transform = Compose(augfns)
    elif aug in AUGMENTATIONS_2_FN_SEV:
        fn, sev_args = AUGMENTATIONS_2_FN_SEV[aug]
        if issubclass(fn, UniversalAdversarialPerturbation):
            transform = fn(sev_args[sev], universal_delta_path)
        else:    
            transform = fn(sev_args[sev])
    return transform

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

    nproc = 2*torch.cuda.device_count() if isinstance(transform, (AbsVoiceConversion)) else 4
    print(dataset[0])
    if transform is not None:
        dataset = dataset.map(transform_, batched=True, batch_size=64, num_proc=nproc, load_from_cache_file=isinstance(transform, AbsVoiceConversion))
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
            if not isinstance(transform, (GaussianNoise, UniformNoise, EnvNoise, RIR, RealRIR)):
                dataset = dataset_
            datasets.append(dataset_)
        # print(datasets[0][0]['audio']['array'], datasets[-1][0]['audio']['array'])
        # print(pert_idx, datasets[0][0]['audio']['array'] - datasets[-1][0]['audio']['array'])
    dataset = concatenate_datasets(datasets)
    dataset = dataset.with_format('np')
    print(dataset[0])
    return dataset

def get_repo_metadata(hf_repo):
    import requests
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[ 502, 503, 504 ])
    s.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    API_URL = f"https://huggingface.co/api/datasets/{hf_repo}/croissant"
    def query():
        response = s.get(API_URL, headers=headers, timeout=10)
        return response.json()
    data = query()
    data = {x['name']:x for x in data['recordSet']}
    return data

def check_split_exists(hf_repo, config, split):
    from huggingface_hub import HfFileSystem
    hffs = HfFileSystem()
    files = hffs.ls(f'datasets/{hf_repo}/{config}')
    for f in files:
        if os.path.basename(f['name']).startswith(split):
            return True
    return False   

# def check_split_exists(hf_repo, config, split):
#     print(hf_repo, config, split)
#     if config not in data:
#         return False
#     splits = data[config]['description'].split('splits:')[1].split('\n',2)[0].split(',')
#     splits = [s.strip() for s in splits]
#     return split in splits
    

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
    parser.add_argument('--dataset', default="librispeech_asr", help='Name for dataset to load from huggingface hub. default: librispeech_asr.')
    parser.add_argument('--hf_repo', required=True, help='HuggingFace repo to push the transformed dataset to.')
    parser.add_argument('--subset', default=None, help='Subset of the dataset to use. default: None')
    parser.add_argument('--split', default='test.clean', help='Split of the dataset to use. default: test.clean')
    parser.add_argument('--text_field', default='text', help='Name of the field in the dataset that contains the text. default: text')
    parser.add_argument('--augmentation', type=str, help='Augmentation to apply to the dataset. Should be of the form <aug>:<sev>, where <aug> is a key in corruptions.AUGMENTATIONS_2_FN_SEV, and <sev> is the severity in range 1-4 (except for voice_conversion_vctk for which it should be 1). default: None')
    parser.add_argument('--universal_delta_path', type=str, help='Path to the universal adversarial perturbation. default: None')
    parser.add_argument('--run_perturb_robustness_eval', action='store_true', help='Run prediction stability analysis. default: False')
    parser.add_argument('--n_perturb_per_sample', type=int, default=30, help='Number of perturbations to generate per sample for stability analysis. default: 30')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples to use for stability analysis. default: 500')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset in the repo. default: False')
    args = parser.parse_args()

    aug, sev = parse_augmentation(args)

    repo_subset = f'{args.subset}_{args.split}' if args.subset else args.split
    repo_split = f'{aug}.{sev}'
    if aug == 'universal_adv':
        tgt_model = args.universal_delta_path.split('/')[-4]
        repo_split = f'universal_adv_{tgt_model}.{sev}'
    if args.run_perturb_robustness_eval:
        repo_subset = f'{repo_subset}_pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
    repo_dataset = f'{args.dataset.split("/")[-1]}-{repo_subset}'
    
    if not args.overwrite:
        if check_split_exists(args.hf_repo, repo_dataset, repo_split):
            print(f'Split {repo_subset}_{repo_split} already exists in the repo. Skipping...')
            exit(0)

    transform = load_augmentation(aug, sev, args.universal_delta_path)
    print(aug, sev)
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    dataset = dataset.map(lambda x: {'text': x[args.text_field]}, num_proc=8, remove_columns=[args.text_field])
    dataset = dataset.filter(lambda x: not x['id'].startswith('inter_segment_gap'))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    if args.augmentation is not None:
        if args.run_perturb_robustness_eval:
            dataset = transform_dataset_for_ptest(dataset, transform, args.n_samples, args.n_perturb_per_sample)
        else:
            dataset = transform_dataset(dataset, transform)

    # subset = f'{args.subset}_{args.split}' if args.subset else args.split
    # if aug == 'universal_adv':
    #     tgt_model = args.universal_delta_path.split('/')[-4]
    #     aug = f'universal_adv_{tgt_model}'
    # if args.run_perturb_robustness_eval:
    #     subset = f'{subset}_pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
    repo_ds_feats = load_dataset(args.hf_repo, repo_dataset, split='gnoise.1', streaming=True).features
    cols_to_remove = [c for c in dataset.column_names if c not in repo_ds_feats]
    dataset = dataset.remove_columns(cols_to_remove)
    for f, t in repo_ds_feats.items():
        dataset = dataset.cast_column(f, t)
    dataset.push_to_hub(args.hf_repo, repo_dataset, split=repo_split)