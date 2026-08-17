from argparse import ArgumentParser
from tqdm import tqdm
import os
from copy import deepcopy
import string
from multiprocessing import cpu_count
import torch
from create_transformed_datasets import load_augmentation, transform_dataset, transform_dataset_for_ptest, parse_augmentation
from models import create_model_pipeline

N_CPUS = cpu_count()
N_GPUS = torch.cuda.device_count()

def normalize_transcript(txt):
    txt = txt.lower()
    puncs = list(string.punctuation)
    for pnc in puncs:
        txt = txt.replace(pnc, '')
    return txt

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True, help='Model name or path compatible with HuggingFace Transformers library.')
    parser.add_argument('--dataset', default="librispeech_asr", help='Name for dataset to load from huggingface hub. Used to run eval on clean data and utterance agnostic (universal) adversarial perturbations. default: librispeech_asr.')
    parser.add_argument('--srb_hf_repo', default='mshah1/speech_robust_bench_public', help='Huggingface repo name for the preprocessed speech robustness benchmark. default: mshah1/speech_robust_bench_public')
    parser.add_argument('--subset', default=None, help='Subset of the dataset to use. default: None')
    parser.add_argument('--split', default='test.clean', help='Split of the dataset to use. default: test.clean')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--augmentation', type=str, help='Augmentation to apply to the dataset. Should be of the form <aug>:<sev>, where <aug> is a key in corruption_info.AUGMENTATIONS_2_SEV, and <sev> is the severity in range 1-4 (except for voice_conversion_vctk for which it should be 1). default: None')
    parser.add_argument('--universal_delta_path', type=str, help='Path to the universal adversarial perturbation. default: None')
    parser.add_argument('--language', default='english', help='Language of the dataset. This is needs to be correctly specified for multi-lingual models. default: english')
    parser.add_argument('--output_dir', default='outputs', help='Output directory for the results. default: outputs')
    parser.add_argument('--model_parallelism', action='store_true', help='Use model parallelism for the model. default: False')
    parser.add_argument('--run_perturb_robustness_eval', action='store_true', help='Run prediction stability analysis. default: False')
    parser.add_argument('--n_perturb_per_sample', type=int, default=5, help='Number of perturbations to generate per sample for stability analysis. default: 30')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to use for stability analysis. default: 500')
    parser.add_argument('--overwrite_result_file', action='store_true', help='Overwrite the result file if it exists. default: False')
    parser.add_argument('--skip_if_result_exists', action='store_true', help='Skip the evaluation if the result file exists. default: False')
    parser.add_argument('--force_retransform', action='store_true', help='If True, do not retrieve perturbed data from hf_repo. default: False')
    parser.add_argument('--text_field', default='text', help='Field name for the text in the dataset. default: None')
    args = parser.parse_args()

    aug, sev = parse_augmentation(args)

    odir = f'{args.output_dir}/{args.model_name.split("/")[-1]}/{args.dataset.split("/")[-1]}'
    if args.subset is not None:
        odir += f':{args.subset}'
    os.makedirs(odir, exist_ok=True)

    if args.augmentation == 'universal_adv':
        aug = f'{aug}_{args.universal_delta_path.split("/")[-3]}'
    ofn = f'{aug}-{sev}'
    if args.run_perturb_robustness_eval:
        assert args.augmentation is not None
        ofn = f'{ofn}-pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
    ofp = f'{odir}/{ofn}.tsv'
    if not args.overwrite_result_file:
        i = 0
        ofp = f'{odir}/{ofn}_{i}.tsv'
        # print(ofp, os.path.exists(ofp))
        if args.skip_if_result_exists and any([f.startswith(ofn) for f in os.listdir(odir)]):#(os.path.exists(ofp) or ((i == 0) and os.path.exists(f'{odir}/{ofn}.tsv'))):
            print(f'Skipping {ofp}')
            exit()
        while os.path.exists(ofp):
            i += 1
            ofp = f'{odir}/{ofn}_{i}.tsv'
    from datasets import load_dataset, Audio, concatenate_datasets
    import evaluate
    import numpy as np
    import pandas as pd
    from corruptions import *

    if (args.augmentation is None) or (aug == 'universal_adv') or args.force_retransform:
        print(f'Loading dataset {args.dataset} {args.subset} {args.split}')
        dataset = load_dataset(args.dataset, args.subset, split=args.split)
        dataset = dataset.filter(lambda x: not x['id'].startswith('inter_segment_gap'))
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        if (aug == 'universal_adv') or args.force_retransform:
            transform = load_augmentation(aug, sev, args.universal_delta_path)
            if args.run_perturb_robustness_eval:
                dataset = transform_dataset_for_ptest(dataset, transform, args.n_samples, args.n_perturb_per_sample)
                print(len(dataset), dataset, transform)
            else:
                dataset = transform_dataset(dataset, transform)

        print(dataset)
    elif aug.startswith('accent'):
        if (args.language == 'English'):
            dataset = load_dataset(args.srb_hf_repo, 'accented_cv', split='test.clean')
        elif (args.language == 'Spanish'):
            dataset = load_dataset(args.srb_hf_repo, 'accented_cv_es', split='test')
        elif (args.language == 'French'):
            dataset = load_dataset(args.srb_hf_repo, 'accented_cv_fr', split='test')
        else:
            raise ValueError(f'Augmentation {aug} is not supported for language {args.language}')
    elif aug.startswith('itw'):
        if (args.language == 'English'):
            if aug == 'itw_nf':
                dataset = load_dataset(args.srb_hf_repo, 'social_chime', split='nearfield')
            elif aug == 'itw_ff':
                dataset = load_dataset(args.srb_hf_repo, 'social_chime', split='farfield')
            elif aug == 'itw_nf_ami':
                dataset = load_dataset(args.srb_hf_repo, 'social_ami', split='nearfield')
            elif aug == 'itw_ff_ami':
                dataset = load_dataset(args.srb_hf_repo, 'social_ami', split='farfield')
            else:
                raise ValueError(f'Augmentation {aug} is not supported. Must be one of itw-nf or itw-ff')
        else:
            raise ValueError(f'Augmentation {aug} is not supported for language {args.language}')    
    else:
        subset = f'{args.subset}_{args.split}' if args.subset else args.split
        if args.run_perturb_robustness_eval:
            subset = f'{subset}_pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
        dataset = load_dataset(args.srb_hf_repo, f'{args.dataset.split("/")[-1]}-{subset}', split=f'{aug}.{sev}')
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    if args.model_parallelism: 
        kwargs = {'device_map': 'auto'}
    else:
        kwargs = {'device': 'cuda:0'}
    pipe = create_model_pipeline(args.model_name, dataset, batch_size=args.batch_size, language=args.language, **kwargs)
    
    output_rows = []
    t = tqdm(zip(pipe, dataset))
    for out, inp in t:
        hyp = out['text'].upper()
        ref = inp[args.text_field].upper()
        
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
        t.set_description(f'{args.model_name.split("/")[-1]}\t{aug}:{sev}')
        t.set_postfix(wer=wer, cer=cer)
    df = pd.DataFrame(output_rows)
    df.to_csv(ofp, sep='\t')