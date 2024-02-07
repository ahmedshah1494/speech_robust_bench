from argparse import ArgumentParser
from create_transformed_datasets import AUGMENTATIONS
from torch.cuda import device_count
from multiprocessing import Queue, Process
import subprocess
import shlex
import os

def run_cmd(queue, device_id):
    while True:
        cmd = queue.get()
        if cmd == 'None':
            break
        cmd = f'CUDA_VISIBLE_DEVICES={device_id} {cmd}'
        # subprocess.Call(shlex.split(cmd))
        if os.system(cmd):
            print(f'Error running {cmd}')

en_models = [
    # ('openai/whisper-tiny.en', 'robust_speech/advattack_data_and_results/attacks/universal/whisper-tiny.en-10/1002/CKPT+2023-11-27+19-04-38+00/delta.ckpt'),
    # ('deepspeech', 'robust_speech/advattack_data_and_results/attacks/universal/deepspeech-10/1002/CKPT+2023-11-27+17-27-39+00/delta.ckpt'),
    # ('facebook/wav2vec2-base-960h', 'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-base-960h-10/1002/CKPT+2023-11-27+15-58-14+00/delta.ckpt'),
    # ('facebook/wav2vec2-large-960h-lv60-self', 'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-large-960h-lv60-self-10/1002/CKPT+2023-11-27+16-06-02+00/delta.ckpt'),
    # ('facebook/hubert-large-ls960-ft', 'robust_speech/advattack_data_and_results/attacks/universal/hubert-large-ls960-ft-10/1002/CKPT+2023-11-27+15-56-27+00/delta.ckpt'),
    # ('facebook/wav2vec2-large-robust-ft-libri-960h', 'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-large-robust-ft-libri-960h-10/1002/CKPT+2023-11-27+16-40-39+00/delta.ckpt'),
    # ('openai/whisper-large-v2', 'robust_speech/advattack_data_and_results/attacks/universal/whisper-large-v2-10/1002/CKPT+2023-11-27+18-55-29+00/delta.ckpt'),
    ('facebook/wav2vec2-large-960h', ''),
    # ('openai/whisper-tiny', ''),
    # ('openai/whisper-small.en', ''),
    # ('openai/whisper-small', ''),
    # ('openai/whisper-base.en', ''),
    # ('openai/whisper-base', ''),
    # ('openai/whisper-medium.en', ''),
    # ('openai/whisper-medium', ''),
    # ('facebook/hubert-xlarge-ls960-ft', ''),
]

es_models = [
    # ('facebook/wav2vec2-large-xlsr-53-spanish', ''),
    # ('facebook/wav2vec2-base-10k-voxpopuli-ft-es',''),
    # ('facebook/mms-1b-fl102', ''),
    ('openai/whisper-large-v2', ''),
    # ('openai/whisper-tiny', ''),
]
PGD_SNRS = [40, 30, 20, 10]
UNIVERSAL_SNRS = [10]
parser = ArgumentParser()
parser.add_argument('--models', nargs='+', default=None, help='List of models to run. Models must be present in en_models or es_models in run_speech_robust_bench_adv.py')
parser.add_argument('--dataset', default="LibriSpeech", help='Dataset to run the attack on. This should be the name of a directory in the <data_root>/data.')
parser.add_argument('--data_root', default="robust_speech/advattack_data_and_results")
parser.add_argument('--data_csv_name', default="test-clean", help='Name of the csv file in the <data_root>/data/<dataset>/csv directory. DO NOT INCLUDE THE FILE EXTENSION.')
parser.add_argument('--attack_type', default="pgd", help='Type of attack to run. Options: pgd, universal', choices=['pgd', 'universal'])
args = parser.parse_args()

def create_pgd_cmd(model, snr):
    repo, model = model.split('/')
    cmd = f'python evaluate.py attack_configs/LibriSpeech/pgd/hf.yaml --root={args.data_root} --model_repo={repo} --model_name={model} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    if args.dataset == 'MLS-ES':
        if model == 'mms-1b-fl102':
            cmd += ' --lang spa'
        else:
            cmd += ' --lang es'
    return cmd

def create_universal_cmd(model, snr):
    repo, model = model.split('/')
    cmd = f'python fit_attacker.py attack_configs/LibriSpeech/universal/hf.yaml --root={args.data_root} --model_repo={repo} --model_name={model} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    if args.dataset == 'MLS-ES':
        if model == 'mms-1b-fl102':
            cmd += ' --lang spa'
        else:
            cmd += ' --lang es'
    return cmd

def create_attack_cmd(model, snr):
    if args.attack_type == 'pgd':
        return create_pgd_cmd(model, snr)
    elif args.attack_type == 'universal':
        return create_universal_cmd(model, snr)
    else:
        raise ValueError(f'Invalid attack type {args.attack_type}')

Q = Queue()
if args.models is not None:
    en_models = [m for m in en_models if m[0] in args.models]
    es_models = [m for m in es_models if m[0] in args.models]

if args.dataset == 'LibriSpeech':
    models = en_models
else:
    models = es_models

SNRS = UNIVERSAL_SNRS if args.attack_type == 'universal' else PGD_SNRS
for model_data in models:
    model, delta_path = model_data
    for snr in SNRS:
        cmd = create_attack_cmd(model, snr)
        Q.put(cmd)
Q.put('None')

processes =[]
for did in range(device_count()):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        did = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[did]
    p = Process(target=run_cmd, args=(Q,did))
    p.daemon = True
    processes.append(p)
    p.start()
for p in processes:
    p.join()

