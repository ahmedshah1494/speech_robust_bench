from argparse import ArgumentParser
from torch.cuda import device_count
from multiprocessing import Queue, Process
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

def is_complete(data_root, dataset, model, snr, attack_type, csv_name):
    model = model.split('/')[-1]
    result_dir = f'{data_root}/attacks/{attack_type}/{dataset}/{model}-{snr}'
    for root, dirs, files in os.walk(result_dir):
        if f'cer_adv_{csv_name}.txt' in files:
            return True

en_models = [
    ('openai/whisper-tiny.en', ''),
    ('deepspeech', ''),
    ('facebook/wav2vec2-base-960h', ''),
    ('facebook/wav2vec2-large-960h-lv60-self', ''),
    ('facebook/hubert-large-ls960-ft', ''),
    ('facebook/wav2vec2-large-robust-ft-libri-960h', ''),
    ('facebook/wav2vec2-large-960h', ''),
    ('openai/whisper-tiny', ''),
    ('openai/whisper-small', ''),
    ('openai/whisper-base', ''),
    ('openai/whisper-medium', ''),
    ('microsoft/speecht5_asr', ''),
    ('facebook/hubert-xlarge-ls960-ft', ''),
    ('openai/whisper-large-v2', ''),
    ('openai/whisper-large-v3', ''),
    ('nvidia/canary-1b', ''),
    ('facebook/mms-1b-fl102', ''),
    ('nvidia/parakeet-ctc-1.1b', ''),
    ('nvidia/parakeet-rnnt-1.1b', ''),
    ('nvidia/parakeet-rnnt-0.6b', ''),
    ('ibm-granite/granite-speech-4.1-2b', ''),
    ('Qwen/Qwen3-ASR-1.7B-hf', ''),
    ('CohereLabs/cohere-transcribe-03-2026', ''),
]

es_models = [
    ('facebook/wav2vec2-large-xlsr-53-spanish', ''),
    ('facebook/wav2vec2-base-10k-voxpopuli-ft-es',''),
    ('openai/whisper-tiny', ''),
    ('openai/whisper-base', ''),
    ('nvidia/canary-1b', ''),
    ('facebook/mms-1b-fl102', ''),
    ('openai/whisper-large-v2', ''),
]

dataset2lang = {
    'LibriSpeech': ('en', 'English'),
    'TEDLIUM': ('en', 'English'),
    'MLS-ES': ('es', 'Spanish'),
}

PGD_SNRS = [40, 30, 20, 10]
UNIVERSAL_SNRS = [40,30,20,10]
parser = ArgumentParser()
parser.add_argument('--models', nargs='+', default=None, help='List of models to run. Models must be present in en_models or es_models in run_speech_robust_bench_adv.py')
parser.add_argument('--dataset', default="LibriSpeech", help='Dataset to run the attack on. This should be the name of a directory in the <data_root>/data.')
parser.add_argument('--data_root', default="robust_speech/advattack_data_and_results")
parser.add_argument('--data_csv_name', default="test-clean", help='Name of the csv file in the <data_root>/data/<dataset>/csv directory. DO NOT INCLUDE THE FILE EXTENSION.')
parser.add_argument('--attack_type', default="pgd", help='Type of attack to run. Options: pgd, universal', choices=['pgd', 'universal', 'cw'])
parser.add_argument('--jobs_per_gpu', default=1)
args = parser.parse_args()

language = dataset2lang[args.dataset][1]

def create_pgd_cmd(model, snr):
    if model == 'deepspeech':
        cmd = f'python evaluate.py attack_configs/LibriSpeech/pgd/deepspeech.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'TEDLIUM':
            cmd += ' --tokenizer_file tokenizer_uncased'
    elif model.startswith('nvidia/'):
        cmd = f'python evaluate.py attack_configs/LibriSpeech/pgd/{model.replace("nvidia/","")}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'MLS-ES':
            cmd += ' --lang es'
    elif model.startswith('ibm-granite/'):
        cmd = f'python evaluate.py attack_configs/LibriSpeech/pgd/{model.replace("ibm-granite/","")}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    elif model.startswith('Qwen/'):
        model_file = model.replace('Qwen/', '').lower()
        cmd = f'python evaluate.py attack_configs/LibriSpeech/pgd/{model_file}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    elif model.startswith('CohereLabs/'):
        cmd = f'python evaluate.py attack_configs/LibriSpeech/pgd/{model.replace("CohereLabs/","")}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    else:
        repo, model = model.split('/')
        cmd = f'python evaluate.py attack_configs/LibriSpeech/pgd/hf.yaml --root={args.data_root} --model_repo={repo} --model_name={model} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'MLS-ES' and model in ['mms-1b-fl102', 'whisper-large-v2', 'whisper-tiny', 'whisper-base']:
            if model == 'mms-1b-fl102':
                cmd += ' --lang spa'
            else:
                cmd += ' --lang es'
    return cmd

def create_universal_cmd(model, snr):
    if model == 'deepspeech':
        cmd = f'python fit_attacker.py attack_configs/LibriSpeech/universal/deepspeech.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'TEDLIUM':
            cmd += ' --tokenizer_file tokenizer_uncased'
    elif model.startswith('nvidia/'):
        cmd = f'python fit_attacker.py attack_configs/LibriSpeech/universal/{model.replace("nvidia/","")}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'MLS-ES':
            cmd += ' --lang es'
    elif model.startswith('ibm-granite/'):
        cmd = f'python fit_attacker.py attack_configs/LibriSpeech/universal/{model.replace("ibm-granite/","")}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    elif model.startswith('Qwen/'):
        model_file = model.replace('Qwen/', '').lower()
        cmd = f'python fit_attacker.py attack_configs/LibriSpeech/universal/{model_file}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    elif model.startswith('CohereLabs/'):
        cmd = f'python fit_attacker.py attack_configs/LibriSpeech/universal/{model.replace("CohereLabs/","")}.yaml --root={args.data_root} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
    else:
        repo, model = model.split('/')
        cmd = f'python fit_attacker.py attack_configs/LibriSpeech/universal/hf.yaml --root={args.data_root} --model_repo={repo} --model_name={model} --snr={snr} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'MLS-ES'  and model in ['mms-1b-fl102', 'whisper-large-v2', 'whisper-tiny', 'whisper-base']:
            if model == 'mms-1b-fl102':
                cmd += ' --lang spa'
            else:
                cmd += ' --lang es'
    return cmd

def create_cw_cmd(model):
    if model == 'deepspeech':
        cmd = f'python evaluate.py attack_configs/LibriSpeech/cw/hf.yaml --root={args.data_root} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'TEDLIUM':
            cmd += ' --tokenizer_file tokenizer_uncased'
    if model == 'nvidia/canary-1b':
        cmd = f'python evaluate.py attack_configs/LibriSpeech/cw/canary-1b.yaml --root={args.data_root} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'MLS-ES':
            cmd += ' --lang es'
    else:
        repo, model = model.split('/')
        cmd = f'python evaluate.py attack_configs/LibriSpeech/cw/hf.yaml --root={args.data_root} --model_repo={repo} --model_name={model} --dataset {args.dataset} --data_csv_name {args.data_csv_name}'
        if args.dataset == 'MLS-ES' and model in ['mms-1b-fl102', 'whisper-large-v2', 'whisper-tiny', 'whisper-base']:
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
    elif args.attack_type == 'cw':
        return create_cw_cmd(model)
    else:
        raise ValueError(f'Invalid attack type {args.attack_type}')

Q = Queue()
if args.models is not None:
    en_models = [m for m in en_models if m[0] in args.models]
    es_models = [m for m in es_models if m[0] in args.models]

if language == 'English':
    models = en_models
elif language == 'Spanish':
    models = es_models
else:
    raise ValueError(f'Language {language} not supported')

SNRS = {
    'pgd': PGD_SNRS,
    'universal': UNIVERSAL_SNRS,
    'cw': [None]
}[args.attack_type]
for model_data in models:
    model, delta_path = model_data
    for snr in SNRS:
        if is_complete(args.data_root, args.dataset, model, snr, args.attack_type, args.data_csv_name):
            print(f'{model} at SNR {snr} is already complete. Skipping...')
            continue
        else:
            print(f'Adding {model} at SNR {snr} to queue...')
            cmd = create_attack_cmd(model, snr)
            Q.put(cmd)
Q.put('None')

processes =[]
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
else:
    devices = range(device_count())
for did in devices:
    for _ in range(int(args.jobs_per_gpu)):
        p = Process(target=run_cmd, args=(Q,did))
        p.daemon = True
        processes.append(p)
        p.start()
for p in processes:
    p.join()

