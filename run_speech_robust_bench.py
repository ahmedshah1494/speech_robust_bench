from argparse import ArgumentParser
from create_transformed_datasets import AUGMENTATIONS, PERT_ROB_AUGMENTATIONS
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
    ('openai/whisper-tiny.en', 'robust_speech/advattack_data_and_results/attacks/universal/whisper-tiny.en-10/1002/CKPT+2023-11-27+19-04-38+00/delta.ckpt'),
    ('deepspeech', 'robust_speech/advattack_data_and_results/attacks/universal/deepspeech-10/1002/CKPT+2023-11-27+17-27-39+00/delta.ckpt'),
    ('facebook/wav2vec2-base-960h', 'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-base-960h-10/1002/CKPT+2023-11-27+15-58-14+00/delta.ckpt'),
    ('facebook/wav2vec2-large-960h-lv60-self', 'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-large-960h-lv60-self-10/1002/CKPT+2023-11-27+16-06-02+00/delta.ckpt'),
    ('facebook/hubert-large-ls960-ft', 'robust_speech/advattack_data_and_results/attacks/universal/hubert-large-ls960-ft-10/1002/CKPT+2023-11-27+15-56-27+00/delta.ckpt'),
    ('facebook/wav2vec2-large-robust-ft-libri-960h', 'robust_speech/advattack_data_and_results/attacks/universal/wav2vec2-large-robust-ft-libri-960h-10/1002/CKPT+2023-11-27+16-40-39+00/delta.ckpt'),
    ('openai/whisper-large-v2', 'robust_speech/advattack_data_and_results/attacks/universal/whisper-large-v2-10/1002/CKPT+2023-11-27+18-55-29+00/delta.ckpt'),
    ('facebook/wav2vec2-large-960h', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/wav2vec2-large-960h-10/1002/CKPT+2024-02-01+20-14-19+00/delta.ckpt'),
    ('facebook/hubert-xlarge-ls960-ft', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/hubert-xlarge-ls960-ft-10/1002/CKPT+2024-01-31+22-33-28+00/delta.ckpt'),
    # ('openai/whisper-tiny', ''),
    # ('openai/whisper-small.en', ''),
    ('openai/whisper-small', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/whisper-small-10/1002/CKPT+2024-02-01+02-08-48+00/delta.ckpt'),
    # ('openai/whisper-base.en', ''),
    ('openai/whisper-base', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/whisper-base-10/1002/CKPT+2024-01-31+22-19-18+00/delta.ckpt'),
    # ('openai/whisper-medium.en', ''),
    ('openai/whisper-medium', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/whisper-medium-10/1002/CKPT+2024-02-01+04-07-06+00/delta.ckpt'),
]

es_models = [
    ('facebook/wav2vec2-large-xlsr-53-spanish', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/wav2vec2-large-xlsr-53-spanish-10/1002/CKPT+2024-01-31+22-26-37+00/delta.ckpt'),
    ('facebook/wav2vec2-base-10k-voxpopuli-ft-es','/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/wav2vec2-base-10k-voxpopuli-ft-es-10/1002/CKPT+2024-01-31+23-42-07+00/delta.ckpt'),
    ('openai/whisper-tiny', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/whisper-tiny-10/1002/CKPT+2024-02-01+20-45-03+00/delta.ckpt'),
    ('facebook/mms-1b-fl102', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/mms-1b-fl102-10/1002/CKPT+2024-02-01+23-21-00+00/delta.ckpt'),
    ('openai/whisper-large-v2', ''),
]

parser = ArgumentParser()
parser.print_help('Utility script to run speech robustness benchmark for multiple models and all perturbations. The args are same as transformers_asr_eval.py.')
parser.add_argument('--models', nargs='+', default=None, help='List of models to run. Models must be present in en_models or es_models in run_speech_robust_bench.py')
parser.add_argument('--dataset', default="librispeech_asr")
parser.add_argument('--subset', default=None)
parser.add_argument('--split', default='test.clean')
parser.add_argument('--batch_size', default=16)
parser.add_argument('--run_perturb_robustness_eval', action='store_true')
parser.add_argument('--skip_if_result_exists', action='store_true')
parser.add_argument('--overwrite_result_file', action='store_true')
args = parser.parse_args()

def create_cmd(model, delta_path, aug, sev):
    language = 'english' if args.dataset == 'librispeech_asr' else ('spa' if 'mms' in model else 'spanish')
    cmd = f'python transformers_asr_eval.py --model_name {model} --batch_size {args.batch_size} --skip_if_result_exists  --dataset {args.dataset} --split {args.split} --language {language}'
    if (delta_path != '') and (aug == 'universal_adv'):
        cmd += f' --universal_delta_path {delta_path}'
    if aug is not None:
        cmd += f' --augmentation {aug}:{sev}'
    if args.subset is not None:
        cmd += f' --subset {args.subset}'
    if args.run_perturb_robustness_eval:
        cmd += ' --run_perturb_robustness_eval'
    if args.overwrite_result_file:
        cmd += ' --overwrite_result_file'
    return cmd    

Q = Queue()
if args.models is not None:
    en_models = [m for m in en_models if m[0] in args.models]
    es_models = [m for m in es_models if m[0] in args.models]

if args.dataset == 'librispeech_asr':
    models = en_models
else:
    models = es_models

for model_data in models:
    model, delta_path = model_data
    if args.run_perturb_robustness_eval:
        for aug, (augcls, settings) in PERT_ROB_AUGMENTATIONS.items():
            cmd = create_cmd(model, delta_path, aug, 1)
            # print(cmd)
            Q.put(cmd)
    else:
        cmd = create_cmd(model, delta_path, None, None)
        Q.put(cmd)
        for aug, (augcls, settings) in AUGMENTATIONS.items():
            # if aug != 'universal_adv':
            #     continue
            for sev, s in enumerate(settings):
                if sev == 0:
                    continue        
                cmd = create_cmd(model, delta_path, aug, sev)
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

