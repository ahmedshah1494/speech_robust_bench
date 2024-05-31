from argparse import ArgumentParser
from create_transformed_datasets import AUGMENTATIONS, PERT_ROB_AUGMENTATIONS
from torch.cuda import device_count
from multiprocessing import Queue, Process
import os

def run_cmd(queue, device_id):
    while True:
        cmd = queue.get()
        if cmd is None:
            break
        cmd = f'CUDA_VISIBLE_DEVICES={device_id} {cmd}'
        # subprocess.Call(shlex.split(cmd))
        if os.system(cmd):
            print(f'Error running {cmd}')

def find_path_to_univeral_adv(model, root):
    if root is None:
        print(f'WARNING: No universal adversarial perturbation directory provided for {model}. The model will not be evaluated against universal adversarial perturbations.')
        return None
    model_name = model.split('/')[-1]
    dirpath = os.path.join(root, f'{model_name}-10')
    delta_paths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file == 'delta.ckpt':
                delta_paths.append(os.path.join(root, file))
    if len(delta_paths) > 0:
        return sorted(delta_paths)[-1]
    else:
        return None

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
    ('openai/whisper-tiny', ''),
    # ('openai/whisper-small.en', ''),
    ('openai/whisper-small', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/whisper-small-10/1002/CKPT+2024-02-01+02-08-48+00/delta.ckpt'),
    # ('openai/whisper-base.en', ''),
    ('openai/whisper-base', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/whisper-base-10/1002/CKPT+2024-01-31+22-19-18+00/delta.ckpt'),
    # ('openai/whisper-medium.en', ''),
    ('openai/whisper-medium', 'robust_speech/advattack_data_and_results/attacks/universal/LibriSpeech/whisper-medium-10/1002/CKPT+2024-02-01+04-07-06+00/delta.ckpt'),
    ('facebook/mms-1b-fl102', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/mms-1b-fl102-10/1002/CKPT+2024-02-01+23-21-00+00/delta.ckpt'),
    # ('facebook/seamless-m4t-v2-large', ''),
    # ('facebook/seamless-m4t-medium', ''),
    ('microsoft/speecht5_asr', ''),
    ('nvidia/canary-1b', '')

]

es_models = [
    ('facebook/wav2vec2-large-xlsr-53-spanish', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/wav2vec2-large-xlsr-53-spanish-10/1002/CKPT+2024-01-31+22-26-37+00/delta.ckpt'),
    ('facebook/wav2vec2-base-10k-voxpopuli-ft-es','/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/wav2vec2-base-10k-voxpopuli-ft-es-10/1002/CKPT+2024-01-31+23-42-07+00/delta.ckpt'),
    ('openai/whisper-tiny', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/whisper-tiny-10/1002/CKPT+2024-02-01+20-45-03+00/delta.ckpt'),
    ('facebook/mms-1b-fl102', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/mms-1b-fl102-10/1002/CKPT+2024-02-01+23-21-00+00/delta.ckpt'),
    ('openai/whisper-large-v2', 'robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/whisper-large-v2-10/1002/CKPT+2024-02-02+09-19-36+00/delta.ckpt'),
    ('openai/whisper-base', 'robust_speech/advattack_data_and_results/attacks/universal/MLE-ES/whisper-base-10/1002/CKPT+2024-01-31+22-19-18+00/delta.ckpt'),
    ('nvidia/canary-1b', '')
]



dataset2lang = {
    'librispeech_asr': ('en', 'English'),
    'LIUM/tedlium': ('en', 'English'),
    'facebook/multilingual_librispeech': ('es', 'Spanish'),
    'common_voice': ('en', 'English'),
}

parser = ArgumentParser()
parser.add_help = True
parser.add_argument('--models', nargs='+', default=None, help='List of models to run. Models must be present in en_models or es_models in run_speech_robust_bench.py')
parser.add_argument('--dataset', default="librispeech_asr")
parser.add_argument('--subset', default=None)
parser.add_argument('--split', default='test.clean')
parser.add_argument('--batch_size', default=16)
parser.add_argument('--augmentations', nargs='+')
parser.add_argument('--run_perturb_robustness_eval', action='store_true')
parser.add_argument('--output_dir', default='outputs', help='Output directory for the results. default: outputs')
parser.add_argument('--skip_if_result_exists', action='store_true', help='Skip evaluation if result file for this model, augmentation and severity exists.')
parser.add_argument('--overwrite_result_file', action='store_true')
parser.add_argument('--run_accent_eval', action='store_true')
parser.add_argument('--run_itw_eval', action='store_true', help='evaluate on in-the-wild data from CHiME')
parser.add_argument('--run_universal_adv_eval', action='store_true')
parser.add_argument('--run_universal_adv_eval_only', action='store_true')
parser.add_argument('--universal_adv_delta_path', help='Directory containing the utterance agnoistic (universal) adversarial perturbations. The script will look for files named delta.ckpt <universal_adv_delta_path>/<model_name>. If multiple are found the full paths to the files will be lexically sorted and the last one will be selected.')
parser.add_argument('--jobs_per_gpu', default=1)
args = parser.parse_args()

args.run_universal_adv_eval = args.run_universal_adv_eval | args.run_universal_adv_eval_only
language = dataset2lang[args.dataset][1]

def create_cmd(model, delta_path, aug, sev):
    if aug == 'accent':
        dataset = 'common_voice'
    elif aug in ['itw_nf', 'itw_ff']:
        dataset = 'chime'
    elif aug in ['itw_nf_ami', 'itw_ff_ami']:
        dataset = 'ami'
    else:
        dataset = args.dataset
    cmd = f'python evaluate_single.py --model_name {model} --batch_size {args.batch_size}  --dataset {dataset} --split {args.split} --language {language} --output_dir {args.output_dir}'
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
    if args.skip_if_result_exists:
        cmd += ' --skip_if_result_exists'
    return cmd    

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
print(models)
for model_data in models:
    model = model_data[0]
    if args.run_universal_adv_eval or args.run_universal_adv_eval_only:
        delta_path = find_path_to_univeral_adv(model, args.universal_adv_delta_path)
    else:
        delta_path = ''
    
    if args.run_accent_eval:
        cmd = create_cmd(model, delta_path, 'accent', 0)
        Q.put(cmd)

    if args.run_perturb_robustness_eval:
        for aug, (augcls, settings) in PERT_ROB_AUGMENTATIONS.items():
            cmd = create_cmd(model, delta_path, aug, 1)
            Q.put(cmd)
    elif args.run_universal_adv_eval_only:
        for i in range(1, 5):
            cmd = create_cmd(model, delta_path, 'universal_adv', i)
            Q.put(cmd)
    else:
        if (not args.run_universal_adv_eval_only) and (args.augmentations is None):
            cmd = create_cmd(model, delta_path, None, None)
            Q.put(cmd)
        for aug, (augcls, settings) in AUGMENTATIONS.items():
            print(aug, args.augmentations)
            if (args.augmentations is not None) and (aug not in args.augmentations):
                continue
            if (aug == 'universal_adv') and ((not args.run_universal_adv_eval) or (delta_path is None)):
                continue
            for sev, s in enumerate(settings):
                if sev == 0:
                    continue        
                cmd = create_cmd(model, delta_path, aug, sev)
                Q.put(cmd)

processes =[]
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
else:
    devices = range(device_count())
for did in devices:
    for _ in range(int(args.jobs_per_gpu)):
        p = Process(target=run_cmd, args=(Q,did))
        Q.put(None)
        p.daemon = True
        processes.append(p)
        p.start()
for p in processes:
    p.join()

