from argparse import ArgumentParser
from corruption_info import AUGMENTATIONS_2_SEV as AUGMENTATIONS, PERT_ROB_AUGMENTATIONS_2_SEV as PERT_ROB_AUGMENTATIONS
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

def find_path_to_univeral_adv(model, root, snr=10):
    if root is None:
        print(f'WARNING: No universal adversarial perturbation directory provided for {model}. The model will not be evaluated against universal adversarial perturbations.')
        return None
    model_name = model.split('/')[-1]
    dirpath = os.path.join(root, f'{model_name}-{snr}')
    delta_paths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file == 'delta.ckpt':
                delta_paths.append(os.path.join(root, file))
    if len(delta_paths) > 0:
        return sorted(delta_paths)[-1]
    elif snr != 10:
        return find_path_to_univeral_adv(model, root, snr=10)
    else:
        return None

en_models = [
    ('openai/whisper-tiny.en', ''),
    ('deepspeech', ''),
    ('facebook/wav2vec2-base-960h', ''),
    ('facebook/wav2vec2-large-960h-lv60-self', ''),
    ('facebook/hubert-large-ls960-ft', ''),
    ('facebook/wav2vec2-large-robust-ft-libri-960h', ''),
    ('openai/whisper-large-v2', ''),
    ('facebook/wav2vec2-large-960h', ''),
    ('facebook/hubert-xlarge-ls960-ft', ''),
    ('openai/whisper-tiny', ''),
    # ('openai/whisper-small.en', ''),
    ('openai/whisper-small', ''),
    # ('openai/whisper-base.en', ''),
    ('openai/whisper-base', ''),
    # ('openai/whisper-medium.en', ''),
    ('openai/whisper-medium', ''),
    ('facebook/mms-1b-fl102', ''),
    ('microsoft/speecht5_asr', ''),
    ('nvidia/canary-1b', ''),
    ('nvidia/parakeet-ctc-1.1b', ''),
    ('nvidia/parakeet-rnnt-0.6b', ''),
    ('nvidia/parakeet-rnnt-1.1b', ''),
    ('nvidia/parakeet-ctc-0.6b', ''),
]

es_models = [
    ('facebook/wav2vec2-large-xlsr-53-spanish', ''),
    ('facebook/wav2vec2-base-10k-voxpopuli-ft-es',''),
    ('openai/whisper-tiny', ''),
    ('facebook/mms-1b-fl102', ''),
    ('openai/whisper-large-v2', ''),
    ('openai/whisper-base', ''),
    ('nvidia/canary-1b', '')
]

du_models = [
    # ('openai/whisper-tiny', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/whisper-tiny-10/1002/CKPT+2024-02-01+20-45-03+00/delta.ckpt'),
    ('facebook/mms-1b-fl102', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/mms-1b-fl102-10/1002/CKPT+2024-02-01+23-21-00+00/delta.ckpt'),
    ('openai/whisper-large-v2', 'robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/whisper-large-v2-10/1002/CKPT+2024-02-02+09-19-36+00/delta.ckpt'),
    ('openai/whisper-base', 'robust_speech/advattack_data_and_results/attacks/universal/MLE-ES/whisper-base-10/1002/CKPT+2024-01-31+22-19-18+00/delta.ckpt'),
    ('nvidia/canary-1b', '')
]

fr_models = [
    # ('openai/whisper-tiny', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/whisper-tiny-10/1002/CKPT+2024-02-01+20-45-03+00/delta.ckpt'),
    ('facebook/mms-1b-fl102', '/jet/home/mshah1/projects/audio_robustness_benchmark/robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/mms-1b-fl102-10/1002/CKPT+2024-02-01+23-21-00+00/delta.ckpt'),
    ('openai/whisper-large-v2', 'robust_speech/advattack_data_and_results/attacks/universal/MLS-ES/whisper-large-v2-10/1002/CKPT+2024-02-02+09-19-36+00/delta.ckpt'),
    ('openai/whisper-base', 'robust_speech/advattack_data_and_results/attacks/universal/MLE-ES/whisper-base-10/1002/CKPT+2024-01-31+22-19-18+00/delta.ckpt'),
    ('nvidia/canary-1b', '')
]


dataset2lang = {
    'librispeech_asr': {
            None: ('en', 'English')
        },
    'LIUM/tedlium': {
            'release3': ('en', 'English')
        },
    'facebook/multilingual_librispeech': {
            'spanish': ('es', 'Spanish'),
            'german': ('de', 'German'),
            'french': ('fr', 'French')
        },
    'common_voice': {
            None: ('en', 'English')
        },
}

parser = ArgumentParser()
parser.add_help = True
parser.add_argument('--models', nargs='+', default=None, help='List of models to run. Models must be present in en_models or es_models in run_speech_robust_bench.py')
parser.add_argument('--dataset', default="librispeech_asr")
parser.add_argument('--subset', default=None)
parser.add_argument('--split', default='test.clean')
parser.add_argument('--batch_size', default=16)
parser.add_argument('--augmentations', nargs='+')
parser.add_argument('--language', default=None)
parser.add_argument('--run_perturb_robustness_eval', action='store_true')
parser.add_argument('--output_dir', default='outputs', help='Output directory for the results. default: outputs')
parser.add_argument('--skip_if_result_exists', action='store_true', help='Skip evaluation if result file for this model, augmentation and severity exists.')
parser.add_argument('--overwrite_result_file', action='store_true')
parser.add_argument('--run_accent_eval', action='store_true')
parser.add_argument('--run_itw_eval', action='store_true', help='evaluate on in-the-wild data from CHiME and AMI')
parser.add_argument('--run_universal_adv_eval', action='store_true')
parser.add_argument('--run_universal_adv_eval_only', action='store_true')
parser.add_argument('--universal_adv_delta_path', help='Directory containing the utterance agnoistic (universal) adversarial perturbations. The script will look for files named delta.ckpt <universal_adv_delta_path>/<model_name>. If multiple are found the full paths to the files will be lexically sorted and the last one will be selected.')
parser.add_argument('--force_retransform', action='store_true', help='If True, do not retrieve perturbed data from hf_repo. default: False')
parser.add_argument('--jobs_per_gpu', default=1)
args = parser.parse_args()

args.run_universal_adv_eval = args.run_universal_adv_eval | args.run_universal_adv_eval_only
if args.language is not None:
    language = args.language
else:
    language = dataset2lang[args.dataset][args.subset][1]

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
    if args.force_retransform:
        cmd += ' --force_retransform'
    if dataset == 'facebook/multilingual_librispeech' and (aug is None):
        cmd += ' --text_field transcript'
    return cmd    

Q = Queue()

if language == 'English':
    models = en_models
elif language == 'Spanish':
    models = es_models
elif language == 'German':
    models = du_models
elif language == 'French':
    models = fr_models
else:
    raise ValueError(f'Language {language} not supported')

if args.models is not None:
    models = [m for m in models if m[0] in args.models]

print(models)
for model_data in models:
    model = model_data[0]
    delta_path = ''
    
    if args.run_accent_eval:
        cmd = create_cmd(model, delta_path, 'accent', 0)
        Q.put(cmd)
    # continue

    if args.run_itw_eval:
        for aug in ['itw_nf', 'itw_ff', 'itw_nf_ami', 'itw_ff_ami']:
            cmd = create_cmd(model, delta_path, aug, 0)
            Q.put(cmd)

    if args.run_perturb_robustness_eval:
        for aug, settings in PERT_ROB_AUGMENTATIONS.items():
            for i in range(1, 5):
                cmd = create_cmd(model, delta_path, aug, i)
                Q.put(cmd)
    elif args.run_universal_adv_eval_only:
        for i in range(1, 5):
            snrs = AUGMENTATIONS['universal_adv']
            delta_path = find_path_to_univeral_adv(model, args.universal_adv_delta_path, snr=snrs[i])
            print(model, snrs[i], delta_path)
            cmd = create_cmd(model, delta_path, 'universal_adv', i)
            Q.put(cmd)
    else:
        if (not args.run_universal_adv_eval_only) and (args.augmentations is None) or args.run_perturb_robustness_eval:
            cmd = create_cmd(model, delta_path, None, None)
            Q.put(cmd)
        for aug, settings in AUGMENTATIONS.items():
            print(aug, args.augmentations)
            if (args.augmentations is not None) and (aug not in args.augmentations):
                continue
            if (aug == 'universal_adv') and ((not args.run_universal_adv_eval) or (delta_path is None)):
                continue
            for sev, s in enumerate(settings):
                if sev == 0:
                    continue        
                cmd = create_cmd(model, delta_path, aug, sev)
                print(cmd)
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

