import speechmetrics
import argparse
from datasets import load_dataset, Audio
from corruption_info import AUGMENTATIONS_2_SEV as AUGMENTATIONS
import pandas as pd
import os
from tqdm import tqdm
import multiprocessing as mp

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
    'openai/whisper-tiny.en',
    'deepspeech',
    'facebook/wav2vec2-base-960h',
    'facebook/wav2vec2-large-960h-lv60-self',
    'facebook/hubert-large-ls960-ft',
    'facebook/wav2vec2-large-robust-ft-libri-960h',
    'openai/whisper-large-v2',
    'facebook/wav2vec2-large-960h',
    'facebook/hubert-xlarge-ls960-ft',
    'openai/whisper-tiny',
    'openai/whisper-small',
    'openai/whisper-base',
    'openai/whisper-medium',
    'facebook/mms-1b-fl102',
    'microsoft/speecht5_asr',
    'nvidia/canary-1b'
]

es_models = [
    'facebook/wav2vec2-large-xlsr-53-spanish',
    'facebook/wav2vec2-base-10k-voxpopuli-ft-es',
    'openai/whisper-tiny',
    'facebook/mms-1b-fl102',
    'openai/whisper-large-v2',
    'openai/whisper-base',
    'nvidia/canary-1b',
]

def create_metric_workers(n_workers):
    def metric_fn(qin, qout):
        print(f'worker started {os.getpid()}...')
        window_length = 5 # seconds
        metrics = speechmetrics.load(['pesq'], window_length)
        while True:
            item = qin.get()
            if item is None:
                print(f'worker {os.getpid()} exiting...')
                break
            x, x_c = item
            assert x['id'] == x_c['id']

            try:
                scores = metrics(x['audio']['array'], x_c['audio']['array'], rate=x['audio']['sampling_rate'])
            except Exception as e:
                print(e)
                scores = {'pesq': [None]}
            r = {
                'filename': x['id'],
                'len_in_sec': len(x['audio']['array'])/x['audio']['sampling_rate'],
                'PESQ': scores['pesq'][0]
            }
            qout.put(r)
    
    qin = mp.Queue()
    qout = mp.Queue()

    workers = []
    for _ in range(n_workers):
        p = mp.Process(target=metric_fn, args=(qin, qout))
        p.start()
        workers.append(p)
    return qin, qout, workers

def main(args, metrics, clean_dataset, aug, sev, model=None):
    subset = None
    try:
        if (aug == 'universal_adv'):
            from create_transformed_datasets import load_augmentation, transform_dataset
            dataset = load_dataset(args.dataset, args.subset, split=args.split)
            dataset = dataset.filter(lambda x: not x['id'].startswith('inter_segment_gap'))
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
            delta_path = find_path_to_univeral_adv(model, args.universal_adv_dir)
            transform = load_augmentation(aug, sev, delta_path)
            dataset = transform_dataset(dataset, transform)

            print(dataset)
        elif aug == 'accent':
            dataset = load_dataset(args.srb_hf_repo, 'accented_cv', split='test.clean')
            args.dataset = 'common_voice'
        elif aug.startswith('itw'):
            if aug == 'itw_nf':
                dataset = load_dataset(args.srb_hf_repo, 'in-the-wild', split='nearfield')
                args.dataset = 'chime6'
            elif aug == 'itw_ff':
                dataset = load_dataset(args.srb_hf_repo, 'in-the-wild', split='farfield')
                args.dataset = 'chime6'
            elif aug == 'itw_nf_ami':
                dataset = load_dataset(args.srb_hf_repo, 'in-the-wild-AMI', split='nearfield')
                args.dataset = 'ami'
            elif aug == 'itw_ff_ami':
                dataset = load_dataset(args.srb_hf_repo, 'in-the-wild-AMI', split='farfield')
                args.dataset = 'ami'
            else:
                raise ValueError(f'Augmentation {aug} is not supported. Must be one of itw-nf or itw-ff')
        else:
            subset = f'{args.subset}_{args.split}' if args.subset else args.split
            split = f'{aug}.{sev}'
            dataset = load_dataset(args.srb_hf_repo, f'{args.dataset.split("/")[-1]}-{subset}', split=split)
    except Exception as e:
        print(e)
    
    ds_name = args.dataset.split('/')[-1]
    subset = f'_{subset}' if subset is not None else ''
    csv_dir = f'{args.output_dir}/{ds_name}{subset}'
    os.makedirs(csv_dir, exist_ok=True)
    ofn = f'{aug}.{sev}'
    if model is not None:
        ofn = f'{model.split("/")[-1]}.{ofn}'
    csv_path = f'{csv_dir}/{ofn}.csv'

    if os.path.exists(csv_path):
        print(f'{csv_path} already exists')
        return
    
    dataset = dataset.sort('id')
    qin, qout, workers = create_metric_workers(4)
    print(f'Computing metrics for {aug} {sev}...')
    print('adding items to queue...')
    for x, x_c in tqdm(zip(dataset, clean_dataset)):
        qin.put((x, x_c))
    for _ in range(len(workers)):
        qin.put(None)

    print('getting results...')
    rows = []
    for _ in tqdm(range(len(dataset))):
        r = qout.get()
        rows.append(r)
    for w in workers:
        w.join()
    # rows = []
    # for x, x_c in tqdm(zip(dataset, clean_dataset)):
    #     assert x['id'] == x_c['id']

    #     scores = metrics(x['audio']['array'], x_c['audio']['array'], rate=x['audio']['sampling_rate'])
    #     r = {
    #         'filename': x['id'],
    #         'len_in_sec': len(x['audio']['array'])/x['audio']['sampling_rate'],
    #         'PESQ': scores['pesq']
    #     }
    #     rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', "--testset_dir", default='.', 
    #                     help='Path to the dir containing audio clips in .wav to be evaluated')
    # parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('--dataset', default="librispeech_asr")
    parser.add_argument('--subset', default=None)
    parser.add_argument('--split', default='test.clean')
    parser.add_argument('--srb_hf_repo', default='mshah1/speech_robust_bench_public', help='Huggingface repo name for the preprocessed speech robustness benchmark. default: mshah1/speech_robust_bench_public')
    parser.add_argument('--universal_adv_dir', default=None, help='Path to the directory containing universal adversarial perturbations')
    parser.add_argument('-o', '--output_dir', default='speechmetrics_csv/PESQ')
    
    args = parser.parse_args()
    clean_dataset = load_dataset(args.dataset, args.subset, split=args.split)
    clean_dataset = clean_dataset.sort('id')
    window_length = 5 # seconds
    metrics = speechmetrics.load(['pesq'], window_length)
    for aug, sevs in AUGMENTATIONS.items():
        for sev in range(1, len(sevs)):
            models = es_models if args.subset == 'spanish' else en_models
            if aug == 'universal_adv':
                for model in models:
                    main(args, metrics, clean_dataset, aug, sev, model)
            else:
                main(args, metrics, clean_dataset, aug, sev)