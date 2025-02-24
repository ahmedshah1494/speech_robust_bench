import pandas as pd
import argparse
import os
import shutil
from pathlib import Path
from compute_DNSMOS import ComputeScore
import concurrent
from tqdm import tqdm
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create accented CV dataset')
    parser.add_argument('--tar', type=str, help='Path to the Common Voice tar file')
    parser.add_argument('--output_dir', type=str, help='Output directory', default='.')
    parser.add_argument('--excluded_accents', type=str, nargs='+', help='Accents to exclude')
    parser.add_argument('--hf_repo', type=str, help='Huggingface repository to which the dataset will be uploaded to')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    args = parser.parse_args()

    data_root = os.path.basename(args.tar).replace('.tar.gz', '')
    data_root = Path(args.output_dir) / data_root[:-3] / data_root[-2:]
    clip_dir = Path(data_root) / 'clips'
    clean_clip_dir = Path(data_root) / 'clean-clips'

    print(f'data_root: {data_root}')
    print(f'clip_dir: {clip_dir}')
    print(f'clean_clip_dir: {clean_clip_dir}')

    if not os.path.exists(data_root / 'test.tsv'):
        os.system(f"tar -xvzf {args.tar} -C {args.output_dir} --wildcards --no-anchored '*test.tsv'")

    df = pd.read_csv(data_root / 'test.tsv', sep='\t')
    print(f'Loaded {len(df)} rows')
    df = df[~(df.accents.isna())]
    print(f'Found {len(df)} rows with accent labels')
    df.path = df.path.apply(lambda x: str(clip_dir / x))
    clips_to_extract = df.path[df.path.apply(lambda x: not os.path.exists(x))].values
    if len(clips_to_extract) > 0:
        with open(data_root / 'clips_with_accent.txt', 'w') as f:
            for path in clips_to_extract:
                if not os.path.exists(path):
                    f.write(f'{path}\n')
            f.flush()
        os.system(f"tar -xvzf {args.tar} --files-from {data_root / 'clips_with_accent.txt'}")
        print(f'Extracted {len(clips_to_extract)} clips')

    p808_model_path = os.path.join('DNSMOS/DNSMOS', 'model_v8.onnx')
    primary_model_path = os.path.join('DNSMOS/DNSMOS', 'sig_bak_ovr.onnx')
    compute_score = ComputeScore(primary_model_path, p808_model_path)

    if not os.path.exists(data_root / 'test_dnsmos_results.tsv'):
        rows = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(compute_score, clip, 16000, False): clip for clip in df.path}
            for future in tqdm(concurrent.futures.as_completed(future_to_url)):
                clip = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (clip, exc))
                else:
                    rows.append(data)
        dnsmos_df = pd.DataFrame(rows)
        dnsmos_df.to_csv(data_root / 'test_dnsmos_results.tsv' , index=False)
    else:
        dnsmos_df = pd.read_csv(data_root / 'test_dnsmos_results.tsv')

    df['dnsmos'] = dnsmos_df['P808_MOS'].values
    df.to_csv(data_root / 'test_wAccent.tsv', index=False)

    df = df[df.dnsmos >= 3.4]
    df.to_csv(data_root / 'test_wAccent_filtered.tsv', index=False)
    print(f'Filtered to {len(df)} clips with MOS >= 3.4')

    df = pd.read_csv(data_root / 'test_wAccent_filtered.tsv')
    if args.excluded_accents is not None:
        df = df[~df.accents.isin(args.excluded_accents)]
    print(f'Loaded {len(df)} rows')

    print(f'Copying {len(df)} clips to {clean_clip_dir}')
    clean_clip_dir.mkdir(exist_ok=True, parents=True)
    for clip in tqdm(df.path):
        if not os.path.exists(clean_clip_dir / os.path.basename(clip)):
            shutil.copy(clip, clean_clip_dir / os.path.basename(clip))

    
    metadata_df = pd.DataFrame(
        {
            'file_name': df.path.apply(lambda x: os.path.basename(x)),
            'accent': df.accents,
            'text': df.sentence,
            'gender': df.gender,
            'age': df.age,
            'locale': df.locale,
            'id': df.path.apply(lambda x: os.path.basename(x).split('.')[0].split('_')[-1]),
        }
    )
    metadata_df.to_csv(clean_clip_dir / 'metadata.csv', index=False)
    ds = load_dataset('audiofolder', data_dir=str(clean_clip_dir))['train']
    print(ds)

    if args.hf_repo is not None:
        ds.push_to_hub(args.hf_repo, args.dataset_name, split='test')