from datasets import load_dataset
import argparse
from tempfile import TemporaryDirectory
import os
import pandas as pd
from compute_DNSMOS import ComputeScore
import concurrent
from tqdm import tqdm

def compute_dnsmos(args):
    p808_model_path = os.path.join('DNSMOS/DNSMOS', 'model_v8.onnx')

    primary_model_path = os.path.join('DNSMOS/DNSMOS', 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    try:
        dataset = load_dataset(args.dataset, args.subset, split=args.split)
        dataset = dataset.map(lambda x: {'id': os.path.basename(x['path'])})
    except Exception as e:
        print(e)
        return
    
    rows = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip, 16000, args.personalized_MOS): clip for clip in dataset}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create accented CV dataset')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name', required=True, help='Name of the dataset in the Huggingface datasets library')
    parser.add_argument('-l', '--language', type=str, help='Language', required=True, help='2-letter language code. Assumes that the language correspons to the dataset config.')
    parser.add_argument('-s', '--split', type=str, help='Split', required=True)
    parser.add_argument('-r', '--hf_repo', type=str, help='Huggingface repository to which the dataset will be uploaded to', required=True)
    args = parser.parse_args()
    
    dnsmos_df = compute_dnsmos(args)
    selected_files = set(dnsmos_df[dnsmos_df['P808_MOS'] > 3.4].filenames.tolist())
    
    dataset = load_dataset(args.dataset, args.lang, args.split)
    dataset = dataset.filter(lambda x: (x['file'] in selected_files) and (x['accent'] == ))

    output_dir = f'accented_cv/{args.lang}'
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk()