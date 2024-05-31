import pandas as pd
import os

if __name__ == '__main__':
    results_dir = 'outputs_precomputed_data'
    metadata = pd.read_csv('/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/cv-corpus-17.0-2024-03-15/en/metadata.csv')
    file_ids = metadata['id'].values
    models = os.listdir(results_dir)
    for m in models:
        result_file_path = f'{results_dir}/{m}/common_voice/accent-1_0.tsv'
        if os.path.exists(result_file_path):
            df = pd.read_csv(result_file_path, sep='\t', index_col=0)
            df = df[df['id'].isin(file_ids)]
            df = df.reset_index()
            df.to_csv(f'{results_dir}/{m}/common_voice/accent-1_0_filtered.tsv', sep='\t')
            