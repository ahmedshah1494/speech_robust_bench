import pandas as pd
import sys
import os

MIN_DNSMOS = 3.4

if __name__ == '__main__':
    orig_meta_file = sys.argv[1]
    audio_dir = 'clips'
    metadata = pd.read_csv(sys.argv[1], sep='\t', index_col=0)
    metadata = metadata.drop(columns=['client_id', 'sentence_id', 'up_votes', 'down_votes', 'sentence_domain','variant', 'segment'])
    dnsmos = pd.read_csv('/jet/home/mshah1/projects/audio_robustness_benchmark/dnsmos_csv/speech_robust_bench_accented_cv/test.csv')
    metadata['id'] = metadata['path'].apply(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    clean_fileids = dnsmos[dnsmos['P808_MOS'] >= MIN_DNSMOS]['filename'].values
    metadata = metadata[~metadata['accents'].str.contains('united states', case=False)]
    metadata = metadata[metadata['id'].isin(clean_fileids)]
    metadata = metadata.rename(columns={'path':'file_name', 'sentence':'text'})
    # metadata['file_name'] = metadata['file_name'].apply(lambda x: audio_dir+'/'+x)
    metadata.to_csv(f'{os.path.dirname(orig_meta_file)}/metadata.csv', index=False)