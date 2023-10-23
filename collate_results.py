import pandas as pd
import numpy as np
from scipy.stats import iqr
import os

results_dir = 'outputs-norm_trans'
result_files = []
for root, dirs, files in os.walk(results_dir):
    for name in files:
        if name.endswith('tsv'):
            result_files.append(os.path.join(root, name))

results = []
result_dfs = []
for rfp in result_files:
    print(rfp)
    _, model, dataset, noise_sev = rfp[:-4].split('/')
    if '-' in noise_sev:
        noise, sev = noise_sev.split('-')
        sev = int(sev)
    else:
        noise = noise_sev
        sev = 0

    df = pd.read_csv(rfp, sep='\t', usecols=[1,2,3,4,5])
    df['model'] = model
    df['augmentation'] = noise
    df['severity'] = sev
    result_dfs.append(df)

    nwords = np.array([len(ref.split()) for ref in df['reference']])
    nchars = np.array([len(ref) for ref in df['reference']])
    werrs = (df['wer'].values * nwords)
    wer = werrs.sum() / nwords.sum()
    cerrs = (df['cer'].values * nchars)
    cer = cerrs.sum() / nchars.sum()

    r = {
        'model': model,
        'dataset': dataset,
        'augmentation': noise,
        'severity': sev,
        'WER': wer,
        'CER': cer,
        # 'WER (avg)': np.mean(wers),
        # 'WER (std)': np.std(wers),
        # 'WER (median)': np.median(wers),
        # 'WER (IQR)': iqr(wers),
        # 'CER (avg)': np.mean(cers),
        # 'CER (std)': np.std(cers),
        # 'CER (median)': np.median(cers),
        # 'CER (IQR)': iqr(cers),
    }
    results.append(r)

results_df = pd.DataFrame(results)
results_df.to_csv('collated_results_all_models-normedtrans.csv')
results_df.to_latex('collated_results_all_models-normedtrans.tex')

full_result_df = pd.concat(result_dfs)
full_result_df.to_csv('full_result_df-normedtrans.csv')