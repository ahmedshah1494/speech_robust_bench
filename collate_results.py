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

def get_metric_from_adv_file(fname):
    with open(fname, 'r') as f:
        metric = float(f.readlines()[0].split()[1])/100
    return metric

adv_results_dir = 'robust_speech/advattack_data_and_results/attacks'
adv_results = []
snr_to_sev = [50, 40, 30, 20]
for root, dirs, files in os.walk(adv_results_dir):
    if 'log.txt' in files:        
        model = root.split('/')[-2]
        sev = snr_to_sev.index(int(model.split('-')[-1]))+1
        model = '-'.join(model.split('-')[:-1])
        augmentation = root.split('/')[-3]
        dataset = 'librispeech_asr'
        r = {
            'model': model,
            'dataset': dataset,
            'augmentation': augmentation,
            'severity': sev,
            'WER': get_metric_from_adv_file(os.path.join(root, 'wer_adv_test-clean-100.txt')),
            'CER': get_metric_from_adv_file(os.path.join(root, 'cer_adv_test-clean-100.txt')),
        }
        adv_results.append(r)
        # r = {
        #     'model': model,
        #     'dataset': dataset,
        #     'augmentation': None,
        #     'severity': 0,
        #     'WER': get_metric_from_adv_file(os.path.join(root, 'wer_test-clean-100.txt')),
        #     'CER': get_metric_from_adv_file(os.path.join(root, 'cer_test-clean-100.txt')),
        # }
        # adv_results.append(r)
adv_results_df = pd.DataFrame(adv_results)
adv_results_df = adv_results_df.drop_duplicates()

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
    df['dataset'] = dataset
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
results_df = pd.concat([results_df, adv_results_df])
results_df.to_csv('results/collated_results_all_models-normedtrans.csv')
results_df.to_latex('results/collated_results_all_models-normedtrans.tex')

full_result_df = pd.concat(result_dfs)
full_result_df.to_csv('results/full_result_df-normedtrans.csv')