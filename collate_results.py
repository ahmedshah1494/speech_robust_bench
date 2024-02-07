import pandas as pd
import numpy as np
from scipy.stats import iqr
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='outputs', help='Directory containing the results of the perturbation robustness evaluation. default: ./outputs')
parser.add_argument('--robust_speech_data_root', default=f'{os.environ['SRB_ROOT']}/robust_speech_data_root', help='Directory containing the results of the perturbation robustness evaluation. default: ./outputs')
args = parser.parse_args()

results_dir = args.results_dir
result_files = []
pert_eval_result_files = []
for root, dirs, files in os.walk(results_dir):
    if os.path.basename(root).startswith('.'):
        continue
    for name in files:
        if name.endswith('tsv'):
            if 'pertEval' in name:
                pert_eval_result_files.append(os.path.join(root, name))
            else:
                result_files.append(os.path.join(root, name))

def get_metric_from_adv_file(fname):
    if not os.path.exists(fname):
        print(f'File {fname} does not exist')
        return None
    with open(fname, 'r') as f:
        lines = f.readlines()
        metric = float(lines[0].split()[1])/100
        nerrs = int(lines[0].split()[3])
        nwords = int(lines[0].split()[5][:-1])
    return metric, nwords, nerrs

def load_full_result_from_adv_file(fname):
    if not os.path.exists(fname):
        print(f'File {fname} does not exist')
        return None
    rows = []
    with open(fname, 'r') as f:
        lines = f.readlines()[12:]
        # lines = lines[::5]
        for i in range(0, len(lines), 5):
            line = lines[i]
            split = line.split()
            utt_id = split[0][:-1]
            wer = float(split[2])
            werrs = int(split[4])
            nwords = int(split[6][:-1])
            ref = ' '.join(lines[i+1].split(' ; '))
            hyp = ' '.join(lines[i+3].split(' ; '))
            r = {
                'id': utt_id,
                'wer': wer,
                # 'werrs': werrs,
                # 'nwords': nwords,
                'reference': ref,
                'prediction': hyp                
            }
            rows.append(r)
    df = pd.DataFrame(rows)
    return df

adv_results_dir = f'{args.robust_speech_data_root}/attacks'
adv_results = []
adv_result_dfs = []
snr_to_sev = [50, 40, 30, 20, 10]

for root, dirs, files in os.walk(adv_results_dir):
    if os.path.basename(root).startswith('.'):
        continue
    if 'log.txt' in files:        
        model = root.split('/')[-2]
        print(root)
        augmentation = root.split('/')[-4]
        if augmentation == 'universal':
            continue
        snr = int(model.split('-')[-1])
        if snr not in snr_to_sev:
            continue
        sev = snr_to_sev.index(snr)
        model = '-'.join(model.split('-')[:-1])
        dataset = root.split('/')[-3]
        
        subsets = [fn.split('.')[0].replace('wer_adv_test-clean','').replace('cer_adv_test-clean', '') for fn in files if (fn.startswith('wer_adv_test-clean') or fn.startswith('cer_adv_test-clean'))]
        subsets = set(subsets)
        print(root, subsets)
        for subset in subsets:
            wer_file = os.path.join(root, f'wer_test-clean{subset}.txt')
            cer_file = os.path.join(root, f'cer_test-clean{subset}.txt')
            wer_adv_file = os.path.join(root, f'wer_adv_test-clean{subset}.txt')
            cer_adv_file = os.path.join(root, f'cer_adv_test-clean{subset}.txt')

            if os.path.exists(wer_file) and os.path.exists(cer_file) and os.path.exists(wer_adv_file) and os.path.exists(cer_adv_file):
                clean_word_df = load_full_result_from_adv_file(wer_file)
                clean_char_df = load_full_result_from_adv_file(cer_file)[['id', 'wer']]
                clean_char_df.rename(columns={'wer': 'cer'}, inplace=True)
                clean_df = pd.merge(clean_word_df, clean_char_df, on='id')
                clean_df['augmentation'] = None
                clean_df['severity'] = 0

                # adv_df = load_full_result_from_adv_file(os.path.join(root, 'wer_adv_test-clean.txt'))
                adv_word_df = load_full_result_from_adv_file(wer_adv_file)
                adv_char_df = load_full_result_from_adv_file(cer_adv_file)[['id', 'wer']]
                adv_char_df.rename(columns={'wer': 'cer'}, inplace=True)
                adv_df = pd.merge(adv_word_df, adv_char_df, on='id')
                adv_df['augmentation'] = augmentation
                adv_df['severity'] = sev

                df = pd.concat([clean_df, adv_df])
                df['model'] = model
                df['dataset'] = dataset
                adv_result_dfs.append(df)

                wer, nwords, nwerrs = get_metric_from_adv_file(wer_adv_file)
                cer, nchars, ncerrs = get_metric_from_adv_file(cer_adv_file)
                r = {
                    'model': model,
                    'dataset': dataset,
                    'augmentation': augmentation,
                    'severity': sev,
                    'WER': wer,
                    'CER': cer,
                    'WED': nwerrs,
                    'CED': ncerrs,
                    'nwords': nwords,
                    'nchars': nchars,
                    'subset': subset.replace('-',''),
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

result_dfs = []
for rfp in pert_eval_result_files:
    if 'pertEval' not in rfp:
        continue
    print(rfp)
    _, model, dataset, noise_sev = rfp[:-4].split('/')

    noise, sev, pe_params = noise_sev.split('-')
    try:
        sev = int(sev)
    except:
        sev = 1
    _, num_samples, num_perturbs = pe_params.split('_')[:3]

    
    df = pd.read_csv(rfp, sep='\t')
    df['model'] = model
    df['augmentation'] = noise
    df['severity'] = sev
    df['dataset'] = dataset
    result_dfs.append(df)
if len(result_dfs) > 0:
    full_result_df = pd.concat(result_dfs)
    full_result_df.to_csv('results/collated_PertRob_results.csv')

results = []
result_dfs = []
for rfp in result_files:
    print(rfp)
    _, model, dataset, noise_sev = rfp[:-4].split('/')
    runid = 0
    if '-' in noise_sev:
        noise, sev = noise_sev.split('-')
        if '_' in sev:
            sev, runid = sev.split('_')
            runid = int(runid)
        else:
            sev = int(sev)
    else:
        noise = noise_sev
        sev = 0

    df = pd.read_csv(rfp, sep='\t', usecols=[1,2,3,4,5])
    df['model'] = model
    df['augmentation'] = noise
    df['severity'] = sev
    df['dataset'] = dataset
    df['runid'] = runid
    result_dfs.append(df)

    if noise == 'universal_adv':
        utt_ids = pd.read_csv('robust_speech/advattack_data_and_results/data/LibriSpeech/csv/test-clean-100.csv')['ID'].values
        print(len(df))
        df = df[df['id'].isin(utt_ids)]
        print(len(df))

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
        'WED': werrs.sum(),
        'CED': cerrs.sum(),
        'nwords': nwords.sum(),
        'nchars': nchars.sum(),
        'runid': runid,
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
results_df.to_csv('results/collated_results_all_models.csv')
# results_df.to_latex('results/collated_results_all_models-normedtrans.tex')

full_result_df = pd.concat(result_dfs+adv_result_dfs)
full_result_df.to_csv('results/full_result_df.csv')
# full_result_df.to_hdf('results/full_result_df-normedtrans.hd5', 'df', mode='w', complevel=9, complib='bzip2')