from time import sleep
from datasets import load_dataset, DownloadMode
from requests.exceptions import HTTPError

srb_hf_repo = 'mshah1/speech_robust_bench_public'
dataset = 'LIUM/tedlium'
subset = 'release3'
split = 'test'

config = f'{dataset.split("/")[-1]}-{subset}_{split}'
full_dataset = load_dataset(srb_hf_repo, config, ignore_verifications=True)
for split, dataset in full_dataset.items():
    print(split, dataset)
    if len(dataset) > 1155:
        dataset = dataset.filter(lambda x: not x['id'].startswith('inter_segment_gap'))
        backoff = 10
        while True:
            try:
                dataset.push_to_hub(srb_hf_repo, config, split=split)
                break
            except HTTPError as e:
                print(f'ecountered HTTPError, sleeping for {backoff} seconds...')
                sleep(backoff)
                backoff *= 1.5
            
# exit()