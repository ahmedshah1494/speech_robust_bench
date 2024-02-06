from argparse import ArgumentParser
from create_transformed_datasets import AUGMENTATIONS, UNIV_ADV_DELTAS
import os 

parser = ArgumentParser()
parser.add_argument('--dataset', default="librispeech_asr")
parser.add_argument('--subset', default=None)
parser.add_argument('--split', default='test.clean')
args = parser.parse_args()

for aug in AUGMENTATIONS.keys():
    for sev in range(1, 4):
        cmd = f'python create_transformed_datasets.py --dataset librispeech_asr --augmentation {aug}:{sev} --dataset {args.dataset} --split {args.split}'
        if args.subset is not None:
            cmd += f' --subset {args.subset}'
        if aug == 'universal_adv':
            for delta_path in UNIV_ADV_DELTAS:
                os.system(cmd + f' --universal_delta_path {delta_path}')
        else:
            os.system(cmd)