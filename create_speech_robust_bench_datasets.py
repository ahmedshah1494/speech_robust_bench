from argparse import ArgumentParser
from corruption_info import AUGMENTATIONS_2_SEV
import os 
from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument('--dataset', default="librispeech_asr")
parser.add_argument('--subset', default=None)
parser.add_argument('--split', default='test.clean')
parser.add_argument('--hf_repo', required=True, help='HuggingFace repo to push the transformed dataset to.')
parser.add_argument('--run_voice_conversion', action='store_true')
parser.add_argument('--text_field')
args = parser.parse_args()

for aug in AUGMENTATIONS_2_SEV.keys():
    if aug.startswith('voice_conversion') and (not args.run_voice_conversion):
        continue
    if aug == 'universal_adv':
        continue
    for sev in range(1, 5):

        cmd = f'python create_transformed_datasets.py --dataset librispeech_asr --augmentation {aug}:{sev} --dataset {args.dataset} --split {args.split} --hf_repo {args.hf_repo}'
        if args.subset is not None:
            cmd += f' --subset {args.subset}'
        if args.text_field is not None:
            cmd += f' --text_field {args.text_field}'
        os.system(cmd)