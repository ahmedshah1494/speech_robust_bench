import os
import argparse

def _merge(files, outfile):
    text = []
    for fp in files:
        with open(fp) as f:
            text.append(f.readlines())

    words = 0
    werrs = 0
    ins = 0
    sub = 0
    Del = 0
    sentences = 0
    serrs = 0
    for t in text:
        split = t[0].split()
        werrs += int(split[3])
        words += int(split[5][:-1])
        ins += int(split[6])
        Del += int(split[8])
        sub += int(split[10])

        split = t[1].split()
        serrs += int(split[3])
        sentences += int(split[5])

    new_text = f'%WER {100 * werrs / words: .2f} [ {werrs} / {words}, {ins} ins, {Del} del, {sub} sub ]\n'
    new_text += f'%SER {100 * serrs / sentences: .2f} [ {serrs} / {sentences} ]\n'
    new_text += f'Scored {sentences} sentences, 0 not present in hyp.\n'
    new_text += ''.join(text[0][3:])
    for t in text[1:]:
        new_text += ''.join(t[11:])
    with open(outfile, 'w') as f:
        f.write(new_text)

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str)
args = parser.parse_args()

ds2csvnames = {
    'LibriSpeech': ['test-clean-1000', 'test-clean-1000-2000', 'test-clean-2000-2620'],
    'MLS-ES': ['test-clean-1000', 'test-clean-1000-2000', 'test-clean-2000-2380'],
}

for root, dirs, files in os.walk(args.root_dir):
    if 'hyperparams.yaml' in files:
        for metric in ['wer', 'wer_adv', 'cer', 'cer_adv']:
            fn = f'{metric}_test-clean.txt'
            if not (fn in files):
                ds = root.split('/')[-3]
                files = [f'{root}/{metric}_{csvn}.txt' for csvn in ds2csvnames[ds]]
                if all([os.path.exists(f) for f in files]):
                    outfile = f'{root}/{fn}'
                    print(f'Merging {files} into {outfile}')
                    _merge(files, outfile)
                else:
                    print(f'Source files not found in {root}')