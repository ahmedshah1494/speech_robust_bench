from datasets import load_dataset, Audio, concatenate_datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import evaluate
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import os
from copy import deepcopy
import string
from corruptions import *
from create_transformed_datasets import load_augmentation, transform_dataset, parse_augmentation
from multiprocessing import cpu_count

N_CPUS = cpu_count()
N_GPUS = torch.cuda.device_count()

def normalize_transcript(txt):
    txt = txt.lower()
    puncs = list(string.punctuation)
    for pnc in puncs:
        txt = txt.replace(pnc, '')
    return txt

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True, help='Model name or path compatible with HuggingFace Transformers library.')
    parser.add_argument('--dataset', default="librispeech_asr", help='Name for dataset to load from huggingface hub. Used to run eval on clean data and utterance agnostic (universal) adversarial perturbations. default: librispeech_asr.')
    parser.add_argument('--srb_hf_repo', default='mshah1/speech_robust_bench', help='Huggingface repo name for the preprocessed speech robustness benchmark. default: mshah1/speech_robust_bench')
    parser.add_argument('--subset', default=None, help='Subset of the dataset to use. default: None')
    parser.add_argument('--split', default='test.clean', help='Split of the dataset to use. default: test.clean')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--augmentation', type=str, help='Augmentation to apply to the dataset. Should be of the form <aug>:<sev>, where <aug> is a key in corruptions.AUGMENTATIONS, and <sev> is the severity in range 1-4 (except for voice_conversion_vctk for which it should be 1). default: None')
    parser.add_argument('--universal_delta_path', type=str, help='Path to the universal adversarial perturbation. default: None')
    parser.add_argument('--language', default='english', help='Language of the dataset. This is needs to be correctly specified for multi-lingual models. default: english')
    parser.add_argument('--output_dir', default='outputs', help='Output directory for the results. default: outputs')
    parser.add_argument('--model_parallelism', action='store_true', help='Use model parallelism for the model. default: False')
    parser.add_argument('--run_perturb_robustness_eval', action='store_true', help='Run prediction stability analysis. default: False')
    parser.add_argument('--n_perturb_per_sample', type=int, default=30, help='Number of perturbations to generate per sample for stability analysis. default: 30')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples to use for stability analysis. default: 500')
    parser.add_argument('--overwrite_result_file', action='store_true', help='Overwrite the result file if it exists. default: False')
    parser.add_argument('--skip_if_result_exists', action='store_true', help='Skip the evaluation if the result file exists. default: False')
    args = parser.parse_args()

    aug, sev = parse_augmentation(args)

    odir = f'{args.output_dir}/{args.model_name.split("/")[-1]}/{args.dataset.split("/")[-1]}'
    os.makedirs(odir, exist_ok=True)

    if args.augmentation == 'universal_adv':
        aug = f'{aug}_{args.universal_delta_path.split("/")[-3]}'
    ofn = f'{aug}-{sev}'
    if args.run_perturb_robustness_eval:
        assert args.augmentation is not None
        ofn = f'{ofn}-pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
    ofp = f'{odir}/{ofn}.tsv'
    if not args.overwrite_result_file:
        i = 0
        ofp = f'{odir}/{ofn}_{i}.tsv'
        # print(ofp, os.path.exists(ofp))
        if args.skip_if_result_exists and (os.path.exists(ofp) or ((i == 1) and os.path.exists(f'{odir}/{ofn}.tsv'))):
            print(f'Skipping {ofp}')
            exit()
        while os.path.exists(ofp):
            i += 1
            ofp = f'{odir}/{ofn}_{i}.tsv'

    if (args.augmentation is None) or (aug == 'universal_adv'):
        dataset = load_dataset(args.dataset, args.subset, split=args.split)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        if aug == 'universal_adv':
            transform = load_augmentation(aug, sev, args.universal_delta_path)
            dataset = transform_dataset(dataset, transform)

        print(dataset)
    else:
        subset = f'{args.subset}_{args.split}' if args.subset else args.split
        if args.run_perturb_robustness_eval:
            subset = f'{subset}_pertEval_{args.n_samples}_{args.n_perturb_per_sample}'
        dataset = load_dataset(args.srb_hf_repo, f'{args.dataset.split("/")[-1]}-{subset}', split=f'{aug}.{sev}')

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    if args.model_name == 'deepspeech':
        from deepspeech_pytorch.model import DeepSpeech
        from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
        from deepspeech_pytorch.decoder import GreedyDecoder

        model = DeepSpeech.load_from_checkpoint(f'{os.environ["SRB_ROOT"]}/deepspeech_ckps/librispeech_pretrained_v3.ckpt')
        parser = ChunkSpectrogramParser(audio_conf=model.spect_cfg)
        def extract_features(x):
            waveform = x['audio']['array']
            spec = list(parser.parse_audio(waveform))[0]
            x['spec'] = spec
            x['lengths'] = spec.shape[1]
            return x
        dataset = dataset.map(extract_features, batched=False, num_proc=N_CPUS//4)
        
        def collate_fn(batch):
            specs = [torch.FloatTensor(batch[i]['spec']).transpose(0,1) for i in range(len(batch))]
            lengths = torch.LongTensor([(batch[i]['lengths']) for i in range(len(batch))])
            specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True).unsqueeze(1).transpose(2,3)
            return {'spec': specs, 'lengths': lengths}
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=N_CPUS//4, shuffle=False)
        decoder = GreedyDecoder(model.labels)
        def transcribe(dataloader):
            for batch in dataloader:
                length_order = torch.argsort(batch['lengths'], descending=True)
                reverse_length_order = torch.argsort(length_order)
                batch['spec'] = batch['spec'][length_order]
                batch['lengths'] = batch['lengths'][length_order]
                out, lens, _ = model(batch['spec'].cuda(), batch['lengths'].cuda())                
                decoded_output, decoded_offsets = decoder.decode(out, lens)
                for i in reverse_length_order:
                    yield {'text': decoded_output[i][0]}
        pipe = transcribe(dataloader)
    elif args.model_name == 'rnnt':
        import torchaudio
        from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH

        feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
        decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder().cuda()
        token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()

        def extract_features(x):
            waveform = torch.FloatTensor(x['audio']['array'])
            spec, length = feature_extractor(waveform)
            x['spec'] = spec
            x['lengths'] = length[0]
            return x
        dataset = dataset.map(extract_features, batched=False, num_proc=4)

        def collate_fn(batch):
            specs = [torch.FloatTensor(batch[i]['spec']) for i in range(len(batch))]
            lengths = torch.LongTensor([(batch[i]['lengths']) for i in range(len(batch))])
            specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
            return {'spec': specs, 'lengths': lengths}      
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False)
        
        def transcribe(dataloader):
            for batch in dataloader:
                with torch.no_grad():
                    for feature, length in zip(batch['spec'].cuda(), batch['lengths'].cuda()):
                        hypotheses = decoder(feature.unsqueeze(0), length.unsqueeze(0), 1)
                        text = token_processor(hypotheses[0][0])
                        yield {'text': text}
        pipe = transcribe(dataloader)
    else:
        if args.model_parallelism: 
            kwargs = {'device_map': 'auto'}
        else:
            kwargs = {'device': 'cuda:0'}
        model = args.model_name
        gen_kwargs = {}
        if ('whisper' in args.model_name) and (args.language != 'English'):
            from transformers import WhisperProcessor
            processor = WhisperProcessor.from_pretrained(args.model_name)
            gen_kwargs = {'forced_decoder_ids': processor.get_decoder_prompt_ids(language=args.language.lower(), task="transcribe")}
            print(gen_kwargs)
        if ('mms' in args.model_name) and (args.language != 'English'):
            from transformers import Wav2Vec2ForCTC, AutoProcessor
            processor = AutoProcessor.from_pretrained(args.model_name, torch_dtype=torch.float16)
            model = Wav2Vec2ForCTC.from_pretrained(args.model_name, torch_dtype=torch.float16)
            from iso639 import Lang
            processor.tokenizer.set_target_lang(Lang(args.language).pt2t)
            kwargs['tokenizer'] = processor.tokenizer
            kwargs['feature_extractor'] = args.model_name
            model.load_adapter(args.language)
            model = model.to(torch.float16)
        print(kwargs)
        pipe = pipeline("automatic-speech-recognition", model=model, batch_size=args.batch_size, torch_dtype=torch.float16, **kwargs, generate_kwargs=gen_kwargs)
        pipe = pipe(KeyDataset(dataset, "audio"))
    
    output_rows = []
    t = tqdm(zip(pipe, dataset))
    for out, inp in t:
        hyp = out['text'].upper()
        ref = inp['text'].upper()
        
        ref = normalize_transcript(ref)
        hyp = normalize_transcript(hyp)

        wer = wer_metric.compute(references=[ref], predictions=[hyp])
        cer = cer_metric.compute(references=[ref], predictions=[hyp])
        r = {
            'id': inp['id'],
            'reference': ref,
            'prediction': hyp,
            'wer': wer,
            'cer': cer
        }
        if 'pert_idx' in inp:
            r['pert_idx'] = inp['pert_idx']
        output_rows.append(r)
        t.set_description(f'{args.model_name.split("/")[-1]}\t{aug}:{sev}')
        t.set_postfix(wer=wer, cer=cer)
    df = pd.DataFrame(output_rows)
    df.to_csv(ofp, sep='\t')