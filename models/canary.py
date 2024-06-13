from nemo.collections.asr.models import EncDecMultiTaskModel
from scipy.io import wavfile
import os
from tqdm import tqdm

def create_model_pipeline(dataset, batch_size=1, language='English', **kwargs):
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    # if aug == 'universal_adv':
    #     print(f'Adversarial attacks against {args.model_name} are not supported.')
    # else:
    import tempfile, json
    long2short_lang = {
        'English': 'en',
        'Spanish': 'es'
    }
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w') as tmp:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f'writing creating manifest in {tmp.name}...')
            for i, item in tqdm(enumerate(dataset)):
                tmpaudiofile = os.path.join(tmpdir, f"{item['id']}.wav")
                wavfile.write(tmpaudiofile, 16000, item['audio']['array'])
                r = {
                        "audio_filepath": tmpaudiofile,  # path to the audio file
                        "duration": item['audio']['array'].shape[-1],  # duration of the audio, can be set to `None` if using NeMo main branch
                        "taskname": "asr",  # use "s2t_translation" for speech-to-text translation with r1.23, or "ast" if using the NeMo main branch
                        "source_lang": long2short_lang[language],  # language of the audio input, set `source_lang`==`target_lang` for ASR, choices=['en','de','es','fr']
                        "target_lang": long2short_lang[language],  # language of the text output, choices=['en','de','es','fr']
                        "pnc": "yes",  # whether to have PnC output, choices=['yes', 'no']
                        "answer": "na", 
                    }
                tmp.write(json.dumps(r)+'\n')
                tmp.flush()
            transcripts = model.transcribe(paths2audio_files=tmp.name, batch_size=batch_size)
        # def pipe():
        #     for t in transcripts:
        #         yield {'text': t}
        pipe = [{'text':t} for t in transcripts]
        return pipe