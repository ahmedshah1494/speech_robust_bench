from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecCTCModelBPE
from scipy.io import wavfile
import os
from tqdm import tqdm

def create_model_pipeline(model, dataset, batch_size=1, language='English', **kwargs):
    if 'ctc' in model:
        model_class = EncDecCTCModelBPE
    elif 'rnnt' in model:
        model_class = EncDecRNNTBPEModel
    else:
        raise ValueError(f'Unsupported model {model}')
    model = model_class.from_pretrained(model)
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    # if aug == 'universal_adv':
    #     print(f'Adversarial attacks against {args.model_name} are not supported.')
    # else:

    tmpdir = os.environ.get('LOCAL', f"/local/slurm-{os.environ['SLURM_JOB_ID']}/tmp/")
    if not os.path.exists(tmpdir):
        tmpdir = None
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w') as tmp:
        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
            print(f'writing copying audio files to {tmpdir}...')
            audiofiles = []
            for i, item in tqdm(enumerate(dataset)):
                tmpaudiofile = os.path.join(tmpdir, f"{item['id']}.wav")
                wavfile.write(tmpaudiofile, 16000, item['audio']['array'])
                audiofiles.append(tmpaudiofile)
            transcripts = model.transcribe(paths2audio_files=audiofiles, batch_size=batch_size)
            if isinstance(transcripts, tuple):
                transcripts = transcripts[0]
            assert isinstance(transcripts, list)
        # def pipe():
        #     for t in transcripts:
        #         yield {'text': t}
        pipe = [{'text':t} for t in transcripts]
        return pipe