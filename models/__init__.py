from models import canary
from models import deepspeech
from models import hf
from models import mms
from models import rnnt
from models import whisper

def create_model_pipeline(model, dataset, batch_size=1, language='English', **kwargs):
    if model == 'deepspeech':
        return deepspeech.create_model_pipeline(dataset, batch_size=batch_size, **kwargs)
    elif model == 'rnnt':
        return rnnt.create_model_pipeline(dataset, batch_size=batch_size, **kwargs)
    elif model.startswith('nvidia/canary'):
        return canary.create_model_pipeline(dataset, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('openai/whisper'):
        return whisper.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('facebook/mms'):
        return mms.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    else:
        return hf.create_model_pipeline(dataset, model, batch_size=batch_size, **kwargs)