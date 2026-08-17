def create_model_pipeline(model, dataset, batch_size=1, language='English', **kwargs):
    if model == 'deepspeech':
        from models import deepspeech
        return deepspeech.create_model_pipeline(dataset, batch_size=batch_size, **kwargs)
    elif model == 'rnnt':
        from models import rnnt
        return rnnt.create_model_pipeline(dataset, batch_size=batch_size, **kwargs)
    elif model.startswith('sb_') or model.startswith('speechbrain/'):
        from models import speechbrain
        return speechbrain.create_model_pipeline(model, dataset, batch_size=batch_size, **kwargs)
    elif model.startswith('nvidia/canary'):
        from models import canary
        return canary.create_model_pipeline(dataset, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('nvidia/parakeet'):
        from models import parakeet
        return parakeet.create_model_pipeline(model, dataset, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('openai/whisper'):
        from models import whisper
        return whisper.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('facebook/mms'):
        from models import mms
        return mms.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('Qwen/Qwen2-Audio-7B'):
        from models import qwen_audio
        return qwen_audio.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('ibm-granite/granite-speech'):
        from models import granite_speech
        return granite_speech.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('Qwen/Qwen3-ASR'):
        from models import qwen3_asr
        return qwen3_asr.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    elif model.startswith('CohereLabs/cohere-transcribe'):
        from models import cohere_transcribe
        return cohere_transcribe.create_model_pipeline(dataset, model, batch_size=batch_size, language=language, **kwargs)
    else:
        from models import hf
        return hf.create_model_pipeline(dataset, model, batch_size=batch_size, **kwargs)