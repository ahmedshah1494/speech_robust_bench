from transformers import WhisperProcessor
from models.hf import create_model_pipeline as create_hf_model_pipeline

def create_model_pipeline(dataset, model, batch_size=1, language='English',**kwargs):
    gen_kwargs = {}
    if language != 'English':
        processor = WhisperProcessor.from_pretrained(model)
        gen_kwargs = {'forced_decoder_ids': processor.get_decoder_prompt_ids(language=language.lower(), task="transcribe")}
    pipe = create_hf_model_pipeline(dataset, model, batch_size=batch_size, gen_kwargs=gen_kwargs, **kwargs)
    return pipe