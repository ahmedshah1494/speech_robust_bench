from transformers import Wav2Vec2ForCTC, AutoProcessor
from iso639 import Lang
import torch
from models.hf import create_model_pipeline as create_hf_model_pipeline

def create_model_pipeline(dataset, model_name, batch_size=1, language='English',**kwargs):
    if language != 'English':
        language = Lang(language).pt2t
        print(language)
        processor = AutoProcessor.from_pretrained(model_name, torch_dtype=torch.float16)
        model = Wav2Vec2ForCTC.from_pretrained(model_name, torch_dtype=torch.float16)
        processor.tokenizer.set_target_lang(language)
        kwargs['tokenizer'] = processor.tokenizer
        kwargs['feature_extractor'] = model_name
        model.load_adapter(language)
        model = model.to(torch.float16)
    else:
        model = model_name
    pipe = create_hf_model_pipeline(dataset, model, batch_size=batch_size, **kwargs)
    return pipe