from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch

def create_model_pipeline(dataset, model, batch_size=1, gen_kwargs={}, **kwargs):
    if model.startswith('openai/whisper'):
        kwargs['chunk_length_s'] = 30
    pipe = pipeline("automatic-speech-recognition", model=model, batch_size=batch_size, torch_dtype=torch.float16, **kwargs, generate_kwargs=gen_kwargs)
    pipe = pipe(KeyDataset(dataset, "audio"))
    return pipe