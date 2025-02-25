from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from torch.utils.data import DataLoader
import torch 

long2short_lang = {
        'English': 'en',
        'Spanish': 'es',
        'German': 'de',
        'French': 'fr'
    }

def create_model_pipeline(dataset, model_name, batch_size=1, language='English',**kwargs):
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name ,trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda')
    processor = AutoProcessor.from_pretrained(model_name ,trust_remote_code=True)

    prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech in {language}:"
    def _collate_fn(batch):
        inputs = processor(text=[prompt]*len(batch), audios=[x['audio']['array'] for x in batch], sampling_rate=16000, return_tensors="pt")
        return inputs
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)
    for inputs in loader:
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_length=256)
        generated_ids = generated_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i in range(batch_size):
            yield {'text': response[i]}