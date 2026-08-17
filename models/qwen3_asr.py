import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def create_model_pipeline(dataset, model_name, batch_size=1, language='English', **kwargs):
    device = kwargs.get('device')
    device_map = kwargs.get('device_map')

    load_kwargs = {'torch_dtype': torch.bfloat16}
    if device_map is not None:
        load_kwargs['device_map'] = device_map
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **load_kwargs)
    if device_map is None:
        model = model.to(device or 'cuda:0')
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    def transcribe():
        n = len(dataset)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = dataset[start:end]
            wavs = [a['array'] for a in batch['audio']]
            langs = [language] * len(wavs)

            inputs = processor.apply_transcription_request(audio=wavs, language=langs)
            inputs = inputs.to(model.device, model.dtype)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=256)

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            texts = processor.decode(generated_ids, return_format="transcription_only")
            if isinstance(texts, str):
                texts = [texts]
            for t in texts:
                yield {'text': t}

    return transcribe()
