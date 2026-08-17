import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def create_model_pipeline(dataset, model_name, batch_size=1, language='English', **kwargs):
    if language != 'English':
        print(f"WARNING: granite_speech has no documented language-selection API; "
              f"requested language={language!r} will be ignored and English transcription assumed.")

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
    tokenizer = processor.tokenizer

    user_prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
    chat = [{"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def transcribe():
        n = len(dataset)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = dataset[start:end]
            wavs = [torch.tensor(a['array'], dtype=torch.float32) for a in batch['audio']]
            prompts = [prompt] * len(wavs)

            model_inputs = processor(prompts, wavs, device=str(model.device), return_tensors="pt")
            model_inputs = model_inputs.to(model.device)

            with torch.no_grad():
                model_outputs = model.generate(
                    **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
                )

            # tokenizer.padding_side == 'left' (confirmed in cached tokenizer_config.json),
            # so every row in the batch has the same input_ids length after padding.
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = model_outputs[:, num_input_tokens:]
            texts = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
            for t in texts:
                yield {'text': t}

    return transcribe()
