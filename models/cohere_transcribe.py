from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import transformers.modeling_utils as _modeling_utils

# CohereLabs/cohere-transcribe-03-2026's bundled remote code defines
# `_keys_to_ignore_on_load_unexpected` as a list, but transformers>=5's
# PreTrainedModel._adjust_missing_and_unexpected_keys ORs it with a set
# (`list | set` -> TypeError). Coerce to a set defensively; harmless no-op
# for any model whose `_keys_to_ignore_on_load_unexpected` is already a set.
_orig_adjust_keys = _modeling_utils.PreTrainedModel._adjust_missing_and_unexpected_keys


def _patched_adjust_keys(self, loading_info):
    if isinstance(self._keys_to_ignore_on_load_unexpected, list):
        self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected)
    return _orig_adjust_keys(self, loading_info)


_modeling_utils.PreTrainedModel._adjust_missing_and_unexpected_keys = _patched_adjust_keys

long2short_lang = {
    'English': 'en',
    'Spanish': 'es',
    'German': 'de',
    'French': 'fr',
}


def create_model_pipeline(dataset, model_name, batch_size=1, language='English', **kwargs):
    if language not in long2short_lang:
        raise ValueError(
            f"cohere_transcribe: no language-code mapping for language={language!r}. "
            f"Supported here: {list(long2short_lang.keys())} "
            f"(model also supports it/pt/nl/pl/el/ar/ja/zh/vi/ko — extend long2short_lang if needed)."
        )
    lang_code = long2short_lang[language]

    device = kwargs.get('device')
    device_map = kwargs.get('device_map')

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {'trust_remote_code': True}
    if device_map is not None:
        load_kwargs['device_map'] = device_map
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **load_kwargs)
    if device_map is None:
        model = model.to(device or 'cuda:0')
    model.eval()

    audio_arrays = [item['audio']['array'] for item in dataset]
    sample_rates = [16000] * len(audio_arrays)

    texts = model.transcribe(
        processor,
        language=lang_code,
        audio_arrays=audio_arrays,
        sample_rates=sample_rates,
        punctuation=True,
        batch_size=batch_size,
    )
    return [{'text': t} for t in texts]
