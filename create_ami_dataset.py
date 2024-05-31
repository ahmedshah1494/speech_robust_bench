from datasets import load_dataset, Audio

for final_name, subset in [('nearfield','ihm'), ('farfield','sdm')]:
    dataset = load_dataset('edinburghcstr/ami', subset, split='test')
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    dataset = dataset.filter(lambda x: len(x['text'].split()) > 3)
    dataset.push_to_hub('mshah1/speech_robust_bench', 'in-the-wild-ami', split=final_name)
