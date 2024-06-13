import torchaudio
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
import torch

def create_model_pipeline(dataset, batch_size=1, **kwargs):
    feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
    decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder().cuda()
    token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()

    def extract_features(x):
        waveform = torch.FloatTensor(x['audio']['array'])
        spec, length = feature_extractor(waveform)
        x['spec'] = spec
        x['lengths'] = length[0]
        return x
    dataset = dataset.map(extract_features, batched=False, num_proc=4)

    def collate_fn(batch):
        specs = [torch.FloatTensor(batch[i]['spec']) for i in range(len(batch))]
        lengths = torch.LongTensor([(batch[i]['lengths']) for i in range(len(batch))])
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
        return {'spec': specs, 'lengths': lengths}      
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False)

    def transcribe(dataloader):
        for batch in dataloader:
            with torch.no_grad():
                for feature, length in zip(batch['spec'].cuda(), batch['lengths'].cuda()):
                    hypotheses = decoder(feature.unsqueeze(0), length.unsqueeze(0), 1)
                    text = token_processor(hypotheses[0][0])
                    yield {'text': text}
    pipe = transcribe(dataloader)
    return pipe