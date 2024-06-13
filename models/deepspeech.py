from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
from deepspeech_pytorch.decoder import GreedyDecoder
import os
import torch
from multiprocessing import cpu_count

N_CPUS = cpu_count()

def create_model_pipeline(dataset, batch_size=1, **kwargs):
    model = DeepSpeech.load_from_checkpoint(f'{os.environ["SRB_ROOT"]}/deepspeech_ckps/librispeech_pretrained_v3.ckpt')
    parser = ChunkSpectrogramParser(audio_conf=model.spect_cfg)
    def extract_features(x):
        waveform = x['audio']['array']
        spec = list(parser.parse_audio(waveform))[0]
        x['spec'] = spec
        x['lengths'] = spec.shape[1]
        return x
    dataset = dataset.map(extract_features, batched=False, num_proc=N_CPUS//4)

    def collate_fn(batch):
        specs = [torch.FloatTensor(batch[i]['spec']).transpose(0,1) for i in range(len(batch))]
        lengths = torch.LongTensor([(batch[i]['lengths']) for i in range(len(batch))])
        specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True).unsqueeze(1).transpose(2,3)
        return {'spec': specs, 'lengths': lengths}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=N_CPUS//4, shuffle=False)
    decoder = GreedyDecoder(model.labels)
    def transcribe(dataloader):
        for batch in dataloader:
            length_order = torch.argsort(batch['lengths'], descending=True)
            reverse_length_order = torch.argsort(length_order)
            batch['spec'] = batch['spec'][length_order]
            batch['lengths'] = batch['lengths'][length_order]
            out, lens, _ = model(batch['spec'].cuda(), batch['lengths'].cuda())                
            decoded_output, decoded_offsets = decoder.decode(out, lens)
            for i in reverse_length_order:
                yield {'text': decoded_output[i][0]}
    pipe = transcribe(dataloader)
    return pipe