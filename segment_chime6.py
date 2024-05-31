import json
import argparse
from scipy.io import wavfile
import os
import pandas as pd
import re
from tqdm import tqdm

def timestamp_to_frame(timestamp, fs):
    h, m, s = timestamp.split(':')
    total_s = int(h) * 3600 + int(m) * 60 + float(s)
    return int(total_s * fs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chime_dir', type=str, help='Path to the CHiME6 dataset directory', required=True)
    args = parser.parse_args()

    split = 'eval'
    audio_dir = f'{args.chime_dir}/audio/{split}'
    transcript_dir = f'{args.chime_dir}/transcriptions/{split}'
    output_dir = f'{args.chime_dir}/segmented_audio/{split}'

    print(f'Processing {split} set')
    print(f'Searching for transcripts...')
    transcripts = []
    for root, dirs, files in os.walk(transcript_dir):
        for name in files:
            if name.endswith('json'):
                transcript_file = os.path.join(transcript_dir, root, name)
                with open(transcript_file) as f:
                    transcripts.extend(json.load(f))
    transcript_df = pd.DataFrame(transcripts)
    print(f'Found {len(transcript_df)} transcripts')


    print(f'Searching for audio files...')
    audios = []
    for root, dirs, files in os.walk(audio_dir):
        for name in files:
            if name.endswith('wav'):
                audio_file = os.path.join(audio_dir, root, name)
                audios.append(audio_file)
    print(f'Found {len(audios)} audio files')
    
    print(f'Processing audio files...')
    metadata = []
    for audio_file in audios:
        print(f'Processing {audio_file}')
        fnsplit = os.path.basename(audio_file).split('.')
        session_id = fnsplit[0].split('_')[0]
        if len(fnsplit) == 3:
            channel = fnsplit[1]
            if int(channel.replace('CH','')) > 1:
                print(f'Skipping channel {channel}')
                continue
            device_id = fnsplit[0].split('_')[1]
            spk_id = None
            output_prefix = f'{output_dir}/farfield/{session_id}_{device_id}_{channel}'
            sub_transcript = transcript_df[
                (transcript_df['session_id'] == session_id) & 
                (transcript_df['ref'] == device_id)
            ]
            if len(sub_transcript) == 0:
                print(f'No transcript found for {session_id}_{device_id}_{channel}. Skipping')
                continue
        else:
            channel = None
            spk_id = fnsplit[0].split('_')[1]
            device_id = None
            output_prefix = f'{output_dir}/nearfield/{session_id}_{spk_id}'
            sub_transcript = transcript_df[
                (transcript_df['session_id'] == session_id) & 
                (transcript_df['speaker'] == spk_id)
            ]
            if len(sub_transcript) == 0:
                print(f'No transcript found for {session_id}_{spk_id}. Skipping')
                continue
        
        fs, audio = wavfile.read(audio_file)

        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        
        print(f'Found {len(sub_transcript)} segments in transcript')
        print(f'Saving segments to {output_prefix}_[start_time]-[end-time]')
        for i, row in tqdm(sub_transcript.iterrows()):
            # convert time stamps to frame ids and extract segment
            start_frame = timestamp_to_frame(row['start_time'], fs)
            end_frame = timestamp_to_frame(row['end_time'], fs)
            segment = audio[start_frame:end_frame]

            text = row['words']
            # remove non-speech markers
            text = re.sub("\[.*?\]", '', text)

            # skip if transcript has one word or less, or if the segment is less than 1s
            if len(text.split()) <= 1 or len(segment) < fs:
                continue

            # write segment to file
            output_file = f"{output_prefix}_{row['start_time'].translate(str.maketrans('', '', ':.'))}-{row['end_time'].translate(str.maketrans('', '', ':.'))}.wav"
            if not os.path.exists(output_file):
                wavfile.write(output_file, fs, segment)

            metarow = row.to_dict()
            metarow['file_name'] = '/'.join(output_file.split('/')[-2:])
            metarow['id'] = os.path.basename(metarow['file_name']).split('.')[0]
            metarow['text'] = text
            metarow.pop('words')
            metadata.append(metarow)

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(f'{output_dir}/metadata.csv', index=False)
            