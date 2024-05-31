from datasets import load_dataset, Audio
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
import os
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

def load_room_info(root):
    headers = [
        'Room_name', 'Room_length', 'Room_width', 'Room_height', 'Receiver_position_x', 'Receiver_position_y', 'Receiver_position_z', 'Absorption_coefficient'
    ]
    room_info = pd.read_csv(f'{root}/room_info', names=headers, sep=' ')
    # room_info['room_size'] = room_info['room_name'].apply(lambda x: x.split('-')[0])
    # room_info['room_number'] = room_info['room_name'].apply(lambda x: x.split('-')[1])
    return room_info

def compute_t60(room_info):
    room_info['Room_volume'] = room_info['Room_length'] * room_info['Room_width'] * room_info['Room_height']
    room_info['Surface_area'] = 2 * (room_info['Room_length'] * room_info['Room_width'] + room_info['Room_length'] * room_info['Room_height'] + room_info['Room_width'] * room_info['Room_height'])
    room_info['RT60'] = 0.161 * room_info['Room_volume'] / (room_info['Surface_area'] * room_info['Absorption_coefficient'])
    return room_info

def file_to_rt60(room_info, rir_dir):
    rows = []
    print('listing rir files')
    for root, dirs, files in tqdm(os.walk(rir_dir)):
        for name in files:
            if name.endswith('wav'):
                rir_file = os.path.join(rir_dir, root, name)
                room_name = rir_file.split('/')[-3].replace('room','')+'-'+rir_file.split('/')[-2]
                rt60 = room_info[room_info['Room_name'] == room_name]['RT60'].values[0]
                r = {
                    'filename': rir_file,
                    'RT60': rt60
                }
                rows.append(r)
    df = pd.DataFrame(rows)
    return df

rir_dir=f'{os.environ["SRB_ROOT"]}/RIRS_NOISES/simulated_rirs'
room_infos = [load_room_info(rir_dir+'/'+room) for room in ['largeroom', 'mediumroom', 'smallroom']]
room_info = pd.concat(room_infos)
room_info = compute_t60(room_info)
file_to_rt60(room_info, rir_dir).to_csv('rir_t60.csv')
# df = pd.DataFrame(room_info)
# df.to_csv('rir_t60.csv')