import numpy as np

NOISE_SNRS = [40, 30, 20, 10, 0]
ADV_SNRS = [50, 40, 30, 20, 10]
SPEEDUP_FACTORS = [1, 1.25, 1.5, 1.75, 2]
SLOWDOWN_FACTORS = [1, 0.875, 0.75, 0.625, 0.5]
PITCH_UP_STEPS = [0, 3, 6, 9, 12]
PITCH_DOWN_STEPS = [0, -3, -6, -9, -12]
RESAMPLING_FACTORS = [1, 0.75, 0.5, 0.25, 0.125]
GAIN_FACTORS = [0, 10, 20, 30, 40]
ECHO_DELAYS = [0, 125, 250, 500, 1000]
PHASER_DECAYS = [0.1, 0.3, 0.5, 0.7, 0.9]
LOWPASS_FREQS = [8000] + np.linspace(4000, 500, 4).astype(int).tolist()
HIGHPASS_FREQS = [0] + np.linspace(500, 3000, 4).astype(int).tolist()
TREMOLO_DEPTHS = [0] + np.linspace(50, 100, 4).astype(int).tolist()
TREBLE_GAIN = [1] + np.linspace(10, 50, 4).astype(int).tolist()
BASS_GAIN = [1] + np.linspace(20, 50, 4).astype(int).tolist()
CHORUS_DELAY = [0, 30, 50, 70, 90]
VC_ACCENTS = [[], ['bdl', 'slt', 'rms', 'clb'], ['jmk'], ['ksp'], ['awb']]
# VC_VCTK_ACCENTS = [['English'], ['Scottish'], ['NorthernIrish'], ['Irish'], ['Indian'], ['Welsh'],
#        ['American'], ['Canadian'], ['SouthAfrican'], ['Australian'],
#        ['NewZealand'], ['British']]
VC_VCTK_ACCENTS = [[], ['English', 'Scottish', 'NorthernIrish', 'Irish', 'Indian', 'Welsh',
                        'American', 'Canadian', 'SouthAfrican', 'Australian',
                        'NewZealand', 'British']]

AUGMENTATIONS_2_SEV = {
    # 'unoise': (UniformNoise, NOISE_SNRS),
    'gnoise':  NOISE_SNRS,
    'env_noise':  NOISE_SNRS,
    'env_noise_esc50':  NOISE_SNRS,
    'env_noise_musan':  NOISE_SNRS,
    'env_noise_wham':  NOISE_SNRS,
    'speedup':  SPEEDUP_FACTORS,
    'slowdown':  SLOWDOWN_FACTORS,
    'pitch_up':  PITCH_UP_STEPS,
    'pitch_down':  PITCH_DOWN_STEPS,
    'rir':  [0,1,2,3,4],
    'real_rir':  [0,1,2,3,4],
    'voice_conversion_vctk':  VC_VCTK_ACCENTS,
    'voice_conversion_bark': [None, None],
    'resample':  RESAMPLING_FACTORS,
    'gain':  GAIN_FACTORS,
    'echo':  ECHO_DELAYS,
    'phaser':  PHASER_DECAYS,
    'tempo_up':  SPEEDUP_FACTORS,
    'tempo_down':  SLOWDOWN_FACTORS,
    'lowpass':  LOWPASS_FREQS,
    'highpass':  HIGHPASS_FREQS,
    'music':  NOISE_SNRS,
    'crosstalk':  NOISE_SNRS,
    'tremolo':  TREMOLO_DEPTHS,
    'treble':  TREBLE_GAIN,
    'bass':  BASS_GAIN,
    'chorus':  CHORUS_DELAY,
    'universal_adv': ADV_SNRS,
}

PERT_ROB_AUGMENTATIONS_2_SEV = {
    'gnoise': NOISE_SNRS,
    'env_noise':  NOISE_SNRS,
    'env_noise_esc50':  NOISE_SNRS,
    'env_noise_musan':  NOISE_SNRS,
    'env_noise_wham':  NOISE_SNRS,
    'music':  NOISE_SNRS,
    'rir':  [0,1,2,3,4],
    'real_rir':  [0,1,2,3,4],
}