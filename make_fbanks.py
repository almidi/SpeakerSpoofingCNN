from __future__ import print_function

import os
import numpy
import soundfile as sf

train_phase = 'train' # 'dev' 'eval'

base_dir ='data'

norm_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train_norm')

if train_phase == 'train':
    wav_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train')
    fbank_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train_fbank')
elif train_phase == 'dev':
    wav_dir = os.path.join(base_dir, 'ASVspoof2017_V2_dev')
    fbank_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train_dev')
elif train_phase == 'eval':
    wav_dir = os.path.join(base_dir, 'ASVspoof2017_V2_eval')
    fbank_dir = os.path.join(base_dir, 'ASVspoof2017_V2_train_eval')

frame_size = 0.025
frame_stride = 0.005
NFFT = 2048
nfilt = 64


filenames = [x for x in os.listdir(wav_dir) if x.endswith(".wav")]

fbanks_list = []
total_frames = 0



for filename in filenames:
    print(filename)
    wav_fullpathname = os.path.join(wav_dir, filename)
  
    signal, sample_rate = sf.read(wav_fullpathname) 
    signal_length = signal.shape[0]

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    
    frames *= numpy.hamming(frame_length)

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = numpy.log10(filter_banks)  # 20* dB
    fbanks_list.append(filter_banks)
    total_frames += filter_banks.shape[0]

print('total_frames = ', total_frames)
 

if not os.path.exists(fbank_dir):
    os.makedirs(fbank_dir)

for filter_banks, filename in zip(fbanks_list, filenames):
    filter_banks = filter_banks.astype('float32')
    fbank_fullpathname = os.path.join(fbank_dir, filename[:-4]+'.cmp')
    with open(fbank_fullpathname, 'wb') as fid:
        filter_banks.tofile(fid)

# import matplotlib.pyplot as plt
# plt.plot(filter_banks[:, 10])
# plt.show()



