# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:34:44 2020

@author: breez
"""
import os
import numpy as np
from scipy.signal import lfilter,find_peaks, firwin
import matplotlib.pyplot as plt
import biosppy
import scipy


def FIR_filter(fs, lowcut, highcut, data, numtaps=100):
    if isinstance(data, (np.ndarray, list)):
        if fs is None:
            raise ValueError("sampling frequency must be specified!")
        fir_tap = numtaps
        bpm_b = firwin(fir_tap, [lowcut, highcut], pass_zero=False, fs=fs)
        bpm_a = 1.0
        ripple_data = lfilter(bpm_b, bpm_a, data)
        return np.array(ripple_data)
    else:
        raise TypeError(
            "Unknown data type {} to filter.".format(str(type(data))))


input_path = "DATA"
filename = '2021-03-10-15-07_ECG.log'
fs = 128 
file_list = os.listdir(input_path)
ecg_index = 1
starttime = ''
ecg_list = []
get_bpm = False
char_inter = '\t'

with open(os.path.join(input_path, filename), 'r', encoding='utf-8') as rpm_input:
    count = 0
    while count <= 3:
        line = rpm_input.readline()
        count += 1
        if 'Start_Time' in line:
            starttime = line
    first_line = line.encode('utf-8').decode('utf-8-sig')
    if ',' in first_line:
        char_inter = ','
    
    data_list = first_line.split(char_inter)
    try:
        bpm = data_list[ecg_index]        
        ecg_list.append(float(bpm))
        get_bpm = True
    except (TypeError, ValueError):
        pass
    
        
        
    for line in rpm_input.readlines():
        if 'End' in line:
            break
        elif 'State' in line:
            continue
        elif 'Time' in line:
            continue

        if get_bpm:
            
            bpm = line.split(char_inter)[ecg_index]
            ecg = (float(bpm))
            ecg_list.append(ecg)
#ecg_list = FIR_filter(fs, 2, 100, ecg_list)
_, _, s  = scipy.signal.stft(ecg_list, window='hamming', nperseg=fs, noverlap=fs/2, nfft=fs*10 , boundary=None)
S = np.abs(s)
S[:5] = 0
plt.pcolormesh(S)

plt.figure()
X = np.linspace(0.5, len(ecg_list)-0.5, len(ecg_list))
plt.plot(X, ecg_list, color='blue')
engine_peaks, _ = find_peaks(ecg_list, height=0.3)
interval = engine_peaks[1:] - engine_peaks[:-1]
plt.show()