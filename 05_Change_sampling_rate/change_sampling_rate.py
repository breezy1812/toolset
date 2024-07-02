import python.src.read_data as rd
from scipy.interpolate import interp1d
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, lfilter, butter
input_path = "Data/input/"
output_path = input_path.replace('input', 'output')
file_list = os.listdir(input_path)
origin_rate = 240
target_rate = 64
plot_fig = False

def butter_lowpass(highcut, fs, order):
    """Objective: generate butterworth lowpass filter coefficient.
        @param highcut: the upper frequency
        @param fs: sampling rate
        @param order: The order of the filter
        @return b,a coefficient of the filter
    """
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a

for filename in file_list:
    pressure_data, acc_x, acc_y, acc_z = rd.read_pressure_acc(input_path, filename, down_scale=1)
    x = np.linspace(0, len(pressure_data), len(pressure_data))
    
    bpm_b, bpm_a = butter_lowpass(target_rate / 2, origin_rate, 8)
    pressure_data = lfilter(bpm_b, bpm_a, pressure_data)
    acc_x = lfilter(bpm_b, bpm_a, acc_x)
    acc_y = lfilter(bpm_b, bpm_a, acc_y)
    acc_z = lfilter(bpm_b, bpm_a, acc_z)

    f_pressure = interp1d(x, pressure_data, kind='cubic')
    f_acc_x = interp1d(x, acc_x, kind='cubic')
    f_acc_y = interp1d(x, acc_y, kind='cubic')
    f_acc_z = interp1d(x, acc_z, kind='cubic')
    x_new = np.linspace(0, len(pressure_data), int(len(pressure_data) / origin_rate * target_rate))
    pressure_new = f_pressure(x_new).astype(int)
    new_acc_x = f_acc_x(x_new).astype(int)
    new_acc_y = f_acc_y(x_new).astype(int)
    new_acc_z = f_acc_z(x_new).astype(int)
    if plot_fig:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(pressure_data)
        plt.subplot(2, 1, 2)
        plt.plot(pressure_new)
        plt.savefig(os.path.join(output_path, filename.replace('.log', '.png')))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(os.path.join(output_path, filename), 'w') as output:
        for i in range(len(pressure_new)):
            output.write(str(i) + ',' + str(pressure_new[i]) + ',' + str(new_acc_x[i]) + ',' + str(new_acc_y[i]) + ',' + str(new_acc_z[i]) + '\n')
