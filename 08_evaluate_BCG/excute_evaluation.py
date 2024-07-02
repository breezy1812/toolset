from turtle import color
import numpy as np
from src.alg_freq_domain import Alg_freq_domain
import src.performance as pf
import src.read_data as rd
from scipy.signal import find_peaks
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import subprocess
import csv
import sys
import platform
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "project")))
#from scipy.signal import hilbert, chirp
# from python.src.get_quality import quality_spectrum, get_correlation, get_correlation_atfreq, get_score_spec
#from python.src.RCF import rolling_ball_background 

def on_xlims_change(axes):
    old_tlocs = axes.get_xticks()
    new_tlocs = [i*factor for i in old_tlocs]


def moving_average(a, n=3):
    moving_avg = np.copy(a)
    if n % 2 == 1:
        m = int((n-1)/2)
        for i in range(len(a)):
            temp = a[max(0, i - m) : min(i + m, len(a)-1)]
            moving_avg[i] = np.mean(temp)
    else:
        m = int(n/2)
        m2 = m-1
        for i in range(len(a)):
            temp = a[max(0, i - m) : min(i + m, len(a)-1)] 
            temp2 = a[max(0, i - m2) : min(i + m2, len(a)-1)]
            moving_avg[i] = (sum(temp) + sum(temp2))/(len(temp) + len(temp2))
        
    return moving_avg

def moving_average_downsample(a, n=3):
    moving_avg = []
    window = []
    for i in range(len(a)):
        window.append(a[i])
        if len(window) == n :
            moving_avg.append(int(np.mean(window)))
            window = []
        
    return np.array(moving_avg)

def down_sample(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)




compute_MCU_pf = 0
show_first_minute = 0
csv_isexist = False
shift_manual = 8

ACC_FEATURE = 0  
read_old = 0
writ_old = 0
sample_rate = 60
down_scale = 1
FFT_window_size= 64
shift = 0
filepath = "input"
filename = "1號氣袋2022_07_05_13_36_31.log"
filename = "3號氣袋2022_07_05_13_48_28.log"
# filename = "5號氣袋2022_07_05_13_54_36.log"
print("filename:", filename)

if __name__ == '__main__':
    pressure_data, acc_x, acc_y, acc_z, raw_status, raw_hr,timestamp, start_time = rd.read_pressure_acc_result(filepath, filename, down_scale)   
    algo = Alg_freq_domain(fs=sample_rate, fft_window_size=FFT_window_size)
    time = np.linspace(0, len(pressure_data) / sample_rate, len(pressure_data))
    for i in range(len(acc_y)):
        acc_y[i] = 65535 - acc_y[i] if  acc_y[i]> 50000 else acc_y[i]
        acc_z[i] = 65535 - acc_z[i] if  acc_z[i]> 50000 else acc_z[i]
        acc_x[i] = 65535 - acc_x[i] if  acc_x[i]> 50000 else acc_x[i]

    algo.main_func(pressure_data, acc_x, acc_y, acc_z)
    algo_data = algo.bpm
    algo_status = algo.status
    conf_level = algo.confidence_level
    overlap_ss = algo.ss_denoise_overlaped
    ss_denoise = algo.ss_denoise

        
    window_size_minute = 2
    raw_status_down = algo.down_sample(raw_status, 2)
    temp = algo.filter_p_sensor_data[raw_status_down == 1]
    first_std = np.std(temp)
    temp = temp[abs(temp - np.mean(temp)) <  first_std ]
    
    print("first " + str(window_size_minute) +"minutes' STD = " + 
        str(np.std(temp))) 

    plt.figure()
    plt.plot(range(len(temp)), temp)




    if '.txt' in filename:
        ground_truth_filename = filename.replace('.txt', '.csv')
    else:
        ground_truth_filename = filename.replace('.log', '.csv')
    if os.path.isfile(os.path.join(filepath, ground_truth_filename)):
        csv_isexist = True 
        
    algo_data = moving_average(algo_data, 3)
    raw_hr = moving_average(raw_hr, 3)
    if csv_isexist:    
        ground_truth_bpm, start_time_golden = rd.read_csv_with_time(os.path.join(filepath, ground_truth_filename))
        ground_truth_bpm = moving_average(ground_truth_bpm, 3)
        if len(start_time_golden) > 0 and len(start_time) > 5:
            if len(start_time_golden) == 6 :
                shift = round(start_time[5] - start_time_golden[5]) + 1 + shift_manual
                if shift < 0:
                    test_data = algo_data
                    ground_truth = ground_truth_bpm[shift * -1 :]
                else:
                    test_data = algo_data
                    ground_truth = ground_truth_bpm[shift:]
                    if len(ground_truth) > len(test_data):
                        ground_truth = ground_truth[:len(test_data)]
                    else:
                        test_data = test_data[:len(ground_truth)]
            else:
                test_data = algo_data[:]
                ground_truth = ground_truth_bpm[:]
        else:
            shift, test_data, ground_truth = pf.get_optimal_acc_rate(algo_data, ground_truth_bpm, 5, max_shift=5)

        if shift < 0:
            bpm = algo_data[ shift * -1:]
        else:
            bpm = algo_data
            
        print("performance (Acc rate(AP3):", pf.performance( test_data, ground_truth, bpm_error_tolerant=3)[0])
        print("performance (Acc rate(AP5):", pf.performance( test_data, ground_truth, bpm_error_tolerant=5)[0])
        print("performance (Acc rate(AP10):", pf.performance( test_data, ground_truth, bpm_error_tolerant=10)[0])
    plt.show()
