# coding=utf-8
# import python.src.read_data as rd
from scipy.interpolate import interp1d
import os
import numpy as np
from scipy.signal import lfilter,find_peaks

input_path = "01_input/03_rpm"
output_path = input_path.replace('01_input', '02_output')
if not os.path.exists(output_path):
    os.mkdir(output_path)
file_list = os.listdir(input_path)

for filename in file_list:
    rpm_list = []
    with open(os.path.join(input_path, filename), 'r', encoding='utf-8') as rpm_input:
        first_line = rpm_input.readline()
        first_line = rpm_input.readline()
        first_line = first_line.encode('utf-8').decode('utf-8-sig')
        rpm = first_line.split(',')
        #rpm_list.append(float(rpm))
        total_index = len(rpm)
        rpm = rpm[total_index-2]
        rpm_list.append(float(rpm))
        for line in rpm_input.readlines():
            rpm = line.split(',')[total_index-2]
            if rpm != '':
                rpm_list.append(float(rpm))
                
    window = 3
    breath = np.zeros(len(rpm_list));
    for i in range(len(rpm_list)):
        if i < window:
            pwind = rpm_list[0: i + window]
        elif i > len(rpm_list) - window:
            pwind = rpm_list[i - window : ]
        else:
            pwind = rpm_list[i - window : i + window]
        breath[i] = np.mean(pwind)
    
        
    #breath
    peaks_breath,_ = find_peaks(breath, width=4)
    
    rpm = peaks_breath[1:] - peaks_breath[0:-1]
    rpm = 60 / (rpm/10)
    rpm_t = (peaks_breath[1:] + peaks_breath[0:-1])/20
    
    start = int(np.ceil(rpm_t[0]))
    end = int(np.floor(rpm_t[-1]))

    #x = np.linspace(0, len(rpm_list) - 1, len(rpm_list))
    #f = interp1d(x, rpm_list, kind='cubic')
    f = interp1d(rpm_t, rpm, kind = 'cubic')
    x_new = np.linspace(start, end, end - start+1)
    rpm_new = f(x_new)
    
    rpm_final = np.zeros(int(np.floor(len(rpm_list)/10)))
    rpm_final[start: end+1] = rpm_new
    rpm_final[:start] = rpm_new[0]
    rpm_final[end:] = rpm_new[-1]
    
    
    with open(os.path.join(output_path, filename), 'w') as rpm_output:
        for rpm_o in rpm_final:
            rpm_output.write(str(rpm_o) + '\n')
            


