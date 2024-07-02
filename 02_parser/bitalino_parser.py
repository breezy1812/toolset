# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:34:44 2020

@author: breez
"""

#%%

#%%
from scipy.interpolate import interp1d
import os
import numpy as np
from scipy.signal import lfilter,find_peaks
import matplotlib.pyplot as plt
import biosppy
import json
import plotly.express as px

def moving_average(a, n=3):
    moving_avg = np.copy(a)
    
    m = int((n-1)/2)
    for i in range(len(a)):
        temp = a[max(0, i - m) : min(i + m, len(a)-1)]
        temp = np.sort(temp)
        moving_avg[i] = np.mean(temp[1:-1])
    
    return moving_avg
def show_image():
    plt.ioff()
    plt.show()


input_path = "01_input/04_bitalino"
sample_rate = 100
output_path = input_path.replace('01_input', '02_output')
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
file_list = os.listdir(input_path)
pzt_index = 4
ecg_index = 5
starttime = ''
use_postprocess = False
clear_distortion = False


plt.ion()
for filename in file_list:
    pzt_list = []
    ecg_list = []
    distortion = []
    ind = 0
    get_rpm = False
    get_bpm = False
    print(filename)
    char_inter = '\t'
    if '.' not in filename:
        continue
    with open(os.path.join(input_path, filename), 'r', encoding='utf-8') as rpm_input:
        count = 0
        
        while count <= 3:
            line = rpm_input.readline()
            count += 1
            if 'Start_Time' in line:
                starttime = line
            elif 'device name' in line:
                line = line[1:]
                info = json.loads(line)

                for iten in info:
                    sub_info = info[iten]
                    starttime = 'Start_Time: ' + sub_info["date"]
                    starttime += '-'+ sub_info["time"].replace(':', '-')
        first_line = line.encode('utf-8').decode('utf-8-sig')
        if ',' in first_line:
            char_inter = ','
        
        data_list = first_line.split(char_inter)
        #rpm_list.append(float(rpm))
        #total_index = len(rpm)
        # try:
        #     rpm = data_list[pzt_index]       
        #     pzt_list.append(float(rpm))
        #     get_rpm = False
        # except (TypeError, ValueError):
        #     pass
        
        try:
            bpm = data_list[ecg_index]        
            ecg_list.append(float(bpm))
            get_bpm = True
        except (TypeError, ValueError):
            pass
        try:
            rpm = data_list[pzt_list]        
            pzt_list.append(float(rpm))
            get_rpm = True
        except (TypeError, ValueError):
            pass
            
            
        for line in rpm_input.readlines():
            if 'End' in line:
                break
            elif 'State' in line:
                continue
            elif 'Time' in line:
                continue
            
            ind += 1
            if get_rpm:
                rpm = line.split(char_inter)[pzt_index]
                pzt_list.append(float(rpm))
            if get_bpm:
                
                bpm = line.split(char_inter)[ecg_index]
                ecg = (float(bpm))
                ecg_list.append(ecg)
                if ecg < 1 :
                    distortion.append(ind)
                
    X = np.arange(len(pzt_list)) / sample_rate    

    if get_rpm : 
        if use_new_rpm :

            da=pzt_list
            bio=biosppy.signals.resp.resp(signal=da, sampling_rate=sample_rate, show=True)
            rpm_freq=bio['resp_rate']
            rpm=rpm_freq*60
            rpm_ts=np.floor(bio['resp_rate_ts'])            
            end_t = int(len(da)/100)

            f = interp1d(rpm_ts, rpm, kind = 'linear')        
            x_new = np.linspace(rpm_ts[1], rpm_ts[-1], end_t )
            rpm_new = f(x_new)
            rpm_final = rpm_new



            # da=pzt_list
            # peaks,_=find_peaks(-da,distance=250,height=np.mean(-da)+np.std(-da))
            # plt.plot(da)
            # plt.plot(peaks,da[peaks],"o")
            # plt.show()
            # di=peaks[1:]-peaks[:-1]
            # 60/(di/100)

        else :

            breath = np.zeros(len(pzt_list))
            baseline = np.zeros(len(pzt_list))
            window = 50
            window2 = 100 * 2
            
            
            for i in range(len(pzt_list)):
                if i < window:
                    pwind = pzt_list[0: i + window]
                elif i > len(pzt_list) - window:
                    pwind = pzt_list[i - window : ]
                else:
                    pwind = pzt_list[i - window : i + window]
                    
                if i < window2:
                    base = pzt_list[0: i + window2]
                elif i > len(pzt_list) - window2:
                    base = pzt_list[i - window2 : ]
                else:
                    base = pzt_list[i - window2 : i + window2]
                    
                breath[i] = np.mean(pwind) - np.mean(base)
                
                #beat_heart[i] = ecg_list[i]
            breath = breath - np.mean(breath)    
            peaks_breath,_ = find_peaks(breath, height = -10, prominence = 20)
            valey_breath,_ = find_peaks(breath * -1, height = -10, prominence = 20)
            i_peak = 0
            i_valy = 0
            
            G = 1 if peaks_breath[0] > valey_breath[0] else 0
            
            while i_peak < len(peaks_breath)-1 and i_valy < len(valey_breath)-1:
                if G == 1 :                
                    if peaks_breath[i_peak + 1] < valey_breath[i_valy] :
                        if breath[peaks_breath[i_peak + 1]] > breath[peaks_breath[i_peak]]:
                            peaks_breath[i_peak] = peaks_breath[i_peak + 1]
                        peaks_breath = np.delete(peaks_breath, i_peak + 1)
                    else:
                        G = 0
                        i_peak = i_peak + 1  
                else:
                    if valey_breath[i_valy + 1]  < peaks_breath[i_peak]:
                        if breath[valey_breath[i_valy + 1]] < breath[valey_breath[i_valy]]:
                            valey_breath[i_valy] = valey_breath[i_valy + 1]
                        valey_breath = np.delete(valey_breath, i_valy+1)
                    else:
                        G = 1
                        i_valy = i_valy + 1
            rpm = (peaks_breath[1:] - peaks_breath[0:-1])
            rpm = 60 / (rpm/sample_rate)
            max_ = max(rpm)
            while max_ > 60:
                i = np.argmax(rpm)
                if peaks_breath[i] >  peaks_breath[i+1]:
                    ind = i+1
                else:
                    ind = i
                peaks_breath = np.delete(peaks_breath, ind)
                rpm = (peaks_breath[1:] - peaks_breath[0:-1])
                rpm = 60 / (rpm/sample_rate)
                max_ = max(rpm)
            rpm_t = (peaks_breath[1:] + peaks_breath[0:-1])/(2 * sample_rate)
            start = int(np.ceil(rpm_t[0]))
            end = int(np.floor(rpm_t[-1]))
            for i in range(1, len(rpm)):
                rpm[i] = (rpm[i] + rpm[i-1])/2
            f = interp1d(rpm_t, rpm, kind = 'nearest')
            x_new = np.linspace(start, end, end - start+1)
            rpm_new = f(x_new)
            
            rpm_final = np.zeros(int(np.floor(len(pzt_list)/sample_rate)))
            rpm_final[start: end+1] = rpm_new
            rpm_final[:start] = rpm_new[0]
            rpm_final[end:] = rpm_new[-1]
            





        with open(os.path.join(output_path, 'rpm', filename ), 'w') as rpm_output:
            for rpm_o in rpm_final:
                rpm_output.write(str(int(rpm_o)) + '\n')
        
        plt.figure()
        plt.title(filename)
        plt.ylabel('PSD')
        plt.xlabel('time(seconds)')
        plt.plot(x_new, rpm_new, color='blue')
        #plt.plot(np.linspace(0,len(breath)-1, len(breath)), breath, color = 'blue')
        #plt.plot(peaks_breath,breath[peaks_breath], color='red', marker='D', linestyle = 'None')

        
    if get_bpm:
        result = biosppy.signals.ecg.ecg(ecg_list,  sampling_rate=sample_rate, show = True)     
        filtered_ECG = result[1]
        rpeaks = result[2]
        
        if use_postprocess:
            rpeak2 = biosppy.signals.ecg.christov_segmenter(signal=filtered_ECG, sampling_rate=sample_rate)
            result = biosppy.signals.tools.get_heart_rate(beats=rpeak2[0], sampling_rate=sample_rate, smooth=False, size=2)

            
            HR_t_org = result[0] / sample_rate
            HR_org = result[1]

            HR_pair = np.array(list(zip(HR_t_org, HR_org)))
            
            # window = 10
            # for i in range(1,len(HR_org)-1):
            #     if abs(HR_org[i] - HR_org[i-1]) + abs(HR_org[i] - HR_org[i+1]) > 70:
            #         HR_pair[i, 1] = (HR_org[i-1] + HR_org[i+1]) / 2
            #     elif abs(HR_org[i] - np.mean(HR_org[i-window:i])) > 20 and i >= window:
            #         if HR_org[i] < np.mean(HR_org[i-window:i]) and HR_org[i-1] < HR_org[i-2]:
            #             HR_pair[i, 1] = HR_org[i-1] - (HR_org[i-2] - HR_org[i-2])
            #         elif HR_org[i] > np.mean(HR_org[i-window:i]) and HR_org[i-1] > HR_org[i-2]:
            #             HR_pair[i, 1] = HR_org[i-1] - (HR_org[i-2] - HR_org[i-2])
            #         HR_pair[i, 1] = np.mean(HR_pair[i-window:i], axis=1)
        else:
            HR_t_org = result[5]
            HR_org = result[6]

            #HR = [[HR_t_org[i], HR_org[i]] for i in range(len(HR_t_org)) ]
            HR_pair = np.array(list(zip(HR_t_org, HR_org)))
            # HR_t = np.copy(HR_t_org)
            # HR = np.copy(HR_org)

        
        plt.figure()  
        plt.title(filename)
        plt.ylabel('HR')
        plt.xlabel('time(seconds)')
        #plt.plot(HR_t, HR, color='blue')
        plt.plot(HR_t_org, HR_org, color='grey')
        list_HR_power = []
        for i in range(len(HR_t_org)):
            ecg_array = ecg_list[int(max(0,(HR_t_org[i]-1) * sample_rate)) : int(min(len(HR_t_org)-1, (HR_t_org[i]+1)) * sample_rate)]
            ecg_array = ecg_array - np.median(ecg_array)
            HR_power = sum(abs(ecg_array)) * (100 / sample_rate)
            list_HR_power.append(HR_power)
            if len(distortion)> 0 and clear_distortion:
                if min(abs(distortion - HR_t_org[i] * sample_rate)) < sample_rate  or HR_power > 10000:
                    HR_pair[i, :] = -1 
                    HR_pair[i + 1, :] = -1   
        
        HR_t = HR_pair[:, 0]
        HR  = HR_pair[:, 1]        
        HR_t = HR_t[HR_t > 0]
        HR  = HR[HR > 0]
        rri_array = temp = (60/np.array(HR))  * 1000  
        start_t = int(np.ceil(HR_t[0]))
        end_t = int(np.floor(HR_t[-1]))
        f = interp1d(HR_t, HR, kind = 'linear')
        
        x_new = np.linspace(start_t, end_t, end_t - start_t + 1)
        
        bpm_new = f(x_new)
        
        
        bpm_final = np.zeros(int(np.ceil(HR_t[-1]))+1)    
        
        bpm_final[start_t: end_t + 1] = bpm_new
        bpm_final[:start_t] = bpm_new[0]
        bpm_final[end_t:] = bpm_new[-1]
        
        bpm_final = moving_average(bpm_final, 20)
        plt.plot(range(len(bpm_final)), bpm_final, color='red')
        
        plt.pause(0.0001)
        golden_filename = filename.replace('.txt', '.csv')
        with open(os.path.join(output_path, 'bpm', golden_filename ), 'w') as bpm_output:
            bpm_output.write(starttime)
            for bpm_o in bpm_final:
                bpm_output.write(str(int(bpm_o)) + '\n')

plt.ioff()
plt.show()
# plt.close()
        