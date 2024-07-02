# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:09:31 2020
.. module:: get_PVT
    :synopsis: A simple module for processing PVT data

.. moduleauthor:: Mads, Jimmy
"""

import os
import numpy as np
import csv
import warnings
import pylab as plt
from . import PVT_analsis
# import PVT_analsis as pvtana

five_min = False
data_head = ['name', 'sleep_time', 'wake_time', 'sleep_2h', 'HR', 'SSS', 'KSS', 'RT', 'RT_fast', 'PVT_inverRT', 'lapse', 'lapse_percent', 'correct rate', 'SNR']
index_profile = 7

def SNR_compute(data):
    """This function get SNR(signal-Noise Ratio) from a data array.
    
    Args:
        data (float[]):  the data array.
        

    Returns:
        param 1 (float): SNR

    Raises:
       AttributeError, KeyError

    
    """
    data = np.asarray(data)
    C = 100
    R = 196
    S = 1 / (data - C)
    W = 1/ (S * R + 1 )
    Q = sum(W) - sum(W * S)
    SNR = (len(data) * sum(W * S)) / sum(W * (S * Q)**2)
    return SNR

def get_PVT_result_array(raw_isi, raw_lor, raw_rt, raw_cor, related_threshold = -1):
    """This function get PVT_result.
    
    Args:
        raw_isi (int[]):  the isi array.
        raw_lor (int[]):  the array of Left(0) or right(1).
        raw_rt  (int[]):  the response time array.
        raw_cor (int[]):  the correct or not array.
        related_threshold (int): the related threshold for person

    Returns:
        param 1 (float): rt_used_mean \n
        param 2 (float): rt_used_sort_mean \n
        param 3 (float): rt_used_inv_mean \n
        param 4 (float): rt_used_lapes_len \n
        param 5 (float): rt_used_lapes_result \n
        param 6 (float): rt_corr_used_mean \n
        param 7 (float): rt_used_SNR
        

    Raises:
       AttributeError, KeyError

    
    """
    rt_outlier = 4000
    isi_outlier = 10000
    if related_threshold == -1:
        lapse_def = 500
    else:
        lapse_def = related_threshold
    false_alarm = 100
    rt_used_loc = np.where( (raw_rt<=rt_outlier) & (raw_isi<isi_outlier) & (raw_rt > 150))
    isi_cum = np.cumsum(raw_isi[rt_used_loc])
    if five_min:
        rt_used_loc_time = np.where(isi_cum >= 5*1000*60)
    else:
        rt_used_loc_time = rt_used_loc

    rt_used = raw_rt[rt_used_loc_time]
    drt_used = rt_used[1:] - rt_used[:-1]
    rt_used_len = len(rt_used)
    rt_used_lapes = rt_used[np.where(rt_used>lapse_def)]
    rt_used_lapes_len = len(rt_used_lapes)
    rt_used_drt250 = len(drt_used[drt_used > 250])
    rt_used_sort = np.sort(rt_used)
    rt_used_inv = 1/rt_used[np.where( (rt_used>false_alarm) & (rt_used<rt_outlier))]

    rt_used_mean = np.mean(rt_used)
    rt_used_sort_mean = np.mean(rt_used_sort[:int(rt_used_len*0.1)])
    rt_used_inv_mean = np.mean(rt_used_inv)
    rt_used_lapes_result = 100*rt_used_lapes_len/rt_used_len
    rt_corr_used_loc = np.where((raw_lor!=-1) & (raw_rt<=rt_outlier) & (raw_isi>isi_outlier))
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            rt_corr_used_mean = np.mean(raw_cor[rt_corr_used_loc])
        except RuntimeWarning:
            rt_corr_used_mean = np.NaN

    rt_used_SNR = SNR_compute(rt_used)    
    return rt_used_mean, rt_used_sort_mean, rt_used_inv_mean, rt_used_lapes_len, rt_used_drt250, rt_corr_used_mean, rt_used_SNR

def RT_filter(ISI, RT, rt_outlier = 4000, isi_outlier = 10000):
    used_loc = np.where( (RT<=rt_outlier) & (RT>150))
    return ISI[used_loc], RT[used_loc]



def parse_PVT_file(sub_dir, filename, data_head, index_profile):
    """This function get data from file.
    
    Args:
        sub_dir      (str):  the directory of target file.
        filename     (str):  the name of target file.
        data_head  (str[]):  the head of data column.
        index_profile(int):  the number of profiles
        

    Returns:
        param 1 (int[]): isi raw data \n
        param 2 (int[]): Left or Right raw data \n
        param 3 (int[]): response time raw data \n
        param 4 (int[]): correct record raw data \n
        param 5 (int): sleep time \n
        param 6 (int): wake time \n
        param 7 (int): sleep time in 2 hour in minute \n
        param 8 (int): HR \n
        param 9 (int): SSS \n
        param10 (int): KSS       

    Raises:
       AttributeError, KeyError    
    """
    data = np.zeros(len(data_head))
    data_start = False
    with open(os.path.join(sub_dir, filename), 'r', encoding='utf-8') as lines:
        raw_isi = [] 
        raw_rt = [] 
        raw_lor = [] 
        raw_cor = [] 
        for line in lines.readlines():
            if '.' in line:
                line = line.replace('.', ',')
            
            line_split = line.split(',')
            if data_start == False:
                for i in range(1, index_profile):
                    if data_head[i] in line_split[0]:
                        if data[i] == 0:
                            data[i] = int(line_split[1])
                        else:
                            data[i] = (int(line_split[1])+data[i])/2
                        break
                if 'RT' in line:
                    data_start = True
            else:
                raw_isi.append(int(line_split[1]))
                raw_lor.append(int(line_split[2]))
                raw_rt.append(int(line_split[3]))
                raw_cor.append(int(line_split[4]))

            sleep_time = data[1]
            wake_time = data[2]
            sleep_2h = data[3]
            HR = data[4]
            SSS = data[5]
            KSS = data[6]
        return np.array(raw_isi), np.array(raw_lor), np.array(raw_rt), np.array(raw_cor), sleep_time, wake_time, sleep_2h, HR, SSS, KSS

def parse_multi_PVT_file(subject_list, data_dir, all_data):
    """This function parse raw data and get PVT information from multi PVT files.
    
    Args:
        subject_list (str[]):  the list of target subjects.
        data_dir       (str):  the target directory.
        all_data     (str[]):  the buffer of result for output.
        

    Returns:
        param 1 (str[]): the data_arr for saving \n
        param 2 (str[]): the output result

    Raises:
       AttributeError, KeyError

    
    """
    data_arr = np.array([])
    for subject in subject_list:
        sub_dir = os.path.join(data_dir, subject)
        data_list = os.listdir(sub_dir)
        
        for filename in data_list:
            data = np.zeros(len(data_head))
            raw_isi, raw_lor, raw_rt, raw_cor, sleep_time, wake_time, sleep_2h, HR, SSS, KSS = parse_PVT_file(sub_dir,filename, data_head, index_profile)  
            rt_used_mean, rt_used_sort_mean, rt_used_inv_mean, rt_used_lapes_len, rt_used_lapes_result, rt_corr_used_mean, rt_used_SNR = get_PVT_result_array(raw_isi, raw_lor, raw_rt, raw_cor)
            
            data[1] = sleep_time
            data[2] = wake_time
            data[3] = sleep_2h
            data[4] = HR
            data[5] = SSS
            data[6] = KSS
            data[index_profile]   = rt_used_mean
            data[index_profile+1] = rt_used_sort_mean
            data[index_profile+2] = rt_used_inv_mean
            data[index_profile+3] = rt_used_lapes_len
            data[index_profile+4] = rt_used_lapes_result
            data[index_profile+5] = rt_corr_used_mean
            data[index_profile+6] = rt_used_SNR 
            consolidate_line = []        
            for i in range(len(data_head)):
                if i == 0:
                    consolidate_line.append(filename)
                else:
                    consolidate_line.append(str(data[i]))
        
            all_data.append(consolidate_line)   
            
            data_arr = np.append(data_arr, data[1:], axis = 0)

        len_data = len(data[1:])
        data_arr = data_arr.reshape(int(len(data_arr)/len_data),len_data)

        return data_arr, all_data

def plot_job(data_arr):
    plt.plot(data_arr[:,1],data_arr[:,6],'.')
    plt.show()

def get_related_threshold(raw_rt):

    mean = np.average(raw_rt)
    std = np.std(raw_rt)

    max_interation = 10
    count_outlier = sum([1 if A > mean + std*3 else 0 for A in raw_rt])
    while count_outlier > 0 or max_interation > 0:
        
        ind = np.where(raw_rt <= mean + std*3)[0]
        new_raw_rt = np.array(raw_rt)[ind]
        mean = np.average(new_raw_rt)
        std = np.std(new_raw_rt)
        count_outlier = sum([1 if A > mean + std*3 else 0 for A in new_raw_rt])
        raw_rt = new_raw_rt
        max_interation -= 1
    #if int(mean + std*3) < 500:
    #    return 500
    
    return int(mean + std*3)
    
def get_PVT_text(text, filename, get_middle = 0, related_threshold = -1, optimal_threshold = False):

    """This function parse raw data from database and get PVT information.
    
    Args:
        text   (str[]):  the raw data from database.
        filename (str):  the test subject id or the name of PVT file.

    Returns:
        param 1 (str): the data_arr for saving \n


    Raises:
       AttributeError, KeyError

    
    """
    data_head = ['name', 'sleep_time', 'wake_time', 'sleep_2h', 'HR', 'SSS', 'KSS', 'RT', 'RT_fast', 'PVT_inverRT', 'lapse', 'lapse_percent']
    index_profile = 7
    text = text.split('\n')[:-1]
    data = np.zeros(len(data_head))
    raw_isi = [] 
    raw_rt = [] 
    raw_lor = [] 
    raw_cor = [] 
    time_stamp_each_S = [0]
    time_stamp = 0
    data_start = False
    for  line in text:       
       
        if '.' in line:
            line = line.replace('.', ',')
        
        line_split = line.split(',')
        if data_start == False:
            for i in range(1, index_profile):
                if data_head[i] in line_split[0]:
                    if data[i] == 0:
                        data[i] = int(line_split[1])
                    else:
                        data[i] = (int(line_split[1])+data[i])/2
                    break
            if 'RT' in line:
                data_start = True
        else:
            
            raw_isi.append(int(line_split[1]))
            raw_lor.append(int(line_split[2]))
            raw_rt.append(int(line_split[3]))
            raw_cor.append(int(line_split[4]))
            time_stamp += raw_isi[-1] + raw_rt[-1]
            time_stamp_each_S.append(time_stamp)
    
    if get_middle:
        total_time = time_stamp_each_S[-1]
        range_middle = [int(total_time * 0.25), int(total_time * 0.75)]
        ind_start = np.argmin(abs(np.array(time_stamp_each_S) - range_middle[0]))
        ind_end = np.argmin(abs(np.array(time_stamp_each_S) - range_middle[1]))
        raw_isi = raw_isi[ind_start : ind_end+1]
        raw_lor = raw_lor[ind_start : ind_end+1]
        raw_rt  = raw_rt[ind_start : ind_end+1]
        raw_cor = raw_cor[ind_start : ind_end+1]
    
    if related_threshold == 0:
        related_threshold = get_related_threshold(raw_rt)
    #threshold = np.copy(related_threshold)
    if optimal_threshold:
        with open('optimal_threshold.csv', 'r') as lines:
            for line in lines.readlines():
                if filename in line:
                   threshold = int(line.split(',')[1]) 
                   break


    

    
    rt_used_mean, rt_used_sort_mean, rt_used_inv_mean, rt_used_lapes_len, rt_used_lapes_result, _, _ = get_PVT_result_array(np.array(raw_isi), np.array(raw_lor), np.array(raw_rt), np.array(raw_cor), related_threshold = related_threshold)
    data[index_profile]   = rt_used_mean
    data[index_profile+1] = rt_used_sort_mean
    data[index_profile+2] = rt_used_inv_mean
    data[index_profile+3] = rt_used_lapes_len
    data[index_profile+4] = rt_used_lapes_result
    #data[index_profile+5] = rt_corr_used_mean
    #data[index_profile+6] = rt_used_SNR 
    consolidate_line = []        
    for i in range(len(data_head)):
        if i == 0:
            consolidate_line.append(filename)
        else:
            consolidate_line.append(str(data[i])) 
    if related_threshold == -1:
        return ','.join(map(str, consolidate_line))
    else:
        return ','.join(map(str, consolidate_line)), related_threshold


def get_KSS_text(text):
    
    text = text.split('\n')[:-1]
    KSS_1 = 0
    KSS_2 = 0
    for  line in text:       
       
        if '.' in line:
            line = line.replace('.', ',')
        
        line_split = line.split(',')

        if 'KSS' in line_split[0]:
            if KSS_1 == 0:
                KSS_1 = int(line_split[1])
            else:
                KSS_2 = int(line_split[1])
                break

    return KSS_1, KSS_2

def get_lapse_threshold(list_RT_all_subject):
    training_data = []
    for term in list_RT_all_subject:
        training_data += term

    bins = np.arange(100, 1000, 5)
    Z_result, threshold = PVT_analsis.get_threshold(np.array(training_data), _type='Wilcoxon', bins = bins[1:])
    bins = np.arange(100, 1100, 20)    
    counts, bins, bars = plt.hist(training_data, bins=bins)

    result = np.array([Z_result[1][i] / counts[int(np.floor(i/4))]**0.5 if counts[int(np.floor(i/4))] != 0 else 0 for i in range(len(Z_result[1]))])
    general_peak = bins[np.argmax(counts) + 2]
    result[Z_result[0] < general_peak] = 0
    threshold = Z_result[0][int(np.argmax(result))]
             
    return threshold
    
