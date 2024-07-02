# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:51:11 2020
.. module:: get_HRV
    :synopsis: A sub module for computing HRV by HR and ECG
    
.. moduleauthor:: Mads

"""

import biosppy
import numpy as np
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d


def moving_average(a, n=3):
    """This function get moving average from a data array.
    
    Args:
        a (float[]):  the data array.
        n (int)  :  the averaging window size.

    Returns:
        param 1 (float[]): the result array 

    Raises:
       AttributeError, KeyError

    
    """
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

def moving_med(a, n = 10):
    """This function get moving median from a data array.
    
    Args:
        a (float[]):  the data array.
        n (int)  :  the window size for geting median.

    Returns:
        param 1 (float[]): the result array 

    Raises:
       AttributeError, KeyError

    
    """
    moving_med = np.copy(a)
    n = int(n / 2)
    for i in range(len(a)):
        temp = a[max(0, i - n) : min(i + n, len(a)-1)]
        moving_med[i] = np.median(temp)
    return moving_med



def ECG_to_HR(ecg, fs, movavg = 1):
    """This function get HR(heart rate) from an ecg data. The HR would be interpolated into second by second.
        The moving average would be processed with input parameter on the HR array.

    
    Args:
        ecg (float[]):  the ecg data array.
        fs     (int):  the sampling rate of ecg data.
        movavg (int):  the window size of moving average for HR.


    Returns:
        param 1 (float[]): the HR array \n
        param 2 (float[]): the time array of HR

    Raises:
       AttributeError, KeyError

    
    """
    result = biosppy.signals.ecg.ecg(ecg,  sampling_rate=fs, show = False)    
    HR_t = result[5]
    HR = result[6]
    
    if len(HR) < (len(ecg) / fs) * 0.5:
        return [],[]

    window = 10
    for i in range(window,len(HR)):
        if abs(HR[i] - np.mean(HR[i-window:i])) > 20:
            HR[i] = np.mean(HR[i-window:i])

   
    return HR, HR_t



def get_HRV(bpm):
    """This function get HRV information from a HR array .
    
    Args:
        bpm (float[]):  the ecg data array.
        
    Returns:
        param 1 (float): the mean_HR \n
        param 2 (float): the RMSSD \n
        param 3 (float): the SDNN \n
        param 4 (float): the VLF \n

    Raises:
       AttributeError, KeyError

    
    """

    rri = (60/np.array(bpm))  * 1000  
    SDNN = np.std(rri)
    RMSSD = np.sqrt(np.mean((rri[1:] - rri[:-1])**2))
    time = np.arange(len(bpm))
    ls = LombScargle(time, bpm, normalization='psd')
    frequency, power = ls.autopower()
    yf2 = np.nan_to_num(power)
    xf2 = np.nan_to_num(frequency)
    HF_freq_band  = (xf2 >= 0.15) & (xf2 <= 0.4) 
    LF_freq_band  = (xf2 >= 0.04) & (xf2 <= 0.15) 
    VLF_freq_band = (xf2 >= 0.003) & (xf2 <= 0.04)     
    VLF = np.trapz(y=abs(yf2[VLF_freq_band]), x=xf2[VLF_freq_band])  
    LF = np.trapz(y=abs(yf2[LF_freq_band]), x=xf2[LF_freq_band]) 
    HF = np.trapz(y=abs(yf2[HF_freq_band]), x=xf2[HF_freq_band])   
    mean_HR = np.mean(bpm)
    
    return mean_HR, RMSSD, SDNN, LF/HF, VLF
def get_skweness(bpm):
    mean = np.mean(bpm)
    
    D = np.array(bpm) - mean
    S1 = sum(D**3) / len(bpm)
    S2 = sum(D**2) / len(bpm)
    S2 = (S2**0.5)**3
    return S1 / S2

def get_kurtosis(bpm):
    mean = np.mean(bpm)
    
    D = np.array(bpm) - mean
    k1 = sum((D)**4) / len(bpm)
    k2 = sum((D)**2) / len(bpm)
    k1 = k1**0.25
    k2 = k2**0.5
    return k1 / k2


def get_DF(bpm):
    """This function get power of frequency band of drowsin from a HR array .
    
    Args:
        bpm (float[]):  the ecg data array.
        
    Returns:
        param 1 (float): the DF \n       

    Raises:
       AttributeError, KeyError

    
    """
    N_low = int(0.443* 1 / 0.02)
    N_high = int(0.443* 1 / 0.08)+1
    mean = np.mean(bpm)
    wave_low =  moving_average(bpm, n = N_low) #0.02 Hz
    wave_high = moving_average(bpm, n = N_high)#0.08 Hz
    
    Power_time = ((wave_high - mean)**2 - (wave_low - mean)**2)
    return sum(Power_time)

def get_pNN50(bpm):
    rri = (60/np.array(bpm))  * 1000  
    rri_interval = abs(rri[1:] - rri[:-1])
    pNN = 0
    for i in rri_interval :
        if i > 50:
            pNN += 1
    return pNN / len(bpm)
    




def get_HRV_txt(ecg_list, is_PVT, fs_ecg, len_HR_min = 5):
    """This function get power of frequency band of drowsin from a HR array .
    
    Args:
        ecg_list (float[]):  the ecg data array.
        is_PVT       (int):  the flag of whether the PVT data is or not.
        fs_ecg       (int):  the sampling rate of the ECG.
        len_HR_min   (int):  the length of HR for computed. 
        
    Returns:
        param 1 (str): the HRV result for saving \n       

    Raises:
       AttributeError, KeyError
    
    """       
    
    HR, _ = ECG_to_HR(ecg_list, fs_ecg, 5)
    if is_PVT:
        start = 0
    else:
        start = len(HR) - int(60 * (len_HR_min))
    if start < 0 :
        start = 0
    bpm = HR[start: start + len_HR_min*60]
    
    if len(bpm) < len_HR_min*60:
        bpm = HR
    mean_HR, RMSSD, SDNN, VLF = get_HRV(bpm)
    HRV_record = str(get_DF(bpm)) + ',' + str(mean_HR) + ',' + str(RMSSD) + ',' + str(SDNN) + ',' + str(VLF)  
    
    return HRV_record
