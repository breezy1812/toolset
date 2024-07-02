# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:40:46 2020

.. module:: get_PERCLOS
    :synopsis: A sub module for computing PERCLOS with EAR
    
.. moduleauthor:: Mads
"""
import numpy as np


def get_threshold(ear_raw):
    """This function get the threshold of eye closing of EAR array. we choose the range of 80% is open and the lowest 5% is close. 
    
    Args:
        ear_raw (int[]):  the EAR data array.
       
    Returns:
        param 1 (float): the threshold 

    Raises:
       AttributeError, KeyError
    
    """
    
    ear = np.copy(ear_raw)
    ear = ear[ear>0.1]
    ear = ear[ear<0.5]
    sort_EAR = np.sort(ear)
    EAR_open = np.median(sort_EAR[int(len(ear)*0.8):])
    EAR_clos = np.median(sort_EAR[:int(len(ear)*0.1)])
    # hist = [0]*60
    # for i in ear:
    #     hist[int(i *100)] += 1
        
    # list_hit.append(hist)
    # hist = np.asarray(hist) 
    #EAR_open = np.argmax(hist)/100
    # hist[hist < 10] = 0
    # y = ar(hist)
    # x = ar(range(60))
    # maxima = np.argmax(y)
    # n = len(x)                          #the number of data
    # mean = sum(x * y) / sum(y)                 #note this correction
    # sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))       #note this correction

    # def Gauss(x, a, x0, sigma):
    #     return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    # popt,pcov = curve_fit(Gauss,x,y,p0=[1,mean,sigma])
    # new_y = Gauss(x,*popt)
    # EAR_open = np.argmax(new_y) / 100
    
    
    # for i in range(50):
    #     if new_y[i] > 5:
    #         hist[i] = 0
    
    # y = ar(hist)
    # mean = sum(x * y) / sum(y)                 #note this correction
    # sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))       #note this correction
    # popt,pcov = curve_fit(Gauss,x,y,p0=[1,mean,sigma])
    # new_y = Gauss(x,*popt)       
    
    #EAR_close = np.argmax(new_y) / 100
    
    threshold = EAR_clos + (EAR_open - EAR_clos)*0.2
    return threshold

def compute_PERCLOS_overall(ear_raw,threshold):
    """This function get the average PERCLOS within overall EAR.
    
    Args:
        ear_raw (int[]):  the EAR data array.
        threshold (float): the threshold for eye closing and opening
       
    Returns:
        param 1 (float): PERCLOS

    Raises:
       AttributeError, KeyError
    
    """

    
    ear = np.copy(ear_raw)
    ear = ear[ear>0.1]
    ear = ear[ear<0.5]
    close_frame = 0
    term_close = [1 if i < threshold else 0 for i in ear]
    
    flag_close = 0
    count = 0
    for i in range(len(ear)):
        if term_close[i] == 1:
            if flag_close == 1:
                count += 1
            flag_close = 1
        else:
            flag_close = 0
            if count > 0:
                close_frame += count
            count = 0
        if i == len(ear) -1 and count > 1:
            close_frame += count
    if len(ear) == 0:
        return -1
    else:
        return close_frame / len(ear)

def compute_PERCLOS_localmax(ear_raw, threshold, FPS):
    """This function get the maximun of the PERCLOS per 1 minute within overall EAR.
    
    Args:
        ear_raw (int[]):  the EAR data array.
        threshold (float): the threshold for eye closing and opening.
        FPS (int): the FPS (frames per seconds) of the vedio 
       
    Returns:
        param 1 (float): PERCLOS

        if -1 :
            the missing detection for each minute is > 5 seconds per minute.

    Raises:
       AttributeError, KeyError
    
    """
    start = 0
    window = int(60 * FPS)
    perclos_list = []
    close_threshold_time = int(0.2 * FPS)
    while start + window < len(ear_raw):
        segment = ear_raw[start : start + window]
        if 1 in segment:
            null_index = np.where(segment == 1)
            start = start + null_index[0][-1] + 1
            continue
        term_null = [1 if i < 0.1 or i > 0.5 else 0 for i in segment]
        term_close = [1 if i < threshold and i > 0.1 else 0 for i in segment]
        A = 0
        close_count = []
        for i in range(len(term_close)):
            if term_close[i] == 1:
                A += 1
            elif A > 0:                
                close_count.append(A)
                A = 0
        close_sum = sum([ C if C >= close_threshold_time else 0 for C in close_count ])
                    
        
        if sum(term_null) < FPS * 5 and close_sum > 0:
            perclos_list.append(close_sum/window)
        start = start + window
    
    if len(perclos_list) > 0:
        return (np.median(perclos_list))
    else:
        return (-1)

def get_perclos_text(text, is_PVT, PERCLOS_threshold, FPS):
    text = text.split('\n')[:-1]
    ear_raw = []
    for line in text:
        line = "".join(filter(lambda ch: ch in '0123456789.', line))
        if line == '' :
            continue
        ear_raw.append(float(line))

    if PERCLOS_threshold == 0:        
        PERCLOS_threshold = get_threshold(ear_raw)

    ear = np.asarray(ear_raw)
    
    return PERCLOS_threshold ,str(compute_PERCLOS_overall(ear, PERCLOS_threshold)), str(compute_PERCLOS_localmax(ear, PERCLOS_threshold, FPS))

    