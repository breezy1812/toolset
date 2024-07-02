"""
.. module:: read_data
    :synopsis: A simple module for reading BCG data

.. moduleauthor:: Mads

"""

import numpy as np
from os import listdir
import os
from numpy import genfromtxt
import re
def read_pressure_acc(path_data, filename):
    """Objective: get pressure data of the file.
        @param path_data: the path of data.
        @param filename: name of the input file.
        @param down_scale:down scale the sample rate of the file.
        @return a list of pressure
    """
    acc_x = []
    acc_y = []
    acc_z = []
    data = []
    start_time = []
    with open(os.path.join(path_data, filename), 'r', encoding='utf-8') as pressure_input:
        time = pressure_input.readline()
        if "Start Time" in time:
            time = re.split(":|/|-", time)
            start_time.extend(list(map(int, time[1:])))
            start_time[5] += start_time[6] / 1000000
            start_time.pop()
        else:
            log_data = time.split(',')
            if len(log_data) > 5:
                data.append(int(log_data[1].strip(' []')))
                acc_x.append(int(log_data[2].strip(' []')))
                acc_y.append(int(log_data[3].strip(' []')))
                acc_z.append(int(log_data[4].strip(' []')))

        for line in pressure_input.readlines():
            if "End Time" in line:
                break
            log_data = line.split(',')
            if len(log_data) > 5:
                data.append(int(log_data[1].strip(' []')))
                acc_x.append(int(log_data[2].strip(' []')))
                acc_y.append(int(log_data[3].strip(' []')))
                acc_z.append(int(log_data[4].strip(' []')))

    return np.array(data), np.array(acc_x), np.array(acc_y), np.array(acc_z), start_time


def read_pressure_acc_text(text):
    """This function read BCG and ACC data from database.
    
    Args:
        text (str):  the data text from database.

    Returns:
        tuple (param1, param2, param3, param4, param5) 
        WHERE
        - param1 - (int array): BCG.
        - param2 - (int array): ACC X.
        - param3 - (int array): ACC Y.
        - param4 - (int array): ACC Z.
        - param5 - (int array): start time.

    Raises:
       AttributeError, KeyError

    
    """
    acc_x = []
    acc_y = []
    acc_z = []
    data = []
    start_time = []
    text = text.split('\n')[:-1]
    time = text[0]
    if "Start Time" in time:
        time = re.split(":|/|-", time)
        start_time.extend(list(map(int, time[1:])))
        start_time[5] += start_time[6] / 1000000
        start_time.pop()
    else:
        log_data = time.split(',')
        data.append(int(log_data[1].strip(' []')))
        acc_x.append(int(log_data[2].strip(' []')))
        acc_y.append(int(log_data[3].strip(' []')))
        acc_z.append(int(log_data[4].strip(' []')))
        
    text = text[1:]
    for line in text:
        if "End Time" in line:
            break
        log_data = line.split(',')
        if len(log_data) < 4:
            continue
        data.append(int(log_data[1].strip(' []')))
        acc_x.append(int(log_data[2].strip(' []')))
        acc_y.append(int(log_data[3].strip(' []')))
        acc_z.append(int(log_data[4].strip(' []')))

    return np.array(data), np.array(acc_x), np.array(acc_y), np.array(acc_z), start_time




