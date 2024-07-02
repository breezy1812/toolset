import requests
import time
import numpy as np
import random
from bson.json_util import loads
import json
url = "http://59.120.189.128:5000/data/biologueData"

def post_data_db(testData):
    
    x = requests.post(url, json = testData, verify=False)
    print(x.text)

def get_data_db(ID):
    send_url = url + '=' +ID
    x = requests.get(send_url, verify=False)
    data = json.loads(x.text)
    return data
    
def get_BCG_db(ID):
    send_url = url + '=' +ID
    x = requests.get(send_url, verify=False)
    data = json.loads(x.text)
    ar_bcg = []
    # ar_acc_x = np.array([])
    # ar_acc_y = np.array([])
    # ar_acc_z = np.array([])
    for d in data:
        bcg = d['bcg']
        # acc_x = str2array(d['acc_x'])
        # acc_y = str2array(d['acc_y'])
        # acc_z = str2array(d['acc_z'])
        #print(len(bcg))
        ar_bcg += (bcg)
        # np.concatenate([ar_acc_x, acc_x], axis=0)
        # np.concatenate([ar_acc_y, acc_y], axis=0)
        # np.concatenate([ar_acc_z, acc_z], axis=0)

    return ar_bcg #, ar_acc_x, ar_acc_y, ar_acc_z

def get_HR_db(ID):
    send_url = url + '=' +ID
    x = requests.get(send_url, verify=False)
    data = json.loads(x.text)
    ar_HR = []
    ar_status = []
    ar_HRstat = []
    ar_confid_level = []
    ar_time = []
    for d in data:
        HR = d['heart_rate']
        status = d['status']
        # confid_level = str2array(d['confid_level'])

        ar_HR += HR
        ar_status += status
        ar_HRstat.append(np.mean(HR))
        ar_time.append(d['timestamp'])
        # np.concatenate([ar_confid_level, confid_level], axis=0)
    return ar_HR, ar_status, ar_HRstat, ar_time