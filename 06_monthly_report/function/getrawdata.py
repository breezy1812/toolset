from datetime import datetime
import time
import json
import base64
from math import *
from tkinter.tix import COLUMN
import requests
import pandas as pd
import os

def timestamp_datetime(value):
    #format = '%H:%M:%S'
    format = '%Y-%m-%d'
    #format = '%Y-%m-%d %H:%M:%S'
    # value為傳入的值為時間戳(整形)，如：1332888820
    value = time.localtime(value)
    # 經過localtime轉換後變成
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最後再經過strftime函式轉換為正常日期格式。
    dt = time.strftime(format, value)
    return dt

def timestamp_datetime_hour(value):
    format = '%H:%M:%S'
    #format = '%Y-%m-%d %H:%M:%S'
    # value為傳入的值為時間戳(整形)，如：1332888820
    value = time.localtime(value)
    # 經過localtime轉換後變成
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=0)
    # 最後再經過strftime函式轉換為正常日期格式。
    dt = time.strftime(format, value)
    return dt

def manager_get_token(url_base):
    test_login_data = {
        "email":"biologue@test.com",
        "password":"036688436"
    }
    url = f"{url_base}adminslogin"
    y = requests.post(url,json=test_login_data)
    token=json.loads(y.text).get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    return headers

def user_id(url_base, user_email):
    url=f"{url_base}users/email={user_email}"
    y=requests.get(url,headers=manager_get_token(url_base))
    y=json.loads(y.text).get("data")
    userid=y[0].get("_id")
    return userid

# 將timestamp轉成datetime
def timestampToDatetime(timestamp):
    return datetime.fromtimestamp(timestamp*0.001).strftime('%Y_%m_%d_%H_%M_%S')

#####algos
def get_algos(userid, url_base, headers, Start,End):
    TestData={
        "userid":f"{userid}",
        #"carid":"609b30bc4b421d9c70b7bf62",
        "Sdatetime":	f"{Start}",
        "Edatetime":	f"{End}"
    }
    url = f"{url_base}algos"
    print(url)
    y=requests.get(url,headers=headers,json=TestData).text
    y=json.loads(y).get("data")
    if(y is not None):
        return len(y),y
    return 0,[],''

#####bcgs
def get_bcgs(userid, carid, url_base, headers, Start, End):
    TestData={
        "userid":f"{userid}",
        "carid":f"{carid}",
        "Sdatetime":	f"{Start}",
        "Edatetime":	f"{End}"
    }
    url = f"{url_base}bcgs"
    print(url)
    x=requests.get(url,headers=headers,json=TestData).text
    x=json.loads(x).get("data")
    if(x is not None):
        return len(x),x
    return 0,[],''
    # algoidlist=[]
    # if(x is not None):
    #     for i in x :
    #         algoidlist.append(i['algoidlist'][0])
    #     return len(x),algoidlist
    # return 0,[]
    
def compare_list(bcglist,algolist):
    errorlist=[]
    count=0
    for i in algo_list:
        if(count==len(bcg_list)):
            errorlist.append(i['timestamp'])
        elif(i['_id']!=bcglist[count]):
            errorlist.append(i['timestamp'])
        else:
            count=count+1
    return errorlist

def convert_timestamp(timestamp_list):
    for i in timestamp_list:
        print(datetime.fromtimestamp(i/1000))