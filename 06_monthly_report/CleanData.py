import pandas as pd
import shutil
import os
from datetime import datetime
import numpy as np 
import calendar
from function.getrawdata import *
from function.compute_fatigue import grouping_data
from statistics import *
   
def get_raw_table(userid, url_base, headers, Start, End, Driver):
    length_algo, algo_list = get_algos(userid, url_base, headers, Start, End)
    print("length_algo:", length_algo)
    Algo = pd.DataFrame(algo_list)
    Algo['datetime'] = Algo['timestamp'].apply(timestampToDatetime)
    if(length_algo > 0):
        carid = algo_list[0]['carid']
        length_bcg, bcg_list = get_bcgs(userid, carid, url_base, headers, Start, End)
        print("length_bcg:", length_bcg)
        Bcg = pd.DataFrame(bcg_list)
    else:
        print("NO Bcg data.")
    return Algo , Bcg

def clean_data(input_data):
    df = input_data.drop_duplicates(subset=['timestamp'])
    if "golden_hr" in df.columns:
        data_idx = df[df["golden_hr"]=="[]"].index
        df = df.drop(data_idx)     # exclude nulls
        df = df[df["golden_hr"].map(len)==60]      # Keep 60 data in 1 minute
        df = df[df["golden_hr"].map(sum)>0]     # Exclude all 0 or -1
        df = df[df["golden_hr"].map(set).map(len)>1]     # Exclude all same
        df = df.sort_values(by=['timestamp'])     # sorting  
    elif "bcg" in df.columns:
        boundary = 3840/2
        b = []
        for i in df['bcg']:
            b.append(i.count(0))
        df['count0'] = b
        df = df[df['count0'] < boundary]
    return df

def three_criteria(mergeresult, groupinglist):
    result_hr = []
    for hr in mergeresult['golden_hr']:
        result_hr.extend(hr)
    #parameter
    Upper_limit = mean(result_hr) + 2*stdev(result_hr)
    Lower_limit = mean(result_hr) - 2*stdev(result_hr)
    Delta = 15
    IQR = np.quantile(result_hr, 0.75) - np.quantile(result_hr, 0.25)
    Lower_Outlier = np.quantile(result_hr, 0.25) - (IQR * 1.5)
    Upper_Outlier = np.quantile(result_hr, 0.75) + (IQR * 1.5)
    # three criteria
    new_grouping_list=[]
    for group in groupinglist:
        hr = []
        for mins in group:
            hr.extend(mins['golden_hr'])
        # condition 1  (Percentage over two standard deviations < 0.7)
        percentage = len([i for i in hr if (i < Upper_limit) & (i > Lower_limit)])/len(hr)   
        # condition 2  (Before and after differences in values < 15)
        D = len([hr[i] for i in range(1,len(hr)-1) if (abs(hr[i]-hr[i+1]) > Delta) & (abs(hr[i]-hr[i-1]) > Delta)]) 
        # condition 3  (Outlier == 0)
        for i in range(len(hr)):
            if (hr[i] < Lower_Outlier) | (hr[i] > Upper_Outlier):
                hr[i] = 0
        if (percentage > 0.7) & (D < 1):
            new_grouping_list.append(group)
    return new_grouping_list

def save_algo(path, algotable, whetherfiltered, driver):
    if whetherfiltered == False:
        raw_dir = os.path.join(path + "rawdata/" + driver + "/Algo/")
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
        os.makedirs(raw_dir)
        algotable.to_csv(raw_dir + "algo_" + driver + ".csv", index=False)
    else:
        algo_dir = os.path.join(path + "sort out/" + driver + "/Algo/")
        if os.path.exists(algo_dir):
            shutil.rmtree(algo_dir)
        os.makedirs(algo_dir)
        algotable.to_csv(algo_dir + "algo_" + driver + ".csv", index=False)
        
def save_cleaned_bcg(path, groupinglist, driver):
    bcg_dir = os.path.join(path + "/sort out/" + driver + "/BCG/")  
    if os.path.exists(bcg_dir):
        shutil.rmtree(bcg_dir)
    os.makedirs(bcg_dir)
    for m in range(len(groupinglist)):
        times=[]
        bcg=[]
        accx=[]
        accy=[]
        accz=[]
        hr=[]
        resp=[]
        status=[]
        for i in range(len(groupinglist[m])):
            time=[]
            timest=int(groupinglist[m][i]['timestampst'])
            time.append(int(timest))
            for h in range(3840-1):
                timest=int(timest+(1000/64))
                time.append(int(timest))
            times.extend(time)
            bcg.extend(groupinglist[m][i]['bcg'])
            accx.extend(groupinglist[m][i]['accx'])
            accy.extend(groupinglist[m][i]['accy'])
            accz.extend(groupinglist[m][i]['accz'])
            hr.extend(list(np.array(groupinglist[m][i]['hr']).repeat(64)))
            resp.extend(list(np.array(groupinglist[m][i]['resp']).repeat(64)))
            status.extend(list(np.array(groupinglist[m][i]['status']).repeat(64)))
        BCG={"times":times, "bcg":bcg, "accx":accx, "accy":accy, "accz":accz, "hr":hr, "resp":resp, "status":status}
        write_in = open(bcg_dir + "/" + timestampToDatetime(groupinglist[m][0]['timestamp']) + ".log", "w")
        write_in.write("fw_version: " + groupinglist[m][0]["hwversion"]+"\n")
        write_in.write("Start_Time: " + timestampToDatetime(groupinglist[m][0]['timestamp'])+"\n")
        write_in.close()
    
        pd.DataFrame(BCG).to_csv(bcg_dir + "/" + timestampToDatetime(groupinglist[m][0]['timestamp']) + ".log", 
                                 sep=",", index=False, header=False, mode="a")
    return True

def save_cleaned_hr(path, grouping_list, driver):
    hr_dir = os.path.join(path + "/sort out/" + driver + "/GoldenHR/")
    if os.path.exists(hr_dir):
        shutil.rmtree(hr_dir)
    os.makedirs(hr_dir)
    golden_hr=[]
    golden_hr_polar=[]  
    for j in range(len(grouping_list)):
        for i in range(len(grouping_list[j])):
            golden_hr.extend(grouping_list[j][i]['golden_hr'])
            if str(grouping_list[j][i]['golden_hr_polar'])=="None":  # Fill [] for empty polar
                golden_hr_polar.extend([[]]*60)
            else:
                golden_hr_polar.extend(grouping_list[j][i]['golden_hr_polar'])
        GoldenHR={"golden_hr": golden_hr, "golden_hr_polar": golden_hr_polar}
        GoldenHR=pd.DataFrame(GoldenHR)
        GoldenHR.to_csv(hr_dir + timestampToDatetime(grouping_list[j][0]['timestamp']) + ".csv", index=False)
        golden_hr=[]
        golden_hr_polar=[]
    return True

### choose one driver ###
# E4695E4694FE, F1D881BB0FAB, F4A5B22A27A4, F8BE9DEE6C66, CD62BFF2FCEF
Driver = "F4A5B22A27A4" 
Start = '2022-06-01 00:00:00'
End = '2022-07-01 00:00:00'
Year = 2022
Month = 6
Output_path = os.path.join(os.getcwd() + "/01_input/" + "clean data/" + str(Year) + "_" + calendar.month_name[Month] + "/")
                    
url_base = "https://www.biologue.taipei/v0/"
# url_base = "http://192.168.0.86:8888/v0/"
Sdatetime=timestamp_datetime(time.time())+" "+timestamp_datetime_hour(time.time()-60*30)
Edatetime=timestamp_datetime(time.time())+" "+timestamp_datetime_hour(time.time())
headers = manager_get_token(url_base)
userid = user_id(url_base, Driver + '@test.com')


### 01 Get data ###
Algo, Bcg = get_raw_table(userid, url_base, headers, Start, End, Driver)


### 02 Clean data ###
Algo_cleaned = clean_data(Algo)
print("Algo_cleaned:", len(Algo_cleaned))
Bcg_cleaned = clean_data(Bcg)
print("Bcg_cleaned:", len(Bcg_cleaned))

    
### 03 Grouping data ###
result = pd.merge(Algo_cleaned, Bcg_cleaned, on=["userid", "carid", "timestamp"])
result = result.sort_values(by=['timestamp'])
print("Final len:", len(result))

if len(result) == 0:
    print("No available data")
    save_algo(path=Output_path, algotable=Algo, whetherfiltered=False, driver=Driver)
    raw_dir = os.path.join(Output_path + "rawdata/" + Driver + "/Bcg/")
    os.makedirs(raw_dir)
    Bcg.to_csv(raw_dir + "bcg_" + Driver + ".csv", index=False)
else:
    grouping_list = grouping_data(result)
    ### 04 Filter grouping data based on three criteria ###
    New_grouping_list = three_criteria(mergeresult = result, groupinglist = grouping_list)
        
    ### 05 Save data ###
    save_algo(path=Output_path, algotable=Algo, whetherfiltered=False, driver=Driver)
    save_algo(path=Output_path, algotable=Algo_cleaned, whetherfiltered=True, driver=Driver) 
    save_cleaned_bcg(path=Output_path, groupinglist=New_grouping_list, driver=Driver)
    save_cleaned_hr(Output_path, New_grouping_list, driver=Driver)