import pandas as pd
import shutil
import os
from datetime import datetime
import numpy as np

def check_data(input_data):
    # 刪除重複的時間
    df = input_data.drop_duplicates(subset=['timestamp', 'goldenhr'])
    # 刪除空值的時間
    data_idx = df[df["goldenhr"]=="[]"].index
    df = df.drop(data_idx)
    for i in df.columns:
        if (type(df[i][0]) == str) & (i != 'hwversion'):
            df[i] = df[i].map(eval) 
    # 全部非 0 or -1 
    # df = df[df["goldenhr"].map(sum)>0]
    # 依時間排序
    df = df.sort_values(by=['timestamp'])
    return df

def timestampToDatetime(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y_%m_%d_%H_%M_%S')

def interpolation(Time, Output, NextTimest, NextOutputst):
    if len(Output) > len(Time):
        Out = Output[:len(Time)]
    else:
        x = Time[:len(Output)] + [NextTimest]
        y = Output + [NextOutputst]
        answer = np.interp(Time, x, y)
        Out = list(answer.astype(int))  
    return Out

def grouping(Data):
    time_diff=0
    grouping_time=75  # gap 75 sec
    grouping_list=[]
    algo_list=[]
    all_algo = Data.to_dict('records')
    i=0
    while(i<len(all_algo)-1):
        algo_list.append(all_algo[i])
        # algo_list.append(i)
        time_diff=all_algo[i+1]["timestamp"]-all_algo[i]["timestamp"]
        if abs(time_diff) > grouping_time:
            grouping_list.append(algo_list)
            algo_list=[]
        i=i+1
    algo_list.append(all_algo[i])
    grouping_list.append(algo_list)
    return grouping_list
 
    
""" workflow """
cleandata_version = "0.1"
dataname = "Bernard_0805_roadtest"
data = pd.read_csv(os.getcwd() + "/01_input/AWS/" + dataname + ".csv")
data = check_data(data)
grouping_list = grouping(Data=data)

""" build output folder """
output_dir = os.path.join(os.getcwd() + "/02_output/" + dataname)   
if os.path.exists(output_dir):
    print("Already exists and overwrites")
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


""" ground_truth_bpm """
golden_hr=[]
for j in range(len(grouping_list)):
    write_in = open(output_dir + "/" + dataname + "_" + timestampToDatetime(grouping_list[j][0]['timestamp']) + ".csv", "w")
    write_in.write("cleandata_version: " + cleandata_version + "\n")
    write_in.close()
    for i in range(len(grouping_list[j])):
        golden_hr.extend(grouping_list[j][i]['goldenhr'][:61])  # 保留每筆 61 sec
    GoldenHR={timestampToDatetime(grouping_list[j][0]['timestamp']): golden_hr}
    GoldenHR=pd.DataFrame(GoldenHR)
    GoldenHR.to_csv(output_dir + "/" + dataname + "_" + timestampToDatetime(grouping_list[j][0]['timestamp']) + ".csv", 
                    index=False, mode="a")
    golden_hr=[]

""" raw """
for m in range(len(grouping_list)):
    times=[]
    bcg=[]
    accx=[]
    accy=[]
    accz=[]
    hr=[]
    resp=[]
    status=[]
    fatigue=[]
    alarm=[]
    for i in range(len(grouping_list[m])-1):    # except last minute
        time = []
        timest = grouping_list[m][i]['mcutime']
        time.append(timest)  
        while True:
            timest = timest + 1000/64
            if (timest > grouping_list[m][i+1]['mcutime']):
                break
            time.append(int(timest))
        times.extend(time) 
        bcg.extend(interpolation(time, grouping_list[m][i]['bcg'], grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['bcg'][0]))
        accx.extend(interpolation(time, grouping_list[m][i]['accx'], grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['accx'][0]))
        accy.extend(interpolation(time, grouping_list[m][i]['accy'], grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['accy'][0]))
        accz.extend(interpolation(time, grouping_list[m][i]['accz'], grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['accz'][0]))
        hr.extend(interpolation(time, list(np.array(grouping_list[m][i]['hr']).repeat(64)), grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['hr'][0]))
        resp.extend(interpolation(time, list(np.array(grouping_list[m][i]['resp']).repeat(64)), grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['resp'][0]))
        status.extend(interpolation(time, list(np.array(grouping_list[m][i]['status']).repeat(64)), grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['status'][0]))  
        fatigue.extend(interpolation(time, list(np.array(grouping_list[m][i]['confidencelevel']).repeat(64)), grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['confidencelevel'][0]))
        alarm.extend(interpolation(time, list(np.array(grouping_list[m][i]['alarm']).repeat(64)), grouping_list[m][i+1]['mcutime'], grouping_list[m][i+1]['alarm'][0]))
    
    # last minute
    finaltimelimit=3840
    finaltime = []
    finaltimest = grouping_list[m][-1]['mcutime']
    finaltime.append(finaltimest)
    for h in range(len(grouping_list[m][-1]['bcg'])-1):
        finaltimest = finaltimest + (1000/64)
        finaltime.append(int(finaltimest))
    times.extend(finaltime[:finaltimelimit])
    bcg.extend(grouping_list[m][-1]['bcg'][:finaltimelimit])
    accx.extend(grouping_list[m][-1]['accx'][:finaltimelimit])
    accy.extend(grouping_list[m][-1]['accy'][:finaltimelimit])
    accz.extend(grouping_list[m][-1]['accz'][:finaltimelimit])
    hr.extend(list(np.array(grouping_list[m][-1]['hr']).repeat(64))[:finaltimelimit])
    resp.extend(list(np.array(grouping_list[m][-1]['resp']).repeat(64))[:finaltimelimit])
    status.extend(list(np.array(grouping_list[m][-1]['status']).repeat(64))[:finaltimelimit])
    fatigue.extend(list(np.array(grouping_list[m][-1]['confidencelevel']).repeat(64))[:finaltimelimit])
    alarm.extend(list(np.array(grouping_list[m][-1]['alarm']).repeat(64))[:finaltimelimit])   

    BCG={"times":times, "bcg":bcg, "accx":accx, "accy":accy, "accz":accz, "hr":hr, "resp":resp, 
         "status":status, "fatigue":fatigue, "alarm":alarm}
    
    write_in = open(output_dir + "/" + dataname + "_" + timestampToDatetime(grouping_list[m][0]['timestamp']) + ".log", "w")
    write_in.write("fw_version: " + grouping_list[m][0]["hwversion"] + "\n")
    write_in.write("cleandata_version: " + cleandata_version + "\n")
    write_in.write("Start_Time: " + timestampToDatetime(grouping_list[m][0]['timestamp']) + "\n")
    write_in.close()
    
    pd.DataFrame(BCG).to_csv(output_dir + "/" + dataname + "_" + timestampToDatetime(grouping_list[m][0]['timestamp']) + ".log", 
                             sep=",", index=False, header=False, mode="a")