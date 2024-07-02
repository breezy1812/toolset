from unittest import result
from src.alg_fatigue import Alg_fatigue
from bson.objectid import ObjectId
from datetime import datetime
import time # 引入time
import pandas as pd


""" 第一次整理 (刪除重複 & 保留完整資料 & 按時間排序) """
def check_algo(input_data):
    ### 刪除重複的時間 ###
    df = input_data.drop_duplicates(subset=['timestamp', 'golden_hr', 'datetime'])
    ### 保留完整資料1min60筆 ###
    df["golden_hr"] = df["golden_hr"].apply(eval)  # hr 
    df = df[df["golden_hr"].map(len)==60]
    ### 依時間排序 ###
    df = df.sort_values(by=['timestamp'])
    return df

""" 轉換時間 """
def timestampToDatetime(timestamp):
    return datetime.fromtimestamp(timestamp*0.001).strftime('%Y/%m/%d-%H:%M:%S')

""" Grouping """
def grouping_data(check_df):
    df_records = check_df.to_dict('records')
    all_algo = df_records
    time_diff=0
    grouping_time=300000
    grouping_list=[]
    algo_list=[]
    i=0
    while(i<len(all_algo)-1):
        algo_list.append(all_algo[i])
        # algo_list.append(i)
        time_diff=all_algo[i+1]["timestamp"]-all_algo[i]["timestamp"]
        if time_diff > grouping_time:
            grouping_list.append(algo_list)
            algo_list=[]
        i=i+1
    algo_list.append(all_algo[i])
    grouping_list.append(algo_list)
    return grouping_list

""" Compute fatigue """

def compute_fatigue_algo_group(group):
    # Parameter 
    scale_paras = 3
    parafile_path = r"src/parameters"
    alg_ft_test = Alg_fatigue(parafile_path)
    window_size_min = 5
    id_list = []
    hr_list = []
    confidence_list = []
    status_list = []
    ts_list = []
    
    for j in group:
        id_list.append(str(j['_id']))
        hr_list.extend(j['golden_hr'])  # hr
        ts_list.append(j['timestamp'])
        confidence_list.extend(j['confidence'])
        # status全設為 1
        status_list = [1]*len(hr_list) # status_list.extend(j['status'])
    # calculate fatigue part
    list_HR, list_SDNN, list_RMSSD, list_DF, list_time_gd = alg_ft_test.get_HRV_list(hr_list, status_list, window_size_min)
    list_person_hr_test = [list_HR, list_SDNN, list_RMSSD]
    for n in range(scale_paras):
        passornot, list_person_hr_test[n] = alg_ft_test.get_normalization_paras(list_person_hr_test[n])
        if not passornot:
            break

    testing_X = []
    for j in range(len(list_person_hr_test[0])):
        x = []
        for n in range(scale_paras):
            x.append(list_person_hr_test[n][j])
        testing_X.append(x)
    list_FI_view, list_alert_stage = alg_ft_test.testing_model_fatigue(testing_X, list_time_gd)
    # compute done

    saveData ={
        "userid": ObjectId(group[0]["userid"]),
        "algolistlen": len(id_list),
        "timestampst": ts_list[0],   # unit: ms
        "timestampend": ts_list[len(ts_list)-1],  # unit: ms
        "timestampelist": ts_list,
        "alarm": list_alert_stage,
        "alarm_meanHR": list_HR
    }
    return saveData

""" 時間紀錄 (day, Period, week) """
def add_timeinfo(result, Year, Month):
    timeString = [i for i in result['timestampst']] # 輸入原始字串
    struct_time = ([time.strptime(j, "%Y/%m/%d-%H:%M:%S") for j in timeString]) # 轉成時間元組
    new_dayString = [time.strftime("%d", k) for k in struct_time]
    result['day'] = new_dayString

    result['sthour'] = [int(i[11:13]) for i in result['timestampst']]
    result['Period'] = 'Morning'
    result['Period'][(result['sthour'] >= 10) & (result['sthour'] < 14)] = 'Noon'
    result['Period'][(result['sthour'] >= 14) & (result['sthour'] < 18)] = 'Afternoon'
    result['Period'][(result['sthour'] >= 0) & (result['sthour'] < 6)] = 'Midnight'
    result['Period'][(result['sthour'] >= 18) & (result['sthour'] < 24)] = 'Night'

    # 將 timestamp 轉成每個月的第幾週 week
    import datetime
    begin_week = datetime.date(Year, Month, 1).strftime("%U")
    new_weekString = [time.strftime("%U", k) for k in struct_time]
    weeks = [(int(l) - int(begin_week) +1) for l in new_weekString]
    result['week'] = ["week " + str(i) for i in weeks]
    from datetime import datetime
    return result

""" show the statistics info """
def show_statistics_info(data_golden_hr):
    total_hr = []
    for hr in data_golden_hr['golden_hr']:
        total_hr.extend(hr)
    # remove -1
    total_hr = [hr for hr in total_hr if hr != -1]
    # remove 0
    total_hr = [hr for hr in total_hr if hr != 0]
    
    print("\n" + "Using time")
    print(len(data_golden_hr[['day','datetime']].groupby('day')))
    print(data_golden_hr[['day','datetime']].groupby('day').count())

    print("\n" + "Total Using Time")
    print(data_golden_hr[['week','datetime']].groupby('week').count().sum())
    print("\n"+"Final valid data: ")
    print(len(total_hr))
    d = pd.Series(total_hr)
    print("\n"+"Statistics Describe", "\n", d.describe().round(1))