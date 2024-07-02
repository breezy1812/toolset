from email import header
from enum import unique
import numpy as np 
import pandas as pd
import os
import datetime
import json
import shutil
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.faker import Faker
from pyecharts.charts import *
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
import calendar
from plotnine import *
import time
from function.compute_fatigue import show_statistics_info
from function.chart import *

def Get_total_algo_info(data, year, month):
    data = data.drop_duplicates(subset=['timestamp', 'golden_hr', 'datetime'])
    # remove []
    data_idx = data[data["golden_hr"]=="[]"].index
    data = data.drop(data_idx)
    data['golden_hr'] = data['golden_hr'].apply(eval)

    # 將 timestamp convert to the week of each month week
    begin_week = datetime.date(year, month, 1).strftime("%U")
    timeString = [i for i in data['datetime']] # 輸入原始字串
    struct_time = ([time.strptime(j, "%Y_%m_%d_%H_%M_%S") for j in timeString]) # 轉成時間元組

    new_weekString = [time.strftime("%U", w) for w in struct_time]
    weeks = [(int(l) - int(begin_week) +1) for l in new_weekString]
    data['week'] = ["week " + str(i) for i in weeks]

    new_dayString = [time.strftime("%d", d) for d in struct_time]
    data['day'] = [int(x) for x in new_dayString]

    new_hourString = [time.strftime("%H", h) for h in struct_time]
    data['hour'] = [int(y) for y in new_hourString]

    data['period'] = 'Morning'
    data['period'][(data['hour'] >= 10) & (data['hour'] < 14)] = 'Noon'
    data['period'][(data['hour'] >= 14) & (data['hour'] < 18)] = 'Afternoon'
    data['period'][(data['hour'] >= 0) & (data['hour'] < 6)] = 'Midnight'
    data['period'][(data['hour'] >= 18) & (data['hour'] < 24)] = 'Night'
    return data

def available_HR(goldenHR_list):  
    # 移除 0 or -1
    target1 = -1
    while(target1 in goldenHR_list):
        goldenHR_list.remove(target1)
    target2 = 0
    while(target2 in goldenHR_list):
        goldenHR_list.remove(target2)
    target3 = None
    while(target3 in goldenHR_list):
        goldenHR_list.remove(target3)
    return goldenHR_list   

def Make_calendar_df(data, days, record, year, month):
    def week_day(date):
        '''generate week days from YYYY-MM-DD format'''
        year, month, day = (int(x) for x in date.split('-'))   
        answer = datetime.date(year, month, day).weekday() 
        answer = int(answer) + 1
        return answer
    def recode_ordered(array,level):
        # recode string data to ordered factors
        cate = pd.api.types.CategoricalDtype(categories=level, ordered= True)
        array = array.astype(cate)
        return array    
    ### Time period ###
    begin = datetime.date(year, month, 1)                                            
    end = datetime.date(year, month, calendar.monthrange(year, month)[1])            
    Times = pd.Series(str(begin + datetime.timedelta(days=i)) for i in range((end - begin).days + 1))
    weekdays = Times.apply(week_day)

    df = pd.DataFrame({'date_time': Times, 'weekdays':weekdays,
                       'mday':days,'month': calendar.month_name[Month],
                       'val': record})
    timeString = [i for i in df['date_time']] # 輸入原始字串
    struct_time = ([time.strptime(j, "%Y-%m-%d") for j in timeString]) # 轉成時間元組
    new_weekString = [time.strftime("%U", k) for k in struct_time]
    begin_week = datetime.date(Year, Month, 1).strftime("%U")
    weeks = [(int(l) - int(begin_week) +1) for l in new_weekString]
    df['monthweek'] = weeks    # 紀錄每個月的第幾週
    num_weekdays = {1:'Mon',2:'Tue',3: 'Wed',4:'Thu',5 :'Fri',6:'Sat', 7 : 'Sun'} # map between number and weekdays
    df['weekdays'] = df['weekdays'].map(num_weekdays)
    wdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] # set factor order level 
    wdays.reverse() # 定義每週由日開始
    df['weekdays'] = recode_ordered(df['weekdays'], wdays) # order reverse 
    return df

def Make_weekly_frequency_df(data, cal_df):
    Weeks = ["week " + str(i) for i in set(cal_df['monthweek'])]
    result = []
    # 每週有多少數據 
    for w in Weeks:
        pt = []
        for col in data['golden_hr'][data['week'] == w]:
            pt.extend(col)
        pt = available_HR(pt)
        result.append(pt)
    for index, value in enumerate(result):
        if value == []:
            result[index] = [0,0,0] # 都沒有數據，補空值0
    df = pd.DataFrame({'Week': Weeks, 'golden_hr': result})
    return df

def Make_periods_df(data):
    Period = ['Morning', 'Noon', 'Afternoon', 'Night', 'Midnight']
    result = []
    # 每週有多少數據 
    for p in Period:
        pt = []
        for col in data['golden_hr'][data['period'] == p]:
            pt.extend(col)
        pt = available_HR(pt)
        result.append(pt)
    for index, value in enumerate(result):
        if value == []:
            result[index] = [0,0,0] # 都沒有數據，補空值0
    df = pd.DataFrame({'period': Period, 'golden_hr': result})
    return df


# choose one driver (E4695E4694FE, F1D881BB0FAB, F4A5B22A27A4, F8BE9DEE6C66, CD62BFF2FCEF)
Driver = "CD62BFF2FCEF"  
Year = 2022
Month = 5  

Input_path = os.path.join(os.getcwd() + "/01_input/clean data/" + str(Year) + "_" + calendar.month_name[Month] + "/rawdata/" + Driver + "/Algo/")
Output_path = os.path.join(os.getcwd() + "/02_output/" + str(Year) + "_" + calendar.month_name[Month] + "/" + str(Driver) + "/")
if os.path.exists(Output_path):
    shutil.rmtree(Output_path)
os.makedirs(Output_path)

Data = pd.read_csv(Input_path + "algo_" + Driver + ".csv")
Data = Get_total_algo_info(Data, Year, Month)
show_statistics_info(Data)

Data.to_csv(Output_path  + str(Driver) + "_" + calendar.month_name[Month] + "_filtered.csv", index=False)

# 幾月有幾天
Days = [day for day in range(1, calendar.monthrange(Year, Month)[1]+1)]
# 記每天的數量
Record = [list(Data['day']).count(day) for day in Days]


### HR Histogram Plot ###
HRresult = available_HR([element for lis in Data['golden_hr'] for element in lis])    
plt = plot_HR_distribution(HRresult)
plt.savefig(Output_path + "Heart rate distribution.png") 


### calendar ###
Cal_df = Make_calendar_df(Data, Days, Record, Year, Month)
chartcalendar = calendar_ggplot(Cal_df)
chartcalendar.save(Output_path + 'Calendar_' + calendar.month_name[Month] + '.png', width = 10, height = 6, dpi = 300)


### daily bar plot ###
make_snapshot(snapshot, daily_barplot(Days, Record, Month).render(Output_path + "Daily_record_" + calendar.month_name[Month] + ".html"), 
              Output_path + "Daily_record_" + calendar.month_name[Month] + ".png")
                                                                                

### Weekly frequency boxplot ###
Weekly_df = Make_weekly_frequency_df(Data, Cal_df)
make_snapshot(snapshot, weekly_frequency_boxplot(Weekly_df, Month).render(Output_path + "Weekly_frequency.html"), 
              Output_path + "Weekly_frequency.png")


### HR at Periods ###
Period_df = Make_periods_df(Data)
make_snapshot(snapshot, periods_hr_boxplot(Period_df, Month).render(Output_path + "Periods_HR.html"), Output_path + "Periods_HR.png")