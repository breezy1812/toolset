from itertools import groupby
import pandas as pd
import numpy as np
from src.alg_fatigue import Alg_fatigue
from bson.objectid import ObjectId
from datetime import datetime
import os
import calendar
import time # 引入time
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
from pyecharts.charts import *
from pyecharts import options as opts
from function.compute_fatigue import *
from function.chart import *

def Prepare_fatigue_table(data, year, month):
    alarm = 1
    mingrouplen = 15
    long_travel = 300
    noalarm = 0
    
    df = check_algo(data) # Cleaning
    grouping_list = grouping_data(df) # Grouping & Compute fatigue
    result=[]
    for group in grouping_list:
        if len(group) < mingrouplen:  # tmie > 15 min
            continue
        else:
            saveData = compute_fatigue_algo_group(group)
            result.append(saveData)
    result = pd.DataFrame(result)
    result['timestampst'] = result['timestampst'].apply(timestampToDatetime)
    result = add_timeinfo(result, year, month)
    result['alarm_count'] = noalarm
    for i in range(len(result['alarm'])):
        result['alarm_count'][i] = result['alarm'][i].count(alarm)
    result['fatigue_total'] = list(map(len, result["alarm"]))
    
    nofatigue_idx = result[(result["algolistlen"] >= long_travel) & (result["alarm_count"] == noalarm)].index
    result = result.drop(nofatigue_idx)
    return result

def show_fatigues_info(result):
    print(result)
    print("\n" + "偵測疲勞的總時間")
    print(sum(result['alarm_count']*5))
    print("\n" + "平均每天發生疲勞的時間")
    print(np.round((sum(result['alarm_count']*5))/(len(set(result['day']))),decimals=2))
    
def unequal_add(a,b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return(c)
    
def Probability_alarm_record(result, min_):
    # 分母
    recordtime=[]                  
    for length in range(max(result['fatigue_total'])):
        accu=[]
        for num in result['fatigue_total']:
            if num > length:
                accu.append(num)
                total=len(accu)
        recordtime.append(total)
    # 分子
    recordalarm = np.array([0])
    for i in range(len(result['alarm'])):
        recordalarm = list(unequal_add(recordalarm, np.array(result['alarm'][i]))) 
    probability = []
    for i in range(len(recordalarm)):
        p = (recordalarm[i]/recordtime[i])*100
        probability.append(int(p)) # '%.2f'%
        
    if min_ == 10:
        # 分母
        recordtime_10min=[]
        for i in range(int(len(recordtime)/2)):
            recordtime_10min.append((recordtime[0+2*i]+recordtime[1+2*i]))    
        # 分子
        recordalarm_10min=[]
        for i in range(int(len(recordalarm)/2)):
            recordalarm_10min.append((recordalarm[0+2*i]+recordalarm[1+2*i]))
        probability = []
        for i in range(len(recordalarm_10min)):
            p = (recordalarm_10min[i]/recordtime_10min[i])*100
            probability.append(int(p))
    return probability

def weeks_in_month(year, month):
    rangemonth = calendar.monthrange(year, month)
    begin_week = datetime.date(year, month, 1).strftime("%U")
    end_week = datetime.date(year, month, rangemonth[1]).strftime("%U")
    weeks = ["week " + str(i+1) for i in range(int(end_week)-int(begin_week)+1)]
    return weeks

def travel_per_weeks(result, xlist):
    travel = result.groupby(['week'])['algolistlen'].agg('sum') # per weeks
    ylist1 = []
    for i in xlist:
        if i in travel:
            ylist1.append(int(travel[i]))
        else:
            ylist1.append(0)
    return ylist1

def alarm_percentage_per_weeks(result, xlist):
    weekline = result.groupby(['week'])['alarm_count'].agg('sum')
    weekalarm=[]
    for i in xlist:
        if i in weekline:
            weekalarm.append(weekline[i])
        else:
            weekalarm.append(0)
    totalline = result.groupby(['week'])['fatigue_total'].agg('sum')
    weektotal=[]
    for i in xlist:
        if i in totalline:
            weektotal.append(totalline[i])
        else:
            weektotal.append(1)
    # 發生alarm的百分比
    ylist2 = []
    for i in range(len(weekalarm)):
        a = weekalarm[i]/weektotal[i]*100
        ylist2.append(round(a))
    return ylist2

def calculate_meanHR_delta(result, earlyboundary):
    def calculate_Deltamean(table_list):
        if len(table_list) == 1:
            ans = int(table_list)
        else:
            ans = mean(table_list)
        return ans
    
    for i in range(len(result['alarm_meanHR'])):
        result['alarm_meanHR'][i] = [round(j) for j in result['alarm_meanHR'][i]]
        
    Lucid=[]
    EarlyFatigue=[]
    Fatigue=[]
    for n in range(len(result['alarm'])):
        Lucidlist=[]
        Fatiguelist=[]
        for i in range(len(result['alarm'][n])):
            if result['alarm'][n][i] == 0:
                Lucidlist.extend(result['alarm_meanHR'][n][i:i+1])
                # Lucidlist = [value for value in Lucidlist if value != -1]
            else:
                Fatiguelist.extend(result['alarm_meanHR'][n][i:i+1])   
        Lucid.append(round(mean(Lucidlist)))
        if sum(Fatiguelist[:earlyboundary]) < 0:
            EarlyFatigue.append(0)
            Fatigue.append(0)
        elif len(Fatiguelist) == 1:   
            EarlyFatigue.append((Fatiguelist[1]))
            Fatigue.append(0)
        elif len(Fatiguelist) <= earlyboundary:
            EarlyFatigue.append(round(mean(Fatiguelist[:earlyboundary])))
            Fatigue.append(0)
        elif len(Fatiguelist) == 4:
            EarlyFatigue.append(round(mean(Fatiguelist[:earlyboundary])))
            Fatigue.append(Fatiguelist[4])
        else:
            EarlyFatigue.append(round(mean(Fatiguelist[:earlyboundary])))
            Fatigue.append(round(mean(Fatiguelist[earlyboundary:])))

    result['Lucid'] = Lucid
    result['EarlyFatigue'] = EarlyFatigue
    result['Fatigue'] = Fatigue
    # 留EarlyFatigue & Fatigue都有的數據
    result = result[(result['EarlyFatigue']>0) | (result['Fatigue']>0)]
    result.index = list(range(len(result['alarm']))) # 改成連續index數值

    result['Delta_EarlyFatigue'] = [result['EarlyFatigue'][i]-result['Lucid'][i] for i in range(len(result))]
    result['Delta_Fatigue'] = [result['Fatigue'][i]-result['Lucid'][i] for i in range(len(result))]
    
    Delta_Lucid_mean = 0
    Delta_EarlyFatigue_mean = calculate_Deltamean(result['Delta_EarlyFatigue'])
    Delta_Fatigue_mean = calculate_Deltamean(result['Delta_Fatigue'])
    return Delta_Lucid_mean, Delta_EarlyFatigue_mean, Delta_Fatigue_mean


# choose one driver (E4695E4694FE, F1D881BB0FAB, F4A5B22A27A4, F8BE9DEE6C66, CD62BFF2FCEF)
Driver = "CD62BFF2FCEF"  
Year = 2022
Month = 5   

Input_path = os.path.join(os.getcwd() + "/01_input/clean data/" + str(Year) + "_" + calendar.month_name[Month] + "/sort out/" + Driver + "/Algo/")
Output_path = os.path.join(os.getcwd() + "/02_output/" + str(Year) + "_" + calendar.month_name[Month] + "/" + str(Driver) + "/")

Data = pd.read_csv(Input_path + "algo_" + Driver + ".csv")

### Prepare fatigue table ###
Result = Prepare_fatigue_table(Data, Year, Month)
show_fatigues_info(Result)

### Save data ###
Result.to_csv(Output_path + Driver + "_" + calendar.month_name[Month] + "_fatigue.csv", index=False)


### Barplot of Fatigue probability in continuous driving ###
Probability_list = Probability_alarm_record(Result, min_ = 10) #  5 or 10 min record
xlist = [(i+1)*10 for i in range(len(Probability_list))]
ylist = Probability_list

make_snapshot(snapshot, Fatigue_probability_barplot(xlist, ylist).render(Output_path + "Fatigue_probability" + ".html"),
              Output_path + "Fatigue_probability" + ".png")


### Pie chart of the times of different periods ###
datapie = Result.groupby(['Period'])['alarm_count'].agg('sum').to_dict()
Period = ['Morning', 'Noon', 'Afternoon', 'Night', 'Midnight']
# Period 補上 0
for period in Period:
    if period in datapie:
        datapie[period]=datapie[period]
    else:
        datapie[period]=0
xlist = list(datapie.keys())
ylist = list(datapie.values())

make_snapshot(snapshot, Periods_piechart(xlist, ylist).render(Output_path + "Periods_counts.html"), Output_path + "Periods_counts.png")


### Lineplot of fatigue probability by week ###
xlist = weeks_in_month(Year, Month)
ylist1 = travel_per_weeks(Result, xlist)
ylist2 = alarm_percentage_per_weeks(Result, xlist)

make_snapshot(snapshot, overlap_mutil_yaxis(xlist, ylist1, ylist2).render(Output_path + "Probability_of_fatigue_weeks.html"), 
              Output_path + "Probability_of_fatigue_weeks.png")


### Heart Rate difference trend in fatigue ###
Result = Result[Result['alarm_count'] > 0] # Only have fatigue alarm
Result.index = list(range(len(Result['alarm']))) # 改成連續index數值
if len(Result) == 0:    
    plt.figure(figsize = (12, 6), dpi = 200)
    plt.title("No alarm data", fontsize=30, weight='bold')
    plt.savefig(Output_path + "MeanHRDelta_of_fatigue_filter_15min.png") 

else:
    Lucid_Dmean, EarlyFatigue_Dmean, Fatigue_Dmean = calculate_meanHR_delta(Result, earlyboundary=3)
    xlist = ['Lucid', 'Early Fatigue', 'Fatigue']
    ylist = [Lucid_Dmean, EarlyFatigue_Dmean, Fatigue_Dmean]

    make_snapshot(snapshot, Deltamean_lineplot(xlist, ylist).render(Output_path + "MeanHRDelta_of_fatigue_filter_15min.html"), 
                  Output_path + "MeanHRDelta_of_fatigue_filter_15min.png")