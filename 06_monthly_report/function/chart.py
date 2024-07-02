from email.utils import encode_rfc2231
from itertools import count
from operator import index
from statistics import mean
from tkinter.tix import ROW
from numpy import row_stack 
import numpy as np 
import pandas as pd
import csv
import os
from requests import head, post
import datetime
import json
import shutil
from turtle import begin_fill, width
from typing import Sequence
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# 將字體換成思源黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
from pyecharts import options as opts
from pyecharts.faker import Faker
from pyecharts.charts import *
from fractions import Fraction  # 分數表示
# 匯入輸出圖片工具
from pyecharts.render import make_snapshot
# 使用snapshot-selenium 渲染圖片
from snapshot_selenium import snapshot
from unittest import result
from xml.etree.ElementInclude import include
import calendar
from plotnine import *

""" HR Histogram Plot """
def plot_HR_distribution(result):
    plt.figure(figsize = (12, 6), dpi = 200)
    n, bins, patches = plt.hist(result, bins = 'auto', density = False, color = 'pink', width = 1)
    plt.xlabel("HR", fontsize = 22)
    plt.ylabel("Counts", fontsize = 20)
    # plt.title("心率表現分布", fontsize=30, weight='bold')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    # plt.show()
    return plt


""" Calendar """
# =============================================================================
# def Calendar_chart(timelist, begin, month) -> Calendar:
#     calendar = (
#         Calendar(opts.InitOpts(width="500px", height="300px"))
#         .add(series_name = "", 
#             yaxis_data = Times, 
#             calendar_opts = opts.CalendarOpts(range_= [str(begin)[0:7]],                
#                                             yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
#                                             pos_top=90,
#                                             pos_right=20,
#                                             # 星期軸的標示
#                                             daylabel_opts=opts.CalendarDayLabelOpts(name_map='cn'),
#                                             # 月份軸的樣式
#                                             monthlabel_opts=opts.CalendarMonthLabelOpts(name_map='cn') 
#                                             )
#             )
#         .set_global_opts(
#             title_opts = opts.TitleOpts(title = "2022 年 " + str(month) + "月 登錄時間(min)紀錄",    
#                                         pos_top=20, 
#                                         title_textstyle_opts=opts.TextStyleOpts(font_size=20)),
#             visualmap_opts = opts.VisualMapOpts(
#                 orient = "horizontal",
#                 is_piecewise=True, 
#                 pos_top="260px",
#                 pos_left="50px",
#                 split_number = 5,
#                 pieces = [{"min": 200},                                       # <-- 選擇範圍區間
#                     {"min": 150, "max": 200}, 
#                     {"min": 100, "max": 150},
#                     {"min": 50, "max": 100},
#                     {"min": 1, "max": 50} # "color": "grey" 
#                 ]
#             )
#         )
#     )
#     return calendar
# =============================================================================
def calendar_ggplot(dataframe):
    plot = (ggplot(dataframe, aes('monthweek', 'weekdays', fill = 'val')) +
                geom_tile() +
                geom_text(aes(label = 'mday'), size = 16, color = 'black') +
                theme_bw() +
                facet_wrap('~month' ,nrow=3) +
                theme(panel_grid_major = element_blank(), panel_grid_minor = element_blank(),
                      # axis_text=(element_blank()),
                      axis_text_x = element_text(size = 18),
                      axis_text_y = element_text(size = 18),
                      axis_title = element_text(size=18),
                      plot_title = element_text(size = 22),
                      strip_text_x = element_text(size = 30),
                   legend_text = element_text(size = 12),
                   legend_key_size = 10,
                   legend_key_height = 20) +
                scale_fill_gradientn(colors = ('white', '#47B5FF', '#4FD3C4', '#FEE440', '#EC4646'))+
                scale_x_continuous() 
    )
    return plot


""" daily bar plot """
def daily_barplot(days, record, month) -> Bar:
    barplot = (
        Bar()
        .add_xaxis(days)
        .add_yaxis(calendar.month_name[month], record, itemstyle_opts=opts.ItemStyleOpts(color='#33c0eb'))
        .set_global_opts(# title_opts=opts.TitleOpts(title="每天使用之分鐘數", title_textstyle_opts=opts.TextStyleOpts(font_size=30)),
                          legend_opts=opts.LegendOpts(is_show=False),
                          xaxis_opts=opts.AxisOpts(name='Date', name_textstyle_opts=opts.TextStyleOpts(font_size=22), name_location='middle', name_gap=35,
                                                   axislabel_opts=opts.LabelOpts(font_size=15)),
                          yaxis_opts=opts.AxisOpts(name='Using time (minute)', axislabel_opts=opts.LabelOpts(font_size=18), name_location="center",
                                                   name_textstyle_opts=opts.TextStyleOpts(font_size=22), name_gap=50))
    )
    return barplot


""" Weekly frequency boxplot """
def weekly_frequency_boxplot(weekly_df, month) -> Boxplot:
    boxplot = (
        Boxplot(init_opts=opts.InitOpts(theme="roma"))
        .add_xaxis(list(weekly_df['Week']))
        .add_yaxis(calendar.month_name[month], Boxplot.prepare_data(weekly_df['golden_hr']))  # itemstyle_opts=opts.ItemStyleOpts(color='blue')
        .set_global_opts(# title_opts=opts.TitleOpts(title="每週心率表現", title_textstyle_opts=opts.TextStyleOpts(font_size=30),
                          #                            subtitle="次數", subtitle_textstyle_opts=opts.TextStyleOpts(font_size=20)),
                          legend_opts=opts.LegendOpts(is_show=False),
                          xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20)),
                          yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=18))
        )
        # .render("Weekly.html")
    )
    return boxplot


""" HR at Periods """
def periods_hr_boxplot(period_df, month) -> Boxplot:
    boxplot = (
        Boxplot() # init_opts=opts.InitOpts(theme=ThemeType.MACARONS)
        .set_colors(['#001E6C'])
        .add_xaxis(list(period_df['period']))
        .add_yaxis(calendar.month_name[month], Boxplot.prepare_data(period_df['golden_hr']))  # itemstyle_opts=opts.ItemStyleOpts(color='blue')
        .set_global_opts(# title_opts=opts.TitleOpts(title="不同時期心率表現", title_textstyle_opts=opts.TextStyleOpts(font_size=30),
                          #                           subtitle="次數", subtitle_textstyle_opts=opts.TextStyleOpts(font_size=20)),
                          legend_opts=opts.LegendOpts(is_show=False),
                          xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20)),
                          yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=18))
        )
    )
    return boxplot


""" Barplot of Fatigue probability in continuous driving """

def Fatigue_probability_barplot(x, y) -> Bar:
    barplot = (
        Bar() # width="1600px", height="800px"
        .add_xaxis(x)
        .add_yaxis('', y, itemstyle_opts=opts.ItemStyleOpts(color='red'), gap='50%', bar_width='50%')
        .extend_axis(yaxis=opts.AxisOpts(type_="value",
                                         name="Probability",
                                         name_textstyle_opts=opts.TextStyleOpts(font_size=22),
                                         position="left"))
        .set_global_opts(# title_opts=opts.TitleOpts(title="偵測疲勞發生率", title_textstyle_opts=opts.TextStyleOpts(font_size=25)), 
                         xaxis_opts=opts.AxisOpts(name='Driving time (minute)', name_textstyle_opts=opts.TextStyleOpts(font_size=22), name_location='middle', name_gap=35,
                                                axislabel_opts=opts.LabelOpts(font_size=15)),
                         yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=True, font_size=18, formatter='{value} %'), type_ ="value"), # 加百分比
                         legend_opts=opts.LegendOpts(item_gap=30, textstyle_opts=opts.TextStyleOpts(font_size=22)))
        .set_series_opts(label_opts=opts.LabelOpts(font_size=16))
    )
    return barplot

"""--------------------------------------------------------------------------------------------"""                                                                                                                                                           
""" Pie chart of the times of different periods """

def Periods_piechart(x, y) -> Pie:
    piechart = (
        Pie()
        .add("", [list(z) for z in zip(x, y)]) # radius=["10%", "55%"]
        .set_colors(["#22577A", "#38A3A5", "#57CC99", "#8FC1D4", "#055052", "#9A86A4"])
        .set_global_opts(# title_opts=opts.TitleOpts(title="各時段疲勞次數", title_textstyle_opts=opts.TextStyleOpts(font_size=30)),
                        legend_opts=opts.LegendOpts(item_gap=25, textstyle_opts=opts.TextStyleOpts(font_size=30), pos_bottom=5)) 
                        # orient="vertical", pos_top="15%", pos_left="2%"
        .set_series_opts(label_opts=opts.LabelOpts(font_size=20, formatter="{b}: {c}")) # a（系列名稱），b（數據項名稱），c（數值）, d（百分比）
    )
    return piechart

"""--------------------------------------------------------------------------------------------"""
""" Lineplot of fatigue probability by week """
## multi_yaxis ###
def overlap_mutil_yaxis(xlist, ylist1, ylist2) -> Grid:
    barplot = (
        Bar()
        .add_xaxis(xlist)
        .add_yaxis('', ylist1, itemstyle_opts=opts.ItemStyleOpts(color='#30AADD'), gap='50%', bar_width='50%', yaxis_index=0)
        .extend_axis(xaxis=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20)),
                    yaxis=opts.AxisOpts(type_="value",
                                        name="Probability",
                                        name_textstyle_opts=opts.TextStyleOpts(font_size=20),
                                        min_=0,
                                        max_=100,
                                        position="left",
                                        axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="red")),
                                        axislabel_opts=opts.LabelOpts(font_size=18, formatter="{value} %"))) 
        .set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20)),
                        yaxis_opts=opts.AxisOpts(name="Using time",
                                                name_textstyle_opts=opts.TextStyleOpts(font_size=20),
                                                # min_=50,
                                                # max_=500,
                                                position="right",
                                                offset=10, #間隔
                                                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#30AADD")),
                                                axislabel_opts=opts.LabelOpts(formatter="{value} min", font_size=18)),
                        # title_opts=opts.TitleOpts("平均週行程之發生疲勞機率"),
                        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                        legend_opts=opts.LegendOpts(item_gap=30, textstyle_opts=opts.TextStyleOpts(font_size=22))
        )
        .set_series_opts(label_opts=opts.LabelOpts(font_size=16))
    )
    lineplot = (
        Line()
        .add_xaxis(xaxis_data = xlist)
        .add_yaxis(series_name = "", y_axis = ylist2, yaxis_index = 1, color = "red", 
                   label_opts = opts.LabelOpts(is_show = True), z_level = 3) # z_level 越大越前面
        # .set_global_opts(legend_opts=opts.LegendOpts(is_show=True, pos_right="10%"),
        .set_series_opts(label_opts = opts.LabelOpts(font_size = 16))
    )
    barplot.overlap(lineplot)
    return barplot

"""--------------------------------------------------------------------------------------------"""
""" 疲勞區間之心率平均 """
# ### Input HR data ###
# # data_filter = pd.read_csv("test" + str(Driver) + "_" + calendar.month_name[Month] + "_" + "filtered.csv")
# data_filter = pd.read_csv(r"D:\Chris\Monthly report\Info\March\test4\test4_March_filtered.csv")
# data_filter['golden_hr'] = list(map(eval, data_filter["golden_hr"]))
# golden_hr=[]
# for i in range(len(data_filter['golden_hr'])):
#     golden_hr.append(str(data_filter['golden_hr'][i]))

# golden_hr_timestamp=[]
# for i in range(len(data_filter['timestamp'])):
#     golden_hr_timestamp.append(int(data_filter['timestamp'][i]))

# HRdf = pd.DataFrame(golden_hr, index = golden_hr_timestamp, columns = ['golden_hr'])

# ### 浮點數轉整數 ###
# data['timestampelist'] = list(map(eval, data["timestampelist"]))
# timelist=[]
# for i in data["timestampelist"]:
#     i = [int(j) for j in i]
#     timelist.append(i)
# data["timestampelist"] = timelist
# data['alarm'] = list(map(eval, data["alarm"]))

# ### Only have fatigue alarm ###
# data=data[data['alarm_count']>0]
# data.index=list(range(len(data['alarm'])))

# def cluster_fatigue_hr(fatigue_data, HR_df):
#     energyTstampList=[]
#     tiredTstampList=[]
#     sleepyTstampList=[]
#     for n in range(len(fatigue_data)):
#         for i in range(len(fatigue_data['alarm'][n])):
#             if fatigue_data['alarm'][n][i] ==0:
#                 energyTstampList.append(fatigue_data['timestampelist'][n][i])
#             elif (fatigue_data['alarm'][n][i] ==1) & (sum(fatigue_data['alarm'][n][:i]) <5):
#                 tiredTstampList.append(fatigue_data['timestampelist'][n][i])
#             else:
#                 sleepyTstampList.append(fatigue_data['timestampelist'][n][i])
#     energydf = pd.DataFrame(energyTstampList, index = energyTstampList, columns = ['energytimestamp'])
#     tireddf = pd.DataFrame(tiredTstampList, index = tiredTstampList, columns = ['tiredtimestamp'])
#     sleepydf = pd.DataFrame(sleepyTstampList, index = sleepyTstampList, columns = ['sleepytimestamp'])
#     energy_inner = HR_df.merge(energydf, how='inner', left_index=True, right_index=True)
#     tireddf_inner = HR_df.merge(tireddf, how='inner', left_index=True, right_index=True)
#     sleepydf_inner = HR_df.merge(sleepydf, how='inner', left_index=True, right_index=True)

#     return list(energy_inner['golden_hr']), list(tireddf_inner['golden_hr']), list(sleepydf_inner['golden_hr'])                                              

# energyHRList, tiredHRList, sleepyHRList = cluster_fatigue_hr(data, HRdf)

# ### Plot ###
# def calculate_HRmean(List):
#     lst = list(map(eval, List))
#     lst = [element for lis in lst for element in lis]
#     target1 = (-1)
#     target2 = 0
#     for x in lst:
#         if x == target1:
#             lst.remove(x)
#         elif x == target2:
#             lst.remove(x)
#     return np.round(np.mean(lst), decimals=2)

# energy = calculate_HRmean(energyHRList)
# tired = calculate_HRmean(tiredHRList)
# sleepy = calculate_HRmean(sleepyHRList)

# x_data = ['Energy'+' '*25+'Tired'+' '*25+'Sleepy']
# y_data1 = [energy]
# y_data2 = [tired]
# y_data3 = [sleepy]

# color = ['skyblue', 'pink', 'red']

# def bar_plot() -> Bar:
#     barplot = (
#         Bar()
#         .add_xaxis(xaxis_data=x_data)
#         .add_yaxis("Energy (fatigue alarm = 0)", y_axis = y_data1, itemstyle_opts=opts.ItemStyleOpts(color=color[0]))
#         .add_yaxis("Tired (fatigue alarm 1~5)", y_axis = y_data2, itemstyle_opts=opts.ItemStyleOpts(color=color[1]))
#         .add_yaxis("Sleepy (fatigue alarm >5)", y_axis = y_data3, itemstyle_opts=opts.ItemStyleOpts(color=color[2]))
#         # .extend_axis(yaxis=opts.AxisOpts(name='心率次數', name_textstyle_opts=opts.TextStyleOpts(font_size=20), position='left'))
#         .set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=20)),
#                         yaxis_opts=opts.AxisOpts(min_=50, max_=110, name='次數', name_textstyle_opts=opts.TextStyleOpts(font_size=20),
#                                                 axislabel_opts=opts.LabelOpts(font_size=15)), # name_location='middle'
#                         legend_opts=opts.LegendOpts(item_gap=30, textstyle_opts=opts.TextStyleOpts(font_size=12))) # pos_bottom=10, 
#         .set_series_opts(label_opts=opts.LabelOpts(font_size=18))
#     )
#     # .render("Probability_of_fatigue.html")
#     return barplot
# make_snapshot(snapshot, bar_plot().render("MeanHR_of_fatigue_filter.html"), "MeanHR_of_fatigue_filter.png")

"""--------------------------------------------------------------------------------------------"""
""" Heart Rate difference trend in fatigue """
# xlist = ['Energy', 'Tired', 'Sleepy']
# ylist = [energy, tired, sleepy]

def Deltamean_lineplot(xlist, ylist) -> Line:
    lineplot = (
        Line()
        .add_xaxis(xaxis_data = xlist)
        .add_yaxis(series_name = "Mean Heart Rate Deviation from Baseline (Lucid)", y_axis = ylist, color = "red", label_opts = opts.LabelOpts(is_show = True), 
                   itemstyle_opts = opts.ItemStyleOpts(border_width = 4, border_color = "pink", color = "red")
                   )   
        .set_global_opts(xaxis_opts = opts.AxisOpts(axislabel_opts = opts.LabelOpts(font_size = 22)),
                         yaxis_opts = opts.AxisOpts(name = '', name_textstyle_opts = opts.TextStyleOpts(font_size = 15), min_ = -15, max_ = 15,
                                                axislabel_opts = opts.LabelOpts(font_size = 15)), # name_location='middle' 
                         legend_opts = opts.LegendOpts(item_gap = 30, textstyle_opts = opts.TextStyleOpts(font_size = 22))
                        )
        .set_series_opts(label_opts = opts.LabelOpts(font_size = "22"))
    )
    return lineplot
