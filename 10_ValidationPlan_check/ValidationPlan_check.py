import os, glob, re
import datetime
import shutil
import numpy as np
from os import listdir
import os
from numpy import genfromtxt
import re
import matplotlib.pyplot as plt
# from python.src.alg_freq_domain import Alg_freq_domain
import src.read_data as rd
from scipy import signal
import src.performance as pf
import pandas as pd
sample_rate = 64
FFT_window_size=32
shift = 10

preprocess = 1
Check = 1

####Setting
filepath = "20230410理想座椅测试"
if os.path.exists(filepath + "\\ground_truth_bpm\\"):
    os.makedirs(filepath + "\\ground_truth_bpm\\")
    
if os.path.exists(filepath + "\\raw\\"):
    os.makedirs(filepath + "\\raw\\")

# preprocess for the doc, move from Download to specific directory and rename csv files
if preprocess == 1 :


    for file in glob.iglob(filepath + "\*\*", recursive=True):
        new = str(file).replace("_Polar_GoldenHR","")
        new = new.split("\\")[-1]
        if "csv" in new :
            new = filepath + "\\ground_truth_bpm\\" + new
            if os.path.isfile(new):
                continue
            else:
                shutil.move(file,new)
        elif "log" in new:
            new = filepath + "/raw/" + new
            if os.path.isfile(new):
                continue
            else:
                shutil.move(file,new)


################################################################ 

# look through each file and calculate the Moving Error and AP5, then write into a csv called "results"
if Check == 1:

    file_list = rd.get_file_list(os.path.join(filepath, "raw"))
    for file in file_list:        

        ground_truth_filename = str(file).replace('log', 'csv')

        _, _, _, _, raw_status, raw_hr,_, start_time = rd.read_pressure_acc_result(os.path.join(filepath, "raw"), file)
        algo_data = raw_hr
        raw_hr_downsample = []
        for i in range(len(raw_hr)):
            if i % sample_rate == 0 and i > sample_rate * 8:
                raw_hr_downsample.append(raw_hr[i])

        algo_data = np.array(raw_hr_downsample)

        if os.path.isfile(os.path.join(filepath, "ground_truth_bpm", ground_truth_filename)):
            csv_isexist = True 
        else:
            print(str(file) + "  golden not found")
            continue
        if csv_isexist:    
            ground_truth_bpm, start_time_golden = rd.read_csv_with_time(os.path.join(filepath, "ground_truth_bpm", ground_truth_filename))

            ground_truth_bpm = ground_truth_bpm[shift:]
            if len(start_time_golden) > 0 and len(start_time) > 5:
                if len(start_time_golden) == 6 :
                    shift = round(start_time[5] - start_time_golden[5]) + 1
                    if shift < 0:
                        test_data = algo_data
                        ground_truth = ground_truth_bpm[shift * -1 :]
                    else:
                        test_data = algo_data
                        ground_truth = ground_truth_bpm[shift:]
                        if len(ground_truth) > len(test_data):
                            ground_truth = ground_truth[:len(test_data)]
                        else:
                            test_data = test_data[:len(ground_truth)]
                else:
                    test_data = algo_data[:]
                    ground_truth = ground_truth_bpm[:]
        
        
        test_data = test_data[len(test_data)-60:]
        ground_truth = ground_truth[len(ground_truth)-60:]
        vals_test_data,counts = np.unique(test_data, return_counts=True)
        mode_test_data = np.argmax(counts)
        vals_ground_truth,counts = np.unique(ground_truth, return_counts=True)
        mode_ground_truth = np.argmax(counts)
        Moving_Error=[]
        for i in range(len(test_data)-10):
            mean_golden = np.mean(ground_truth[i:i+10])
            Moving_Error.append(np.max(np.abs(test_data[i:i+10]-mean_golden)))

        AP5 = (pf.performance( test_data, ground_truth, bpm_error_tolerant=5)[0])
        MoE = (np.min(Moving_Error))
        print(str(file) + " AP5 = " + str(AP5) + "  MoE = " + str(MoE))


"""
    # mse_array = np.reshape(mse,(int(len(mse)/col),col))
    # Error_array = np.reshape(Error,(int(len(Error)/col),col))
    AP5_array = np.reshape(AP5,(int(len(AP5)/col),col))
    filecontents = np.reshape(filecontents,(int(len(filecontents)/col),col))
    MoE_array = np.reshape(MoE,(int(len(MoE)/col),col))


    if col ==5:
        colname = ['120度','100度說話','翹腳','側坐','泡綿']
    elif col ==7:
        colname = ['120度','100度說話','翹腳','側坐','泡綿','棉布','記憶海綿']

    # mse = pd.DataFrame(data = mse_array,columns=colname,index = np.arange(1, len(mse_array)+1))
    # Error = pd.DataFrame(data = Error_array,columns=colname,index = np.arange(1, len(Error_array)+1))
    AP5 = pd.DataFrame(data = AP5_array,columns=colname,index = np.arange(1, len(AP5_array)+1))
    file = pd.DataFrame(data = filecontents,columns=colname,index = np.arange(1, len(filecontents)+1))
    MoE = pd.DataFrame(data = MoE_array,columns=colname,index = np.arange(1, len(MoE_array)+1))
    savename = "project\Data\\verification_data\\" + "results.xlsx"
    writer =pd.ExcelWriter(savename)
    # mse.to_excel(writer, sheet_name='MSE', header=True, index=True, engine=None)
    # Error.to_excel(writer, sheet_name='Error', header=True, index=True, engine=None)
    AP5.to_excel(writer, sheet_name='AP5', header=True, index=True, engine=None)
    file.to_excel(writer, sheet_name='path', header=True, index=True, engine=None)
    MoE.to_excel(writer, sheet_name='Moving Error', header=True, index=True, engine=None)
    writer.save()

        # print("performance (Acc rate(AP5):", AP5)
        # print("performance (Error):", Error)
        # print("performance (MSE):", mse)
        # print(bpm.shape)

    # plt.show()
"""
