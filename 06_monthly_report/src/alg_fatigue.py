
import numpy as np
import os
import datetime
import re

from numpy.core.numeric import NaN
import src.get_PVT as PVT
from scipy.interpolate import interp1d
import src.get_PERCLOS as perclos
import src.get_HRV as HRV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def get_correlation(x, y):
        x = x - np.mean(x)
        y = y - np.mean(y)
        C1 = np.dot(x, x.conj()) 
        C2 = np.dot(y, y.conj()) 
        C3 = np.dot(x, y.conj()) 
        result = [0,0,0]
        for i in range(len(x)):
            result[0] += x[i]**2
            result[1] += y[i]**2
            result[2] += x[i]*y[i]

        cor = C3 / (C1 * C2) **0.5
        cor2 = result[2] / (result[0] * result[1]) **0.5
        return cor
def resample_HR(HR, status, RESAMPLE_SIZE):
    HR_resmple = []
    temp = []
    temp_status = []
    for i in range(len(HR)):
        temp.append(HR[i])        
        temp_status.append(status[i])
        if len(temp) == RESAMPLE_SIZE :
            if np.mean(temp_status) < 3.5:
                HR_resmple.append(np.median(temp))
            temp = []
            temp_status = []
    return HR_resmple


def ar(alist):
    return np.array(alist)


def datetime_convert(date_time): 
    format = '%Y-%m-%d-%H-%M' # The format     
    datetime_str = datetime.datetime.strptime(date_time, format)    
    return datetime_str 



class Alg_fatigue():


    def __init__(self, model_path = ""): #fft_window_size=30
        """
            initial the main class with the path of the model.
        Args:
            model_path: Path of the model
        returns:
            None
        """
        self.list_result = []
        self.total_scores = []
        self.List_subject = []
        self.List_all_HR = []
        self.list_time_gd = []
        self.list_person_hr = []
        self.list_threshold = []
        self.list_gd_train = []

        self.List_KSS = []
        self.List_time_gd = []
        self.List_person_hr = []
        self.List_gd_train = []
        self.train_parameters = []
        self.train_interc = 0

        #=====================================
        self.number_collect_fatigue = 3
        self.RESAMPLE_SIZE = 30
        self.GLOBAL_THRESHOLD_PERSENTAGE = 40
        self.window_size_minute = 5
        self.KSS_min_traning = 4
        self.regression_mode = 0
        if model_path != "":
            self.load_model_file(model_path)

    def norm_HRlist_model(self, HR_list):
        """
            generate a normalized HR model.
        Args:
            HR_list: a serial of HR in the specific time long maybe 3 hours.
        returns:
            array of HR in the type of percentage
        """
        HR_list_sorted = sorted(HR_list)
        norm_list = []
        for i in range(100):
            
            norm_list.append(HR_list_sorted[int(i * len(HR_list_sorted) / 100)])
        return np.array(norm_list)

    def Fmodel_compute(self, List_all_HR):
        restingHR = (min(List_all_HR[int(60 * 2 / self.RESAMPLE_SIZE): int(60 * 10 / self.RESAMPLE_SIZE)]))
        d = restingHR - min(List_all_HR)
        d_ratio = round(d / restingHR, 4)
        List_all_HR = ar(sorted(List_all_HR))  
        personal_model = self.norm_HRlist_model(List_all_HR)
        y = np.arange(1,101)            
        R_value = get_correlation(ar(personal_model), y)
        fmodel_index = round(R_value + 0.004 * d, 3)

        return fmodel_index, d_ratio, personal_model
    def get_HR_list(self, HRV_dir):
        """
            read file from raw hr data.
        Args:
            HRV_dir: the path of the HR data.
        returns:
            the list of hr, status and confidence level
        """
        HRV_file_list = os.listdir(HRV_dir)
        file = HRV_file_list[0]

        HR = []
        status = []
        confidence_level = []
        with open(os.path.join(HRV_dir, file), 'r') as HRfile:
            for line in HRfile.readlines():
                line = line.split(',')
                HR.append(float(line[0]))
                status.append(float(line[1]))
                confidence_level.append(int(line[2]))
        return HR, status, confidence_level

    
    def get_time_from_raw(self, raw_path):
        time = []
        HRV_file_list = os.listdir(raw_path)
        for file in HRV_file_list:
            if ".log" in file:
                with open(os.path.join(raw_path, file), 'r') as RAWfile:
                    for line in RAWfile.readlines() :
                        if "Start Time" in line:
                            line = re.split(":|/|-", line)
                            time.extend(list(map(int, line[1:])))
                            time.pop()
                            break
                break
        return datetime.datetime(time[0], time[1], time[2], time[3], time[4])


    def get_HRV_list(self, HR, status, window_size_min):
        """
            compute the HRV from the HR serial.
        Args:
            HR: Heart rate serial by seconds.
            status: status serila by seconds.
            window_size_min: the windows length of minute for compute HRV
        returns:
            list_HR, list_SDNN, list_RMSSD, list_DF, and list_time_gd
            list_time_gd: the time of the HRV data
        """
        list_HR = []
        list_SDNN = []
        list_RMSSD = []
        list_DF = []
        self.list_time_gd = []
        self.window_size_minute = window_size_min
        start = 0
        starttime = datetime.datetime(1, 1, 1, 0, 0)
        currtime = datetime.datetime(1, 1, 1, 0, 0)
        delta_time = currtime - starttime
        while start < len(HR):   
            bpm = []
            sec = 0
            while start + sec < len(HR)-1:
                if HR[start + sec] > 55 and HR[start + sec] < 120:
                    bpm.append(HR[start + sec])
                if sec == self.window_size_minute * 60:
                    break                
                sec += 1
                                    
            if sec >= self.window_size_minute * 60 * 0.8: 
                start += sec                                      
                delta_time += datetime.timedelta(seconds = sec) # result show the fatigue in last 5 minutes
                if len(bpm) > sec * 0.7:
                    rri = (60/np.array(bpm))  * 1000
                    mean_HR = np.mean(bpm)  
                    SDNN = np.std(rri)
                    RMSSD = np.sqrt(np.mean((rri[1:] - rri[:-1])**2))
                    DF = HRV.get_DF(bpm)   
                else: 
                    mean_HR = -1
                    SDNN = -1
                    RMSSD = -1
                    DF = -1
                
            else:
                break                   

            self.list_time_gd.append(delta_time.seconds / 3600)
            list_HR.append(mean_HR)
            list_SDNN.append(SDNN)
            list_RMSSD.append(RMSSD)
            list_DF.append(DF)
    
        return list_HR, list_SDNN, list_RMSSD, list_DF, self.list_time_gd

    def get_KSS_from_PVT(self, PVT_dir):
        """
            get the KSS data from PVT data file, and interpolate the KSS to 
            the time corresponding the HRV.
        Args:
            PVT_dir: the path of the PVT data
        returns:
            list_PVT_result_interp
        """
        list_PVT_time = []
        list_PVT_result = []
        list_PVT_result_interp = []
        pvt_file_list = os.listdir(PVT_dir)
        starttime = 0
        for file in pvt_file_list:
            with open(os.path.join(PVT_dir, file), 'r') as line:
                pvt_data = line.read()
            #print('processing ', file)
            file = file.replace('PVT_', '')
            date_string = file.replace('.csv', '').split('_')[1] 
            currtime = datetime_convert(date_string)
            if starttime == 0:
                starttime = currtime
            curr_deltatime = currtime - starttime
            

            # if use_time == 0 or use_time == 1:
            list_PVT_time.append(curr_deltatime.seconds /3600)
            list_PVT_time.append((curr_deltatime.seconds + 600) / 3600)
            KSS1, KSS2 = PVT.get_KSS_text(pvt_data)
            list_PVT_result.append(KSS1)
            list_PVT_result.append(KSS2)
        interpolation = interp1d(list_PVT_time, list_PVT_result, kind = 'linear')


        if self.list_time_gd[0] < list_PVT_time[0]:
            self.list_time_gd[0] = list_PVT_time[0]
        for i in range(len(self.list_time_gd)):
            if self.list_time_gd[i] > list_PVT_time[-1]:
                self.list_time_gd[i] = list_PVT_time[-1]
        list_PVT_result_interp = interpolation(self.list_time_gd)
        list_PVT_result_interp = [np.round(i,1) for i in list_PVT_result_interp]
        return  list_PVT_result_interp         

    def get_normalization_paras(self, input):
        buf = []
        for i in range(len(input)):
            if input[i] > 0 :
                buf.append(input[i])
            if len(buf) >= 3 and np.mean(buf) != 0:
                input = input / np.mean(buf)
                return True, input
            
        return False, input
        

    def training_model_fatigue(self, X, Y, regression_mode):
        """
            train model for assigned X and Y by the regression. 
            The trained model would keep int this class. 
        Args:
            X: input dataset
            Y: output dataset
            regression_mode: the training method.
        returns:
            None
        """
        self.regression_mode = regression_mode
        if regression_mode == 1:
            lrGD=LogisticRegression(tol=0.001, max_iter=800, random_state=1)
            lrGD.fit(X,Y)
            self.train_interc = lrGD.intercept_[0] + 0.5
            self.train_parameters = lrGD.coef_[0] 
        elif regression_mode == 2:
            reg = LinearRegression().fit(X, Y)
            self.train_interc = reg.intercept_
            self.train_parameters = reg.coef_
        elif regression_mode == 3:
            pipe_svc = make_pipeline(StandardScaler(),
                            SVC(random_state=1))

            param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

            param_grid = [
                {'svc__C': param_range, 
                'svc__gamma': param_range, 
                'svc__kernel': ['rbf']}]

            svm = GridSearchCV(
                estimator=pipe_svc, 
                param_grid=param_grid, 
                scoring='accuracy', 
                refit=True,
                cv=5,
                n_jobs=-1
                )
            self.svm = svm.fit(X, Y)
            self.train_parameters = svm.best_params_
            self.train_interc = 0

        print('parameters in modes is ' + ','.join(map(str, self.train_parameters)) + ',' + str(self.train_interc))
    
    def testing_model_fatigue(self, X, list_time_gd = [], regression_mode = 0): 
        """
            test model for assigned X by the regression method
            which come from the the train function "training_model_fatigue"
            or the input model from "load_model_file"  
        Args:
            X: input dataset
            list_time_gd: option input for checking the alarm time
            regression_mode: the training method.
        returns:
            list_FI_view, list_alert_stage
        """
        if len(self.train_parameters) < 1:
            return
        scale_paras = len(self.train_parameters)
        if len(X) < 1:
            return
        if regression_mode == 0:
            if self.regression_mode == 0:
                return 0
            else:
                regression_mode = self.regression_mode

        list_FI_view = []
        list_alert_stage = []
        number_collect_fatigue = 0
        alarm_count = 0
        threshold = 0.5
        for j in range(len(X)): 
             
            fatigue_condition = 0
            isfatigue = 0
            fatigue_index = 0

            if  regression_mode == 3:
                temp = [X[j][k] for k in range(scale_paras)]
                fatigue_index = self.svm.predict([temp])[0] > 0.5
                number_collect_fatigue = 1
                isfatigue = fatigue_index
            else:
                if len(X[j]) == scale_paras:
                    for k in range(scale_paras):                    
                        fatigue_index += X[j][k] * self.train_parameters[k]                
                    fatigue_index += self.train_interc
                else:
                    fatigue_index = 0                      
                
                if regression_mode == 1:
                    isfatigue = fatigue_index >= threshold
                    number_collect_fatigue = 2
                elif regression_mode == 2:
                    isfatigue = fatigue_index > threshold
                    number_collect_fatigue = 3
            list_FI_view.append(fatigue_index)
            if len(list_time_gd) != 0:
                if list_time_gd[j] <= 1:
                    isfatigue = False

            if isfatigue:
                alarm_count += 1
                if alarm_count >= number_collect_fatigue and j * self.window_size_minute > 30:
                    fatigue_condition = 1                                      
            else:
                alarm_count = 0
            list_alert_stage.append(fatigue_condition)
        list_FI_view, list_alert_stage = self.optimize_fatigueindex(list_FI_view,list_alert_stage, threshold)
        
        return list_FI_view, list_alert_stage
    
    def save_model_file(self,path):
        """
            save the regression method
            which come from the the train function "training_model_fatigue"
            or the input model from "load_model_file"  
        Args:
            path: save path            
        returns:
            None
        """
        with open(path, 'w') as parafile:
            parafile.write(str(self.regression_mode) + '\n')
            for i in self.train_parameters:
                parafile.write(str(i) + '\n')
            parafile.write(str(self.train_interc) + '\n')    
        return 0
    
    def load_model_file(self, path):
        """
            load model method and parameters in the model file.
        Args:
            path: load path            
        returns:
            None
        """
        
        with open(path, 'r') as parafile:
            line = parafile.readline()
            self.regression_mode = int(line)
            if self.regression_mode == 3:
                scale_paras = 4
            else:
                scale_paras = 3
            for i in range(scale_paras):
                para = float(parafile.readline())
                self.train_parameters.append(para)
            para = float(parafile.readline())
            self.train_interc = para                
        return 0

    def optimize_fatigueindex(self, FI, alarm, threshold):
        status = 0
        time_to_fatigue = 12 * 60 / self.window_size_minute
        new_FI = []
        new_alarm = []
        last_FI_buffer = 0
        for i in range(len(FI)):
            if alarm[i] == 1 and status == 0:
                status = 1
                last_FI_buffer = FI[i]
            
            
            if status == 0:                
                new_FI.append(max(FI[i], threshold * i / time_to_fatigue))
                
            elif status == 1:
                if FI[i] >= last_FI_buffer:
                    last_FI_buffer = FI[i]
                    new_FI.append(FI[i])
                else:
                    new_FI.append(last_FI_buffer)
            new_alarm.append(status)
        return new_FI, new_alarm




        
        
        
        