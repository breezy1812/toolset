import numpy as np
import scipy
import math
from scipy.interpolate import interp1d
from . import filter
from scipy.signal import find_peaks, lfilter, peak_prominences,firwin, filtfilt, hilbert
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import pylab as plt
import json
from scipy.signal import decimate

import pandas as pd
import os
import pickle
from scipy.stats import kurtosis, skew


    
# with open('status_svm.pickle', 'rb') as f:
#     status_svm = pickle.load(f)

# with open('re_ss_SVM_20220610_2.pickle', 'rb') as f:
# with open('svm.model', 'rb') as f:
#     re_svm_2=pickle.load(f)
# with open('re_ss_SVM_20220622_1.pickle', 'rb') as f:
#     re_svm_1=pickle.load(f)


def normalize_peak(data):
    max_data = np.max(data)
    min_data = np.min(data)
    rang_data = max_data - min_data
    data_n = (data - min_data) / rang_data
    return data_n

def normalize_area(data):
    data_sum = np.sum(data)
    data_n = data/data_sum
    return data_n

class Alg_freq_domain():
    def __init__(self, fs=64, fft_window_size=16): #fft_window_size=30
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []
        self.fft_window_size = fft_window_size
        self.fs = fs
        # =========Parameters==========
        self.max_overlap = 7
        self.iteration_times = 1
        self.overlap_weight = np.array([1, 1, 1, 1, 1, 1, 1])
        self.overlap_weight_default = np.array([1, 1, 1, 1, 1, 1, 1])
        self.bdt_weight = 0.4
        self.als_threshold = 4
        self.engine_threshold = 2
        self.reliable_threshold = 0.75
        self.reserved_freq_lower_bound = int(self.fft_window_size / 3)    # 10
        self.reserved_freq_upper_bound = int(self.fft_window_size * 10)   # 300
        self.SITTING_THRESHOLD = 3000000
        self.DRIVING_THRESHOLD = 3
        self.IDLE_THRESHOLD = 0.8 # 0.8
        self.MOVEMENT_THRESHOLD_LIMIT = 0.025 # 0.025
        self.scenario_clip_th_para = 3
        self.ss_t_len = 6  # default:5 
        self.threshold_nobody_std = 500
        self.threshold_engine_peak = 0.3    #by mads
        self.threshold_down_acc_truck = 3   #by mads
        self.HRV_t_len = 60 * 5  # 5 minutes
        self.HRV_t_interval = 60 * 1 # 1 minute
        self.vigor_parameter = [-3.62352, -0.347400, 0.19230, 4.07]
        self.number_collect_fatigue = 3
        self.number_learning_vigor = 15
        # ==========Flags===============
        # 0: No Spectrum subtraction, 1: ALS, 2: Direct ss, 3: spec_sub_trans, 4: spec_sub_old, 5: spec_sub_diff
        self.USE_SPECTRUM_SUBTRACTION = 4
        # 0: No Filter, 1: FIR filter, 2: IIR filter
        self.USE_FILTER = 1 
        self.USE_TIME_FUSION = 0
        self.USE_BDT = 1
        self.USE_ENGINE_DENOISE = 1
        self.USE_TRANSFER_FUNC = 1
        self.USE_BPM_POST_PROCESSING = 1
        self.USE_NO_JUMP_BPM = 1
        self.USE_GET_RPM = 0
        self.USE_LOADED_OVERLAP_WEIGHT = 2 # 0: generate list, 1: load stored list, 2: use default list
        self.USE_WHOLE_LOG_TRANSFER_FUNC = 0
        self.USE_WHOLE_LOG_SCENARIO = 0
        self.USE_HIGH_HR_STRATEGY = 1       #by mads
        self.USE_SLIGHT_MOVE_STRATEGY = 0   #by mads
        self.USE_EGINE_DETECT_EXTEND = 1    #by mads
        self.USE_NEW_NO_VITAL_SIGN = 0      #by mads
        self.USE_STATE_MECHINE = 1          #by mads
        self.USE_TRAIN_FOR_DEEPBREATH = 0
        self.USE_EXTEND_FFT_RESOLUTION = 0
        self.USE_REAL_DECIMATE = 1
        self.USE_ORIGIN_STFT = 0
        self.USE_CHECK_STATUS_AFTER_SS = 1
        self.USE_DE_ENGINE_NOISE_BOUND = 1  #by mads
        self.USE_DETECT_TRUCK = 1           #by mads
        self.USE_KNN_SCENARIO = 0           #by andrew
        self.USE_ACC_FEATURE = 0 
        self.USE_VIGOR_COMPUTE = 1          #by mads
        self.USE_FATIGUE_ALARM = 1          #by mads  
        self.USE_SVM_reliability = 0        #by andrew
        # ==========Filter==============
        self.bpm_filter_order = 4
        self.bpm_p_cutoff_freq = [1.6, 10]
        self.bpm_g_cutoff_freq = [1.6, 15.5]#[1.6, 31.5]
        self.rpm_filter_order = 2
        self.rpm_cutoff_freq = [0.05, 1]
        self.bpm_search_lower = int(self.fft_window_size*25/30) # 25
        self.bpm_search_upper = int(self.fft_window_size*55/30) # 55
        self.bpm_overlap_upper = int(self.fft_window_size*60/30) # 60
        self.bpm_engine_search_lower = int(self.fft_window_size*150/30) #150
        self.bpm_engine_search_upper = int(self.fft_window_size * 10)   # 300
        # ============================== buffer
        self.ground_truth_bpm = np.array([])
        self.stable_index = []
        self.p_sensor_data = np.array([])
        self.g_sensor_data = np.array([])
        self.filter_p_sensor_data = np.array([])
        self.filter_g_sensor_data = np.array([])
        self.filter_time_bpm_data = np.array([])
        self.filter_time_g_sensor_data = np.array([])
        self.status = np.array([])
        self.confidence_level = [] 
        self.bcg_power = np.array([])
        self.acc_power = np.array([])
        self.ss = np.array([])
        self.nss = np.array([])
        self.tss = np.array([])
        self.time_fusion_data = np.array([])
        self.rpm_s = np.array([])
        self.rpm_data_out = np.array([])
        self.filter_rpm_data = np.array([])
        self.rpm = np.array([])
        self.rpm_overlap = np.array([])
        self.non_filter_g_spec = np.array([])
        self.abs_g_spec = np.array([])
        self.bpm_pre = np.array([])
        self.bpm = np.array([])
        self.ss_denoise_overlaped = np.array([])
        self.time_corrcoef = np.array([])
        self.ss_denoise = np.array([])
        self.bpm_interval = np.array([])
        self.engine_noise = np.array([])
        self.similarity = []   #by mads
        self.spec_peak_height = [] # by mads
        self.reliability = []
        self.nss_sum = []      #by mads
        self.engine_peaks = []
        self.bcg_std_power = [] #by mads
        self.HR_serial_HRV = []
        self.status_serial_HRV = []
        self.HRV_reference_vigor = []
        self.FEWS_vigor = []
        self.FEWS_alarm = []
        self.power_diff = []
        # ============================== ALS ========================================
        self.peaks_locs = np.array([])
        self.peaks_amps = np.array([])
        self.peaks_locs_o = np.array([])
        self.peaks_amps_o = np.array([])
        self.ss_max_arr = np.array([])  # ss normaization value
        self.nss_max_arr = np.array([])  # ss normaization value
        self.ss_status = np.array([])  # ss normaization value
        self.ld = 100
        self.engine_noise_search_range = 2
        self.bpm_no_jump_range = int(self.fft_window_size*5/30) #10
        self.try_range_extend = 0
        self.trans_func_time_len = 120
        self.stft_reduce_scale = 1
        self.decimate_scale = 1
        self.golden_bpm = np.array([])
        self.golden_harmonic_spectrum = np.array([])
        self.ss_clip = np.array([])
        self.nss_clip = np.array([])
        self.acc_org_peak_ratio = np.array([])
        self.state_mechine = 0
        self.truck_or_not = 0 #add by mads. used to check truck engine
        self.vigor_learning_progress = 0
        


    def load_paras_from_json_file(self, filename):                
        with open(filename) as data_file:
            data_loaded = json.load(data_file)
            self.overlap_weight = np.array(data_loaded['overlap_weight'])

    @staticmethod
    def normalization(STFT):
        ss_max = np.zeros(STFT.shape[1])
        for i in range(STFT.shape[1]):
            max_array = np.amax(STFT[:, i])
            ss_max[i] = max_array
            if max_array == 0:
                max_array = 1
            STFT[:, i] = STFT[:, i] / max_array
        return ss_max

    @staticmethod
    def cos_sim(vector_a, vector_b):
        vector_a = vector_a.copy() - 0.5
        vector_b = vector_b.copy() - 0.5
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom if denom != 0 else -1
        sim = 0.5 + 0.5 * cos
        return sim

    @staticmethod
    def reserve_freq_band(SS, lower_bound, upper_bound):
        SS[0:lower_bound] = 0
        SS[upper_bound:] = 0

    def spec_sub_direct(self, ss, nss):
        result = np.zeros(ss.shape[0])
        h_range = [self.fft_window_size, self.fft_window_size*10]
        result[h_range[0]:h_range[1]] = normalize_peak(ss[int(h_range[0]):int(h_range[1])]) - normalize_peak((nss[int(h_range[0]):int(h_range[1])]))        
        result[np.where(result<0)] = 0        
        result = normalize_peak(result)       
        
        return result

    def spec_sub_old(self,ss, nss):
        dr = 3
        heart_beat_range = [self.fft_window_size, self.fft_window_size*10]  # [32, 320]
        result = np.zeros(ss.shape[0])
        ss = ss[int(heart_beat_range[0]):int(heart_beat_range[1])]
        nss = nss[int(heart_beat_range[0]):int(heart_beat_range[1])]
        diff_ss = np.around((ss[dr:] - ss[:-dr]) * 50)
        diff_nss = np.around((nss[dr:] - nss[:-dr]) * 50)
        spec_subtract = diff_ss > diff_nss
        count = 0
        for i in range(len(spec_subtract)):
            if spec_subtract[i]:
                count += 1
                result[i + int(heart_beat_range[0] + 1)] = count ** 0.8
            elif count > 0:
                count -= 1

                result[i + int(heart_beat_range[0] + 1)] = count ** 0.8

        return result

    def spec_sub_diff(self,ss, nss):
        result = np.zeros(ss.shape[0])
        dr = 3
        drh = dr//2
        hr0  = int(self.fft_window_size)
        hr1  = int(self.fft_window_size*10)
        ss  =  ss[hr0:hr1]
        nss = nss[hr0:hr1]        
        diff  = (ss[dr:] -  ss[:-dr]) - (nss[dr:] - nss[:-dr])
        for i in range(len(diff)):
            result[i + hr0 + drh] = diff[i]*5+0.6 #*4+0.5
            if result[i + hr0 + drh] < 0:  
                result[i + hr0 + drh] = 0

        return result
    
    def add_elemet_peaks(self, peaks, element):
        new_list = []
        for peak in peaks:
            new_list.append(peak)
        new_list.append(element)
        return np.array(new_list)
    
    def find_boundary_peaks(self, data, peaks, bound_len):
        start = len(data) - bound_len
        end = len(data)
        bound_peak, bound_peak_val = find_peaks(data[start:end])
        if len(bound_peak) == 0 or len(peaks) == 0:
            return peaks
        else:
            for p in bound_peak:
                new_peak = start + p
                if min(abs(peaks - new_peak)) <= 2:
                    continue
                peaks = self.add_elemet_peaks(peaks , new_peak)
            return peaks


    def de_engine_noise(self, ss, nss, engine_noise, truck_or_not):
        bound_left = self.bpm_engine_search_lower
        bound_right = self.bpm_engine_search_upper
        engine_noise_window = engine_noise[bound_left:bound_right]
        engine_peaks, _ = find_peaks(engine_noise_window, height=self.threshold_engine_peak)
        if truck_or_not:
            self.threshold_engine_peak = 0.2

        ss_window = ss[bound_left:bound_right]
        if np.max(ss_window) > 0:
            ss_window = ss_window / np.max(ss_window)
        nss_window = nss[bound_left:bound_right]
        if np.max(nss_window) > 0:
            nss_window = nss_window / np.max(nss_window)

        nss_peaks, _ = find_peaks(nss_window, height=self.threshold_engine_peak)
        ss_peaks, _ = find_peaks(ss_window, height=self.threshold_engine_peak)
        
        # bound_peaks, _ = find_peaks(nss_window[bound_right - bound_left -1 - 10 :], height=0.1)
        
        if self.USE_DE_ENGINE_NOISE_BOUND == 1 :

            engine_peaks = self.find_boundary_peaks(engine_noise_window, engine_peaks, 10)
            nss_peaks = self.find_boundary_peaks(nss_window, nss_peaks, 10)
            ss_peaks = self.find_boundary_peaks(ss_window, ss_peaks, 10)
            
        
        if len(engine_peaks) == 0 or len(nss_peaks) == 0 or len(ss_peaks) == 0:
            return ss

        if len(self.engine_peaks) > 0:
            engine_peaks = np.append(engine_peaks, self.engine_peaks[-1])

        peak_list = []
        upper_est = 0
        lower_est = bound_right
        for engine_peak in engine_peaks:
            nss_engine_peak = np.abs(nss_peaks - engine_peak)
            if np.min(nss_engine_peak) < 5:
                close_peak = nss_peaks[np.argmin(nss_engine_peak)]
                ss_engine_peak = np.abs(ss_peaks - close_peak)
                if np.min(ss_engine_peak) < 5:
                    max_peak = ss_peaks[np.argmin(ss_engine_peak)] + bound_left
                    if lower_est <= max_peak <= upper_est:
                        continue
                    max_value = ss[max_peak]
                    upper = max_peak + 1
                    lower = max_peak - 1
                    value = max_value
                    while ss[upper] < value:
                        value = ss[upper]
                        upper += 1

                    top_value = value

                    value = max_value
                    while ss[lower] < value:
                        value = ss[lower]
                        lower -= 1

                    ss[int(lower):int(upper)] = np.linspace(value, top_value, num=int(upper)-int(lower))
                    peak_list.append(max_peak - bound_left)
                    if lower_est > lower:
                        lower_est = lower
                    if upper_est < upper:
                        upper_est = upper
        self.engine_peaks.append(peak_list)
        return ss
    
    def get_trans_func_whole_log(self, ss, nss, status):
        driving_spec = ss[:, status == 5]
        driving_acc = nss[:, status == 5]
        
        driving_sum = np.sum(driving_spec, axis=1)
        driving_acc_sum = np.sum(driving_acc, axis=1)
        driving_sum[driving_acc_sum == 0] = 1
        driving_acc_sum[driving_acc_sum == 0] = 1
        return driving_sum / driving_acc_sum
        
    def get_trans_func_clip_log(self, ss, nss, status):
        driving_sum = np.zeros(ss.shape)
        driving_acc_sum = np.zeros(ss.shape)

        if self.trans_func_time_len == 0 or self.trans_func_time_len > 30 * 60:
            self.trans_func_time_len = ss.shape[1]

        for i in range(ss.shape[1]):
            if i < self.trans_func_time_len:
                driving_sum[:, i] = np.sum(ss[:, 0: i + 1][:, status[0: i + 1] == 5], axis=1)
                driving_acc_sum[:, i] = np.sum(nss[:, 0: i + 1][:, status[0: i + 1] == 5], axis=1)
            else:
                driving_sum[:, i] = np.sum(ss[:, i + 1 - self.trans_func_time_len: i + 1][:, status[i + 1 - self.trans_func_time_len: i + 1] == 5], axis=1)
                driving_acc_sum[:, i] = np.sum(nss[:, i + 1 - self.trans_func_time_len: i + 1][:, status[i + 1 - self.trans_func_time_len: i + 1] == 5], axis=1)
        driving_sum[driving_acc_sum == 0] = 1
        driving_acc_sum[driving_acc_sum == 0] = 1
        return driving_sum / driving_acc_sum

    def get_trans_func(self, ss, nss, status):
        if self.USE_WHOLE_LOG_TRANSFER_FUNC == 0:
            self.trans_func = self.get_trans_func_clip_log(ss, nss, status)
        else:
            self.trans_func = self.get_trans_func_whole_log(ss, nss, status)

    def get_ss(self, ss, nss, status, engine_noise):
        ss_status = np.zeros(ss.shape[1]) 
        if self.USE_SPECTRUM_SUBTRACTION == 0:
            return ss, ss_status        

        ss = ss.T
        nss = nss.T
        engine_noise = engine_noise.T
        ss_all = np.copy(ss)
        
        for i in range(ss.shape[0]):
            #ss_status[i] = 1
            if status[i] > self.als_threshold:
                ss_status[i] = 3
                if self.USE_TRANSFER_FUNC:
                    if len(self.trans_func.shape) == 1:
                        nss[i] = self.trans_func*nss[i]
                    elif len(self.trans_func.shape) == 2:
                        nss[i] = self.trans_func[:, i]*nss[i]
                        #nss[i,np.where(nss[i]>1)] = 1
                if self.USE_SPECTRUM_SUBTRACTION == 1:
                    ss_all[i] = ss_all[i] = self.get_gd_of_one_window_opt(ss[i], nss[i],[self.fft_window_size, self.fft_window_size * 10])  
                elif self.USE_SPECTRUM_SUBTRACTION == 2:
                    ss_all[i] = self.spec_sub_direct(ss[i], nss[i])  
                elif self.USE_SPECTRUM_SUBTRACTION == 4:
                    ss_all[i] = self.spec_sub_old(ss[i], nss[i])  
                elif self.USE_SPECTRUM_SUBTRACTION == 5:
                    ss_all[i] = self.spec_sub_diff(ss[i], nss[i])  
                    #plt.plot(ss_all[i],'r')
                    #plt.plot(self.golden_harmonic_spectrum[:,idx],'g')
                    #plt.show()                    
            elif status[i] > self.engine_threshold and self.USE_ENGINE_DENOISE:
                ss_status[i] = 4
                ss_all[i] = self.de_engine_noise(ss[i], nss[i], engine_noise[i], self.truck_or_not)  # org
                #ss_all[i] = self.spec_sub_diff(ss[i], nss[i],i)  

        return ss_all.T, ss_status

    def cost_function(self, a, x, y):
        z = (y - a * x)
        d2z = (z[2:]) - 2 * (z[1:-1]) + (z[0:-2])
        w = np.ones(len(a)) * 0.1
        w[z <= 0] = round((1 - 0.1) * 2 / (1 + math.exp(1)), 3)
        fit = np.sum(w * (z ** 2))
        smooth = np.sum(d2z ** 2)
        return fit + self.ld * smooth

    def get_gd_of_one_window_opt(self, BCG, ACC, prs_range):
        bcg = BCG[prs_range[0]:prs_range[1]]
        acc = ACC[prs_range[0]:prs_range[1]]
        scaling = np.ones(bcg.size) * 1
        SS = np.zeros(BCG.size)
        scaling_best = minimize(self.cost_function, scaling, args=(acc, bcg), method='Powell', tol=0.1).x
        ss = bcg - scaling_best * acc
        SS[prs_range[0]:prs_range[1]] = ss
        SS = SS - min(SS)  # Jimmy
        return SS  # Jimmy

    def get_overlap_spectrum(self, STFT, overlap_weight):
        stft_overlap = np.zeros((STFT.shape[0], STFT.shape[1]))
        for i in range(0, self.bpm_overlap_upper *2):
            for j in range(len(overlap_weight)):
                if ((i + 1) * (j + 1)) < STFT.shape[0]:
                    start =  i * (j + 1);
                    end = (i + 1) * (j + 1);
                    moving_max = np.mean(STFT[start: end], axis=0)
                    stft_overlap[i] += moving_max * overlap_weight[j]
        return stft_overlap      
        
    def compensate_overlap_spectrum(self, stft_overlap_in, overlap_weight):
        stft_overlap = np.zeros((stft_overlap_in.shape[0], stft_overlap_in.shape[1]))
        for i in range(self.bpm_overlap_upper):
            reimburse = 0
            count = 0
            for j in range(7):
                for k in range(j):
                    idx = int(i*(k+1)/(j+1) + 0.5)
                    #if np.mean(stft_overlap_in[idx])>0.1:
                    if (k+1)/(j+1)>0.4 and k!=j :
                        reimburse += stft_overlap_in[idx] * overlap_weight[k] * overlap_weight[j]
                        count += 1 
            
            sss2 = (stft_overlap_in[i] + reimburse/count) #*2+0.5
            sss2[sss2 < 0] = 0
            stft_overlap[i] = sss2
        return stft_overlap

    def get_overlap_index(self,ssi, bpm_peak):
        peak_num = 0
        overlap_index = 0
        for mul in range(1, self.max_overlap + 1):
            overlap_ss = ssi[::mul]
            peaks, _ = find_peaks(overlap_ss, height=0.5)
            if not peaks.any():
                continue
            diff = np.abs(peaks - bpm_peak)
            min_index = np.argmin(diff)
            min_diff = diff[min_index]
            if min_diff < 2:
                overlap_index += 1 << (mul - 1)
                peak_num += 1
        return overlap_index, peak_num

    def get_overlap_index_mads(self,ssi, bpm_peak):
        peak_num = 0
        overlap_index = 0
        for mul in range(1, self.max_overlap + 1):
            overlap_ss = ssi[::mul]
            heigh = 0.5
            while heigh >= 0.4 : 
                peaks, _ = find_peaks(overlap_ss, height=heigh)                
                if len(peaks) > 2 :
                    diff_peaks = peaks[1:] - peaks[:-1]
                    if (abs((diff_peaks) - 2*bpm_peak) <= 2).all() and mul ==1:
                        bpm_peak = bpm_peak *2
                    break
                else:
                    heigh -= 0.1
            if not peaks.any():
                continue        
            diff = np.abs(peaks - bpm_peak)
            min_index = np.argmin(diff)
            min_diff = diff[min_index]
            if min_diff < 2:
                overlap_index += 1 << (mul - 1)
                peak_num += 1

        return overlap_index, peak_num, bpm_peak

    def overlap_index_2_weight(self, overlap_index):
        overlap_weight = np.zeros(self.max_overlap)
        if overlap_index == 0:
            overlap_weight[1:5] = 1        
        else:
            for mul in range(self.max_overlap):
                overlap_weight[mul] = (overlap_index >> mul) & 1
        return overlap_weight

    def get_overlap_list(self, confidence_level, bpm_spectrum):
        if self.USE_LOADED_OVERLAP_WEIGHT == 1:
            return self.overlap_weight
        elif self.USE_LOADED_OVERLAP_WEIGHT == 2:
            return self.overlap_weight_default

        ss = self.ss_denoise.T
        count_overlap_index =  np.zeros(2**self.max_overlap)
        overlap_index_selected = 0
        list_peak_num = []
        for i in range(len(ss)):#len(ss)
            # in static and idle and confidence_level 1    
            overlap_index, peak_num, bpm_peak = self.get_overlap_index_mads(ss[i], bpm_spectrum[i])        
            list_peak_num.append(peak_num)
            if confidence_level[i] and (self.status[i] == 1 or self.status[i] == 3):
                bpm_peak = 0
                if self.USE_HIGH_HR_STRATEGY :
                    overlap_index, peak_num, bpm_peak = self.get_overlap_index_mads(ss[i], bpm_spectrum[i])
                else:
                    overlap_index, peak_num = self.get_overlap_index(ss[i], bpm_spectrum[i])
                if peak_num>=4:
                    count_overlap_index[overlap_index] += 1
                elif peak_num > 2 and bpm_peak > 55:
                        count_overlap_index[overlap_index] += 1
                # elif peak_num >= 2 and bpm_peak < 35:
                #     count_overlap_index[overlap_index] += 1
                
                if count_overlap_index[overlap_index] >= 4:
                    overlap_index_selected = overlap_index
                    break
        if self.USE_HIGH_HR_STRATEGY and (overlap_index_selected <= 7 and overlap_index_selected > 0):
            self.bpm_search_upper  = int(self.fft_window_size*70/30) 
            self.bpm_overlap_upper = int(self.fft_window_size*70/30)
        overlap_weight = self.overlap_index_2_weight(overlap_index_selected)
        print('time to get overlap index:%d, overlap weight: [%d,%d,%d,%d,%d,%d,%d], overlap_index: %d' % (i, overlap_weight[0], overlap_weight[1], overlap_weight[2], overlap_weight[3], overlap_weight[4], overlap_weight[5], overlap_weight[6], overlap_index_selected))
        return overlap_weight

    def get_range_peak_idx(self, spectrum, lower, upper):
        result_bpm = []
        self.bpm_interval = np.zeros(spectrum.shape)
        self.bpm_interval[lower:upper] = np.copy(spectrum[lower:upper])
        self.normalization(self.bpm_interval)
        bpm_interval = self.bpm_interval.T

        for i in range(len(bpm_interval)):
            result_bpm.append(np.argmax(bpm_interval[i]))

        result_bpm = np.array(result_bpm)
        return result_bpm


    def get_bpm_final(self, bpm_overlap_data, lower, upper, status=np.array([])):
        result_bpm = []
        stable_index_list = []
        self.bpm_interval = np.zeros(bpm_overlap_data.shape)
        self.bpm_interval[lower:upper] = np.copy(bpm_overlap_data[lower:upper])
        self.normalization(self.bpm_interval)
        bpm_interval = self.bpm_interval.T

        for i in range(len(bpm_interval)):
            '''Slight body movement by mads'''
            if self.USE_SLIGHT_MOVE_STRATEGY == 1:
                if status[i] == 3 and len(stable_index_list) > 5:
                    truth_peak_index = np.median(np.copy(stable_index_list[(len(stable_index_list) + 1) % 2:]))
                    if abs(np.argmax(bpm_interval[i]) - truth_peak_index) > 10:
                        status[i] += 1
            '''extend serarching range for people with low HR by mads'''
            if self.try_range_extend == 1:
                if np.mean(stable_index_list) > upper - 5  or np.mean(stable_index_list) < lower + 5:
                    lower -= 5
                    upper += 5
                    self.bpm_interval = np.zeros(bpm_overlap_data.shape)
                    self.bpm_interval[lower:upper] = np.copy(bpm_overlap_data[lower:upper])
                    self.normalization(self.bpm_interval)
                    bpm_interval = self.bpm_interval.T
            if self.USE_TRAIN_FOR_DEEPBREATH == 1 and (status[i] == 2 or status[i] == 4):
                status[i] -= 1

            if len(status) == 0 or self.USE_BDT == 0:
                truth_peak_index = np.argmax(bpm_interval[i])


            elif self.confidence_level_svm2[i]==1:
                self.stable_index.append(i)
                truth_peak_index = int(np.argmax(self.ss_unnorm[:,i])/2)
                stable_index_list.append(truth_peak_index)
                # print('2:',stable_index_list)
                if len(stable_index_list) > 10: # TODO
                    stable_index_list.pop(0)
            elif self.confidence_level_svm1[i]==1:
                self.stable_index.append(i)
                truth_peak_index = int(np.argmax(self.ss_unnorm[:,i]))
                stable_index_list.append(truth_peak_index)
                # print('1:',stable_index_list)
                if len(stable_index_list) > 10: # TODO
                    stable_index_list.pop(0) 
                                        
            elif (status[i] == 1 or status[i] == 3) and self.confidence_level[i] == 1:
                self.stable_index.append(i)
                truth_peak_index = np.argmax(bpm_interval[i])
                stable_index_list.append(truth_peak_index)
                if len(stable_index_list) > 10: # TODO
                    stable_index_list.pop(0)
            elif i <= 6:
                truth_peak_index = np.argmax(bpm_interval[i])
            else:
                if len(stable_index_list) == 0:
                    truth_peak_index = np.argmax(bpm_interval[i])
                else:
                    if (self.confidence_level[i] == 1) and (self.confidence_level[i-1] == 1) :
                        
                        stable_index_list.append(truth_peak_index)
                        # print('4:',stable_index_list)
                        if len(stable_index_list) > 10: # TODO
                            stable_index_list.pop(0)

                    truth_peak_index = np.median(np.copy(stable_index_list[(len(stable_index_list) + 1) % 2:]))
                    height = 0.5
                    peaks, _ = find_peaks(bpm_interval[i], height=height)
                    peak_candidate_amp = bpm_interval[i][peaks]
                    # determind the strategy to get bpm within the status that is not stable
                    # add the strategy for status of 0 and -1 by mads
                    if status[i] == -1:
                        truth_peak_index = np.median(stable_index_list)
                    elif len(peaks) > 2:
                        truth_peak_index = self.decide_rule_bdt(peak_candidate_amp, peaks, truth_peak_index)
                    elif status[i] == 0:
                        truth_peak_index = np.median(stable_index_list)
                    else:
                        truth_peak_index = np.argmax(bpm_interval[i])

                if self.USE_NO_JUMP_BPM == 1: # fix PVT_2020-11-11-18-55        
                    if i>self.bpm_no_jump_range:
                        mm_bpm = np.median(result_bpm[-(self.bpm_no_jump_range-1):])
                        if (truth_peak_index - mm_bpm)>self.bpm_no_jump_range:
                            truth_peak_index = mm_bpm + 2
                        elif (truth_peak_index - mm_bpm) < -self.bpm_no_jump_range:
                            truth_peak_index = mm_bpm - 2
   
            if status[i] > 0 :
                last_status = status[i]
            result_bpm.append(truth_peak_index)

        result_bpm = np.array(result_bpm)
        result_bpm = result_bpm * 60 / self.fft_window_size + 1
        return result_bpm

    @staticmethod
    def get_peak_sorted(ss, height, dis):
        if dis>0:
            peaks_loc, peak_property = find_peaks(ss, height=height, distance=dis ) 
        else:
            peaks_loc, peak_property = find_peaks(ss, height=height ) 
        peaks_height = peak_property['peak_heights']
        bb = np.argsort(peaks_height)
        cc = bb[::-1]
        peaks_loc_sort = peaks_loc[cc]
        peaks_height_sort = peaks_height[cc]

        return peaks_loc_sort, peaks_height_sort, peaks_loc, peaks_height

    def get_peaks_array(self, bpm_overlap_data, dis):
        default_peak_len = 8
        peaks_locs_sort = np.zeros((bpm_overlap_data.shape[1],default_peak_len))
        peaks_amps_sort = np.zeros((bpm_overlap_data.shape[1],default_peak_len))

        for i in range(bpm_overlap_data.shape[1]):
            peaks_loc_sort, peaks_height_sort, peaks_loc, peaks_height = self.get_peak_sorted(bpm_overlap_data[:,i], 0.2, dis)
            if len(peaks_loc) > default_peak_len:
                peak_len = default_peak_len
            else:
                peak_len = len(peaks_loc)

            for i1 in range(peak_len): 
                peaks_locs_sort[i][i1] = peaks_loc_sort[i1]
                peaks_amps_sort[i][i1] = peaks_height_sort[i1]

        return peaks_locs_sort, peaks_amps_sort

    @staticmethod
    def dismantling(f_data, lower, upper):
        interval = np.copy(f_data[lower:upper])
        return np.array(interval)

    @staticmethod
    def get_rpm(rpm_interval_data, lower):
        rpm_result = np.argmax(rpm_interval_data, axis=1)
        rpm_result = (rpm_result + lower)
        return rpm_result

    def decide_rule_bdt(self, ss_candidate_amp, ss_candidate_index, truth_peak_index):
        peak_sep = np.abs(ss_candidate_index - truth_peak_index)
        peak_sep_decay = np.exp(-2 * (peak_sep*self.bdt_weight) ** 2)
        peak_amp = ss_candidate_amp ** 1.5
        truth_peak_index = ss_candidate_index[np.argmax(peak_sep_decay * peak_amp)]
        return truth_peak_index

    @staticmethod
    def exclude_abnormal_peak(acc_signal):
        down_acc_window = acc_signal
        down_acc_window = np.abs(down_acc_window - np.mean(down_acc_window))
        down_acc_window_peaks, _ = find_peaks(down_acc_window)
        if len(down_acc_window_peaks)!= 0:
            peaks_mean = np.mean(down_acc_window[down_acc_window_peaks])
        else:
            return []

        over_peaks = down_acc_window_peaks[down_acc_window[down_acc_window_peaks] > peaks_mean * 2]
        for max_peak in over_peaks:
            max_value = down_acc_window[max_peak]
            upper = max_peak + 1
            lower = max_peak - 1
            value = max_value
            while down_acc_window[upper] < value: # find upper valley location
                value = down_acc_window[upper]
                upper += 1
                if upper >= len(down_acc_window):
                    break

            value = max_value
            while down_acc_window[lower] < value: # fine lower valley location
                value = down_acc_window[lower]
                lower -= 1
                if lower <= 0:
                    break

            down_acc_window[int(lower):int(upper)] = down_acc_window[int(lower):int(upper)] * peaks_mean / max_value
        return down_acc_window

    def get_scenario_whole_log(self, filter_bcg_signal, filter_acc_signal, acc_org_peak_ratio):
        start = 0
        window = self.fs
        result = np.zeros(int((len(filter_acc_signal) + window-1) / window))
        down_acc_power = np.zeros(int((len(filter_acc_signal) + window-1) / window))
        bcg_mean_power = np.zeros(int((len(filter_acc_signal) + window-1) / window))

        search_bound = len(filter_acc_signal) - window * (self.ss_t_len-1)
        while start < search_bound:
            i = int(start / window)
            bcg_window = filter_bcg_signal[start: start + window * self.ss_t_len]
            acc_window = filter_acc_signal[start: start + window * self.ss_t_len]
            down_acc_power[i] = np.mean(self.exclude_abnormal_peak(acc_window))
            bcg_mean_power[i] = np.mean(np.abs(bcg_window)) - np.abs(np.mean(bcg_window))
            start += window

            curr_state = 0
            if down_acc_power[i] > self.DRIVING_THRESHOLD:
                if down_acc_power[i] < self.IDLE_THRESHOLD:
                    curr_state = 1
                else:
                    curr_state = 5
            elif down_acc_power[i] > self.IDLE_THRESHOLD:
                if acc_org_peak_ratio[i] == 1:
                    curr_state = 3
                else:
                    curr_state = 1                    
            else:
                curr_state = 1
            result[i + 2] = curr_state    

        result = self.moving_median(result, 2)
        acc_max_idx = np.argmax(down_acc_power)
        bcg_seq_max = bcg_mean_power[acc_max_idx]
        acc_seq_max = down_acc_power[acc_max_idx]
        if acc_seq_max == 0:
            acc_seq_max = 1
        if bcg_seq_max == 0:
            bcg_seq_max = 1        
        power_diff = bcg_mean_power /bcg_seq_max - down_acc_power / acc_seq_max
        power_diff[power_diff < 0] = 0

        motion_threshold = [0,0,0,0,0,0,0]
        for curr_state in [1,3,5]:
            motion_threshold[curr_state] = 1
            if np.sum(result == curr_state) != 0:
                motion_threshold[curr_state] = np.mean(power_diff[result == curr_state])
                if motion_threshold[curr_state] < self.MOVEMENT_THRESHOLD_LIMIT:
                    motion_threshold[curr_state] = self.MOVEMENT_THRESHOLD_LIMIT
            result[np.logical_and(result == curr_state, power_diff > (motion_threshold[curr_state] / 2))] += 1

        self.acc_power = down_acc_power
        self.bcg_power = bcg_mean_power
        half_t_len = round(self.ss_t_len/2)
        return np.array(result[half_t_len-1:-half_t_len])
        
    def get_scenario_clip_log(self, filter_bcg_signal, filter_acc_signal, acc_org_peak_ratio):
        start = 0
        window = self.fs
        len_data = int((len(filter_acc_signal) + window-1) / window)
        result = np.zeros(len_data)
        down_acc_power = np.zeros(len_data)
        bcg_mean_power = np.zeros(len_data)
        power_diff     = np.zeros(len_data)
        bcg_std_power  = np.zeros(len_data)
        buffer_for_truck_engine = []
        # loop 1
        while start < len(filter_acc_signal) - window * (self.ss_t_len - 1):
            i = int(start / window)
            half_ss_t = math.ceil(self.ss_t_len / 2) - 1
            if start < window * half_ss_t:
                idx_s = 0
            else:
                idx_s = start - (half_ss_t * window)
            
            if start > window * (len(filter_acc_signal) // window - self.ss_t_len // 2 - 1):
                idx_e = -1
            else:
                idx_e = start + (self.ss_t_len // 2 + 1) * window
            # consider the percision of status check, the window length could be 1sec and would not affect other function
            idx_s_s = np.max([(start - int(window/2)),0])
            idx_e_s = np.min([(start + int(window/2)), len(filter_acc_signal)-1]) 
            bcg_window_small = filter_bcg_signal[idx_s_s : idx_e_s]
            bcg_std_power[i] = np.nanstd(self.exclude_abnormal_peak(bcg_window_small)) if max(bcg_window_small) != min(bcg_window_small) else 0
        
            bcg_window = filter_bcg_signal[idx_s : idx_e]
            acc_window = filter_acc_signal[idx_s : idx_e]

            down_acc_power[i] = np.mean(self.exclude_abnormal_peak(acc_window)) 
            bcg_mean_power[i] = np.mean(self.exclude_abnormal_peak(bcg_window)) if len(self.exclude_abnormal_peak(bcg_window)) > 0 else 0#np.mean(np.abs(bcg_window)) - np.abs(np.mean(bcg_window))
            start += window

            curr_state = 0
            if acc_org_peak_ratio[i] == 1 :
                curr_state = 3
                if len(buffer_for_truck_engine) < 20:
                    buffer_for_truck_engine.append(down_acc_power[i])
                else:
                    if np.mean(buffer_for_truck_engine) > self.threshold_down_acc_truck and self.truck_or_not == 0:
                        self.truck_or_not = 1
                        print("truck mode")
                
                if self.USE_DETECT_TRUCK and self.truck_or_not:
                    if down_acc_power[i] > self.DRIVING_THRESHOLD * 10:
                        curr_state = 5
            else:
                if down_acc_power[i] > self.DRIVING_THRESHOLD:
                    curr_state = 5
                else:
                    if result[i-1] >= 5:
                        curr_state = 3
                    else:
                        curr_state = 1

            result[i] =  curr_state

            '''compute the similarity between acc and bcg to check vital sign exist by mads'''
            if self.USE_NEW_NO_VITAL_SIGN == 1:
                if len(acc_window) == 0 : # for case which there is no peak in the filter_acc_signal_window
                    self.similarity.append(0)
                else:
                    if max(acc_window) == min(acc_window): # for sepcial case of satasets without acc signal
                        acc_window[0] = 1
                    self.similarity.append(np.corrcoef(bcg_window, acc_window)[0,1]) # TODO correlation no need to normalize
                
        result = self.moving_mode_status(result, 2)

        if self.USE_CHECK_STATUS_AFTER_SS == 1:
            
            self.bcg_std_power = bcg_std_power           
            self.acc_power = down_acc_power
            self.bcg_power = bcg_mean_power
            return np.array(result[:-(self.ss_t_len - 1)])

        max_power_buffer = 0
        len_result = len(result)  
        max_power_buffer_serial  = np.zeros(len_result) 
        time_check = 0  
        time_after_body_up = -1  
        # loop 2 
        for i in range(len_result):
            idx_pre = np.max([i - (math.ceil(self.ss_t_len / 2) - 1),0])
            idx_end = np.min([i + (self.ss_t_len // 2 + 1), len_result]) 

            '''check the body up or down by mads'''    
            if self.USE_NEW_NO_VITAL_SIGN == 1: #2021-05-17-15-49_changepeople.log
                threshold_body_up = 0.3
                maxi_sec_body_up = 2
                mini_sec_body_up = 2
                # in the first 6 second, the threshold for check body could be decide #mads
                if i == 5:
                    self.compute_std_threshold_nobody(result, bcg_std_power)
                max_power = np.max(bcg_mean_power[idx_pre:idx_end])                
                if result[i] >= 5:
                    continue                 
                if (bcg_mean_power[i] < threshold_body_up*max_power or bcg_mean_power[i] > 2*max_power) and max_power!=0:
                    time_check +=1
                elif time_check >= mini_sec_body_up and (time_after_body_up <= maxi_sec_body_up or bcg_mean_power[i] > threshold_body_up*max_power_buffer):
                    time_check +=1
                else:
                    max_power_buffer = 0
                    time_check = 0

                max_power_buffer_serial[i] = max_power_buffer
                if time_check > 0: 
                    if bcg_mean_power[i] == max_power or max_power == 0:
                        time_after_body_up = 0
                    else:
                        time_after_body_up += 1
                else:
                    time_after_body_up = -1

                if bcg_std_power[i] < self.threshold_nobody_std and result[i] == 1:
                    result[i] = 0
                if bcg_std_power[i] == 0:
                    result[i] = -1
                if time_check >= mini_sec_body_up :
                    result[i] = -1
                    # result[i-1] = -1
                    max_power_buffer = max(max_power_buffer, max_power)


        
        if self.USE_STATE_MECHINE == 1 and self.USE_CHECK_STATUS_AFTER_SS == 0:
            result = self.moving_mode_status(result, 2)
            result = self.check_state_by_mechine(result) 
            

        # loop 3
        for i in range(len_result):
            motion_threshold = [0,0,0,0,0,0,0]
            idx_pre = np.max([i - (math.ceil(self.ss_t_len / 2) - 1),0])
            idx_end = np.min([i + (self.ss_t_len // 2 + 1), len_result]) 
            #======clip from last for loop 2========
            if i == 0 :
                for j in range(idx_end):
                    idx_pre_sub = np.max([j - (math.ceil(self.ss_t_len / 2) - 1),0])
                    idx_end_sub = np.min([j + (self.ss_t_len // 2 + 1), len_result]) 

                    acc_max_idx = idx_pre_sub + np.argmax(down_acc_power[idx_pre_sub: idx_end_sub])
                    bcg_seq_max = bcg_mean_power[acc_max_idx]
                    acc_seq_max = down_acc_power[acc_max_idx]
                    if bcg_seq_max!=0 and acc_seq_max!=0:
                        power_diff[j] = bcg_mean_power[j] / bcg_seq_max - down_acc_power[j] / acc_seq_max
                    if power_diff[j] < 0 : 
                        power_diff[j] = 0
            elif i < len_result:
                idx_pre_sub = np.max([(idx_end-1) - (math.ceil(self.ss_t_len / 2) - 1),0])
                idx_end_sub = np.min([(idx_end-1) + (self.ss_t_len // 2 + 1), len_result]) 
                acc_max_idx = idx_pre_sub + np.argmax(down_acc_power[idx_pre_sub: idx_end_sub])
                bcg_seq_max = bcg_mean_power[acc_max_idx]
                acc_seq_max = down_acc_power[acc_max_idx]
                if bcg_seq_max!=0 and acc_seq_max!=0:
                    power_diff[(idx_end-1)] = bcg_mean_power[(idx_end-1)] / bcg_seq_max - down_acc_power[(idx_end-1)] / acc_seq_max
                if power_diff[(idx_end-1)] < 0 : 
                    power_diff[(idx_end-1)] = 0

            #======================================
            for curr_state in [1,3,5]:
                motion_threshold[curr_state] =  1
                if np.sum(result[idx_pre: idx_end] == curr_state) != 0:
                    motion_threshold[curr_state] =  np.mean(power_diff[idx_pre: idx_end][result[idx_pre: idx_end] == curr_state])
                    if motion_threshold[curr_state] < self.MOVEMENT_THRESHOLD_LIMIT:
                        motion_threshold[curr_state] = self.MOVEMENT_THRESHOLD_LIMIT 
                if result[i] == curr_state and (power_diff[i] > (motion_threshold[curr_state] * self.scenario_clip_th_para)):
                    result[i] += 1

        
        # half_t_len = round(self.ss_t_len/2)
        return np.array(result[:-(self.ss_t_len - 1)])


    def check_normal_move(self):
        len_result = len(self.status)
        self.power_diff= np.zeros(len_result)
        result         = np.copy(self.status)
        down_acc_power = np.copy(self.acc_power)
        bcg_mean_power = np.copy(self.bcg_power)

        for i in range(len_result):
            motion_threshold = [0,0,0,0,0,0,0]
            idx_pre = np.max([i - (math.ceil(self.ss_t_len / 2) - 1),0])
            idx_end = np.min([i + (self.ss_t_len // 2 + 1), len_result]) 
            if i == 0 :
                for j in range(idx_end):
                    idx_pre_sub = np.max([j - (math.ceil(self.ss_t_len / 2) - 1),0])
                    idx_end_sub = np.min([j + (self.ss_t_len // 2 + 1), len_result]) 

                    acc_max_idx = idx_pre_sub + np.argmax(down_acc_power[idx_pre_sub: idx_end_sub])
                    bcg_seq_max = bcg_mean_power[acc_max_idx]
                    acc_seq_max = down_acc_power[acc_max_idx]
                    if bcg_seq_max!=0 and acc_seq_max!=0:
                        self.power_diff[j] = bcg_mean_power[j] / bcg_seq_max - down_acc_power[j] / acc_seq_max
                    if self.power_diff[j] < 0 : 
                        self.power_diff[j] = 0
            elif i < len_result:
                idx_pre_sub = np.max([(idx_end-1) - (math.ceil(self.ss_t_len / 2) - 1),0])
                idx_end_sub = np.min([(idx_end-1) + (self.ss_t_len // 2 + 1), len_result]) 
                acc_max_idx = idx_pre_sub + np.argmax(down_acc_power[idx_pre_sub: idx_end_sub])
                bcg_seq_max = bcg_mean_power[acc_max_idx]
                acc_seq_max = down_acc_power[acc_max_idx]
                if bcg_seq_max!=0 and acc_seq_max!=0:
                    self.power_diff[(idx_end-1)] = bcg_mean_power[(idx_end-1)] / bcg_seq_max - down_acc_power[(idx_end-1)] / acc_seq_max
                if self.power_diff[(idx_end-1)] < 0 : 
                    self.power_diff[(idx_end-1)] = 0

            for curr_state in [1,3,5]:
                motion_threshold[curr_state] =  1
                if np.sum(result[idx_pre: idx_end] == curr_state) != 0:
                    motion_threshold[curr_state] =  np.mean(self.power_diff[idx_pre: idx_end][result[idx_pre: idx_end] == curr_state])
                    if motion_threshold[curr_state] < self.MOVEMENT_THRESHOLD_LIMIT:
                        motion_threshold[curr_state] = self.MOVEMENT_THRESHOLD_LIMIT 
                if result[i] == curr_state and (self.power_diff[i] > (motion_threshold[curr_state] * self.scenario_clip_th_para)):
                    result[i] += 1
        return result


    def get_scenario(self, filter_bcg_signal, filter_acc_signal, acc_org_peak_ratio):
        
        if self.USE_WHOLE_LOG_SCENARIO == 0:
            result = self.get_scenario_clip_log(filter_bcg_signal, filter_acc_signal, acc_org_peak_ratio)
        else:
            result = self.get_scenario_whole_log(filter_bcg_signal, filter_acc_signal, acc_org_peak_ratio)

        return result
    
    def compute_std_threshold_nobody(self, status, std_serial):
        counts = [0] * 7
        for i in status[:6]:
            counts[int(i) + 1] += 1
        self.state_mechine = np.argmax(counts) - 1
        if self.state_mechine == 0 and max(std_serial[:6]) < 2000:
            self.threshold_nobody_std = max(std_serial[:6])
            print("threshold_nobody_std is ", int(self.threshold_nobody_std))               


    def check_state_by_mechine(self, status):        
    
        status[self.ss_t_len - 1] = self.state_mechine
        for i in range(self.ss_t_len , len(status)):
            if status[i] * self.state_mechine == 0 and status[i] + self.state_mechine > 0:
                if status[i - 1] == -1:
                    self.state_mechine = status[i]
                else:
                    status[i] = status[i - 1]
        return status


    def calc_variance(self, data):
        CALC_VARIANCE_LEVEL_AMOUNT =  5000 #12500
        CALC_VARIANCE_LEVEL_NUM = 15
        var = np.zeros(CALC_VARIANCE_LEVEL_NUM)
        temp = abs(data) / CALC_VARIANCE_LEVEL_AMOUNT
        for i in range(len(temp)):
            if temp[i] > CALC_VARIANCE_LEVEL_NUM - 1:
                var[CALC_VARIANCE_LEVEL_NUM - 1] +=1
            else:
                var[int(temp[i])] += 1
        return var


    def find_peak_local_height(self, spec, peak_idx):
        min_index = int(peak_idx)
        lower_min = spec[min_index]

        min_index -= 1
        while min_index >= 0 and spec[min_index] < lower_min:
            lower_min = spec[min_index]
            min_index -= 1

        min_index = int(peak_idx)
        upper_min = spec[min_index]
        min_index += 1
        while min_index <= self.fft_window_size*10 and spec[min_index] < upper_min:
            upper_min = spec[min_index]
            min_index += 1

        return spec[int(peak_idx)] * 2 - upper_min - lower_min

    def get_confidence_level(self, spec_data, sepc_denoised, peak_index, status):
        spec_data = spec_data.T
        sepc_denoised = sepc_denoised.T
        result = []
        list_peak_height = []
        
        
        for i in range(len(peak_index)):
            CL_threshold = self.reliable_threshold           
                
            peaks , _ = find_peaks(sepc_denoised[i], height = CL_threshold)
            peak_height = self.find_peak_local_height(spec_data[i], peak_index[i])
            overlap_index, peak_num = self.get_overlap_index(spec_data[i], peak_index[i] )
            list_peak_height.append(peak_height)
            if status[i] <= 4:
                if peak_height > CL_threshold:
                    if len(peaks) <= 1:
                        result.append(0)
                        status[i] = 0
                    else:
                        result.append(1)
                else: 
                    result.append(0)
            else:
                result.append( (overlap_index > 2) and (peak_height > CL_threshold))

        return np.array(result), np.array(list_peak_height)

    def get_confidence_level2(self, spec_data):
        spec_data = spec_data.T
        result2 = []
        result1 = []
        # print(str(len(spec_data[:,1])))       
        for i in range(len(spec_data[:,1])):
            input = self.convert_spec_feature(spec_data[i])          
            x_logs=np.array(input)
            reli_svm2 = 0
            reli_svm1 = 0

            if pd.DataFrame(x_logs).isnull().sum()[0]==0 and len(x_logs) == 10 :
                reli_svm2 = int(re_svm_2.predict([x_logs])[0])
                reli_svm1 = int(re_svm_1.predict([x_logs])[0])
                
            result1.append(reli_svm1)
            result2.append(reli_svm2)


        return np.array(result2),np.array(result1)
    def get_engine_noise(self, spec_data):
        spec_data_overlap = np.zeros((spec_data.shape[0], spec_data.shape[1]))
        overlap_weight = [1, 1, 1, 1]
        engine_search_low = self.bpm_engine_search_lower
        if self.USE_EGINE_DETECT_EXTEND == 1:
            engine_search_low = int(self.fft_window_size*3/self.stft_reduce_scale)
        for i in range(engine_search_low, self.bpm_engine_search_upper):
            for j in range(len(overlap_weight)):
                search_bound = self.fft_window_size*self.fs/2/self.stft_reduce_scale/self.decimate_scale
                if i*(j+1) < search_bound:
                    moving_max = np.amax(spec_data[i*(j+1): (i+1)*(j+1)], axis=0)
                    if j == 0 :
                        if self.USE_EGINE_DETECT_EXTEND == 1: # add by mads
                            moving_max[moving_max < self.threshold_engine_peak] = 0  # hint: it is better to modified the weight by seconds but 
                        else:                                 # in here it would cost so much runtime. it may be further 
                            continue                          # studyed before the porting on the C code

                    spec_data_overlap[i] += moving_max * overlap_weight[j]

        return spec_data_overlap

    @staticmethod
    def moving_median(ppi, half_range):
        ppi1 = np.copy(ppi)
        for i in range(half_range, len(ppi) - half_range):
            ppi1[i] = np.median(ppi[i - half_range:i + half_range+1])
        return ppi1

    @staticmethod
    def moving_mode_status(ppi, half_range):
        ppi1 = np.copy(ppi)
        for i in range(half_range, len(ppi) - half_range):
            counts = [0] * 8
            status = ppi[i - half_range:i + half_range + 1]
            for j in status:
                counts[int(j) + 1] += 1
            ppi1[i] = np.argmax(counts) - 1
        return ppi1

    @staticmethod
    def _post_process_bpm(ppi, status):
        ppi1 = np.copy(ppi)
        for i in range(2, len(ppi) - 2):
            if status[i] < 5:
                ppi1[i] = np.median(ppi[i - 2:i + 3])
            else:
                ppi1[i] = np.mean(ppi[i - 2:i + 3])
        return ppi1

    @staticmethod
    def _pre_process(data, g_sensor_data_x, g_sensor_data_y=[], g_sensor_data_z=[]):
        data = np.array(data)
        data = data - data[0]
        g_sensor_data = g_sensor_data_x

        if len(g_sensor_data_y) != 0:
            B = np.array([(g_sensor_data_x), (g_sensor_data_y), (g_sensor_data_z)])
            for i in range(1, len(B)):
                for j in range(3):
                    if B[i, j] > 10000:
                        B[i, j] = B[i-1, j]
                g_sensor_data[i] = (B[i, 0] ** 2 + B[i, 1] ** 2 + B[i, 2] ** 2) ** 0.5

        else:
            # spike remove only in python code.
            for i in range(1, len(g_sensor_data)):
                if g_sensor_data[i] > 10000:
                    g_sensor_data[i] = g_sensor_data[i-1]

        
        g_sensor_data = np.array(g_sensor_data)
        g_sensor_data = g_sensor_data - g_sensor_data[0]
        return data, g_sensor_data

    @staticmethod
    def cos_sim(vector_a, vector_b):
        vector_a = vector_a.copy() - 0.5
        vector_b = vector_b.copy() - 0.5
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom if denom != 0 else -1
        sim = 0.5 + 0.5 * cos
        return sim

    @staticmethod
    def FIR_filter(fs, lowcut, highcut, data, numtaps=100):
        if isinstance(data, (np.ndarray, list)):
            if fs is None:
                raise ValueError("sampling frequency must be specified!")
            fir_tap = numtaps
            bpm_b = firwin(fir_tap, [lowcut, highcut], pass_zero=False, fs=fs)
            bpm_a = 1.0
            ripple_data = lfilter(bpm_b, bpm_a, data)
            return np.array(ripple_data)
        else:
            raise TypeError(
                "Unknown data type {} to filter.".format(str(type(data))))

    def time_fusion(self, bcg_data, acc_data):
        bcg_data = self.FIR_filter(self.fs, 1, 3, bcg_data)
        acc_data = self.FIR_filter(self.fs, 1, 3, acc_data)

        self.filter_time_bpm_data = bcg_data
        self.filter_time_g_sensor_data = acc_data
        start = 0
        window = 64
        acc_data = np.concatenate((np.array([0, 0, 0, 0, 0]), acc_data), axis=0)
        acc_data = np.concatenate((acc_data, np.array([0, 0, 0, 0, 0])), axis=0)
        result = []
        while start < len(bcg_data):
            bcg_window = bcg_data[start: start + window]
            max_corrcoef = -1

            for i in range(10):
                acc_window = acc_data[start + i: start + i + len(bcg_window)]
                corrcoef = np.corrcoef(bcg_window, acc_window)[0][1]
                if corrcoef > max_corrcoef:
                    offset = i
                    max_corrcoef = corrcoef
            if max_corrcoef < 0.7:
                offset = -1
                result.extend(bcg_window.tolist())
            else:
                acc_window = acc_data[start + offset: start + offset + len(bcg_window)]
                acc_window = np.array(acc_window)
                bcg_window = np.array(bcg_window)
                count = 0
                slice_point = 0
                for i in range(len(bcg_window) - 1):
                    if (int(bcg_window[i]) & int(bcg_window[i + 1])) < 0:
                        count += 1
                        if count == 2:
                            acc_slice = acc_window[slice_point:i]
                            bcg_slice = bcg_window[slice_point:i]
                            if max(acc_slice) == min(acc_slice):
                                scale = 1
                            else:
                                scale = (max(bcg_slice) - min(bcg_slice)) / (max(acc_slice) - min(acc_slice))
                            acc_slice = acc_slice * scale
                            offset = max(bcg_slice) - max(acc_slice)
                            acc_slice = acc_slice + offset
                            bcg_slice = bcg_slice - acc_slice
                            result.extend(bcg_slice.tolist())
                            count = 0
                            slice_point = i

                acc_slice = acc_window[slice_point:]
                bcg_slice = bcg_window[slice_point:]
                if max(acc_slice) == min(acc_slice):
                    scale = 1
                else:
                    scale = (max(bcg_slice) - min(bcg_slice)) / (max(acc_slice) - min(acc_slice))

                acc_slice = acc_slice * scale
                offset = max(bcg_slice) - max(acc_slice)
                acc_slice = acc_slice + offset
                bcg_slice = bcg_slice - acc_slice
                result.extend(bcg_slice.tolist())

            start = start + self.fs

        result = savgol_filter(result, 21, 7)

        return result

    def clip_data(self, input_signal):
        output_signal = input_signal
        clip_range = 20.0
        clip_reduce = 2.0
        clip_th = max(input_signal) / clip_range
        len_p = len(input_signal)
        for i in range(len_p):
            p_value = input_signal[i]
            if abs(p_value) > clip_th:
                amp_d = ((abs(p_value) - clip_th) / clip_th / clip_reduce + 1.0)
                output_signal[i] = input_signal[i] / amp_d
        return output_signal

    def filtering_signal(self, input_signal, cutoff_freq_low, cutoff_freq_high):
        if self.USE_FILTER == 0: # no filter
            output_signal = input_signal
        elif self.USE_FILTER == 1: # FIR filter
            output_signal = self.FIR_filter(self.fs,cutoff_freq_low, cutoff_freq_high, input_signal)
        elif self.USE_FILTER == 2: # IIR filter
            bpm_b, bpm_a = filter.butter_bandpass(cutoff_freq_low, cutoff_freq_high, self.fs, self.bpm_filter_order)
            output_signal = lfilter(bpm_b, bpm_a, input_signal)
        return output_signal

    def get_time_fusion_spectrum(self, ss_denoise,p_sensor_data, g_sensor_data):
        self.time_fusion_data = self.time_fusion(p_sensor_data, g_sensor_data)
        _, _, s = scipy.signal.stft(self.time_fusion_data, window='hamming', nperseg=self.fs * 5, noverlap=self.fs * 4, nfft=self.fs * self.fft_window_size, boundary=None)
        self.tss = np.abs(s[:,1:])
        self.normalization(self.tss)
        ss_denoise[0:100, self.status > self.als_threshold] = self.tss[0:100, self.status > self.als_threshold]
        return ss_denoise

    def get_golden_pickle_data(self,filename):
        with open(filename,'rb') as f:
            golden_data = pickle.load(f)
        return golden_data

    def get_golden_harmonic_spectrum(self, golden_data, spec_len):
        golden_harmonic_spectrum = np.zeros((spec_len,len(golden_data)))
        for i in range(len(golden_data)):
            dd = golden_data[i]
            golden_harmonic_spectrum[int(dd)  ,i] = 1
            golden_harmonic_spectrum[int(dd*2),i] = 1
            golden_harmonic_spectrum[int(dd*3),i] = 1
            golden_harmonic_spectrum[int(dd*4),i] = 1
            golden_harmonic_spectrum[int(dd*5),i] = 1
            golden_harmonic_spectrum[int(dd*6),i] = 1
        return golden_harmonic_spectrum 

    def update_status_other(self, ss_denoise):
        result = np.copy(self.status)
        len_result = len(result)        
        max_power_buffer = 0                    
        threshold_body_up = 0.3
        threshold_max_ss = 0.3
        maxi_sec_body_up = 2
        mini_sec_body_up = 2        
        time_check = 0  
        time_after_body_up = -1  
        max_ss_denoise_list = np.max(ss_denoise[int(self.fft_window_size*2):int(self.fft_window_size*7),:],0) 
        for i in range(len_result):
            max_ss_denoise = max_ss_denoise_list[i-1] if i > 0 else 0
            idx_pre = np.max([i - (math.ceil(self.ss_t_len / 2) - 1),0])
            idx_end = np.min([i + (self.ss_t_len // 2 + 1), len_result])             
            '''check the body up or down by mads'''    

            # in the first 6 second, the threshold for check body could be decide #mads
            if i == 6:
                self.compute_std_threshold_nobody(result, self.bcg_std_power)
            max_power = np.max(self.bcg_power[idx_pre:idx_end])                
            if result[i] >= 5:
                continue                 
            if (self.bcg_power[i] < threshold_body_up*max_power or self.bcg_power[i] > 2*max_power) and max_power!=0:
                time_check +=1
            elif time_check >= mini_sec_body_up and (time_after_body_up <= maxi_sec_body_up or self.bcg_power[i] > threshold_body_up*max_power_buffer):
                time_check +=1
            elif max_power != 0 :
                max_power_buffer = 0
                time_check = 0

            if time_check > 0: 
                if self.bcg_power[i] == max_power:
                    time_after_body_up = 0
                else:
                    time_after_body_up += 1
            else:
                time_after_body_up = -1

            if self.bcg_std_power[i] < self.threshold_nobody_std and result[i] <= 3:
                result[i] = 0
            if self.bcg_std_power[i] == 0: 
                result[i] = -1
            if time_check >= mini_sec_body_up :
                result[i] = -1
                max_power_buffer = max(max_power_buffer, max_power)
            if self.status[i] == 3 and max_ss_denoise < threshold_max_ss:
                result[i] = 0

        result = self.moving_mode_status(result, 2)
        result = self.check_state_by_mechine(result)
        return result

    
    def update_status(self):
        threshold_peak_height_static = 0.55
        threshold_peak_height_idle = 0.3
        threshold_body_up_down = 0.2
        new_status = np.copy(self.status)
        mean_spec_peak_height = 1
        nss_sum = np.sum(np.abs(self.nss[1:] - self.nss[:-1]), axis=0)
        rescale_factor = int(self.fft_window_size/self.decimate_scale/2) 
        for i in range(len(self.status)):
            novital = 0 
            nss_sum_threshold_body_up =  rescale_factor * 1.8
            nss_sum_threshold_no_vital = rescale_factor / 3
            if i > 3:
                mean_spec_peak_height = np.mean(self.spec_peak_height[i-3: i+1])
                if np.mean(self.status[i-3 : i+1]) > 2:
                    nss_sum_threshold_body_up = rescale_factor * 32 
                    nss_sum_threshold_no_vital = rescale_factor / 0.75

            #double check the big move happend
            if nss_sum[i] > nss_sum_threshold_body_up and self.similarity[i] < threshold_body_up_down:
                if i > 2:
                    if new_status[i-1] == -1:
                        new_status[i] = -1
            elif self.status[i] == -1 and self.status[i-1] != -1:
                new_status[i] = 2 if nss_sum_threshold_body_up >= 40 else 4

            # check if the vital signal exist
            if self.status[i] <= 2: 
                if mean_spec_peak_height < threshold_peak_height_static:
                    if nss_sum[i] < 5 :
                        new_status[i] = min(0, self.status[i])  
                    elif nss_sum[i] < nss_sum_threshold_body_up : 
                        new_status[i] = min(-1, self.status[i])                  
            elif self.status[i] <= 4  and (nss_sum[i] > nss_sum_threshold_no_vital  or mean_spec_peak_height < threshold_peak_height_idle):
                new_status[i] = min(0, self.status[i])                

            if new_status[i] > 0 and (new_status[i-1] <= 0 and new_status[i-2] > 0):
                new_status[i-1] = self.status[i-1]
        
        new_status = self.moving_median(new_status, 1)
        

        return new_status

    @staticmethod
    def down_sample(in_data, scale):
        return in_data[range(0,len(in_data),scale)]
    
    def get_scenario_knn(self, pressure_data, acc_x, acc_y, acc_z):
        data = pd.concat([pd.DataFrame(pressure_data), pd.DataFrame(acc_x), pd.DataFrame(
            acc_y), pd.DataFrame(acc_z)], axis=1)
        data.columns = (['pressure_data', 'acc_x','acc_y', 'acc_z'])

        aa = data
        gap = pd.DataFrame([])
        x_sd = pd.DataFrame([])
        y_sd = pd.DataFrame([])
        z_sd = pd.DataFrame([])
        for i in range(int(len(aa)/64)):
            gap = gap.append([max(aa['pressure_data'][i*64:(i+1)*64-1]) -
                             min(aa['pressure_data'][i*64:(i+1)*64-1])])
            x_sd = x_sd.append([np.std(aa['acc_x'][i*64:(i+1)*64-1])])
            y_sd = y_sd.append([np.std(aa['acc_y'][i*64:(i+1)*64-1])])
            z_sd = z_sd.append([np.std(aa['acc_z'][i*64:(i+1)*64-1])])

        data = pd.concat([x_sd, y_sd, z_sd, gap], axis=1)
        data.columns = (['x_sd', 'y_sd', 'z_sd', 'gap'])

        pp = status_svm.predict(data.iloc[:, [0, 1, 2]])
        data['p3c'] = pp

        for i in range(len(pp)):
            if pp[i] == 2:
                pp[i] = 3
            elif pp[i] == 3:
                pp[i] = 5

        return pp

    
    def move(self):

        move = np.zeros(len(self.g_nor))
        for i in range(len(self.g_nor)):
            if np.abs(self.g_nor[i]-self.p_nor[i])>3.88:
                move[i]=1
            else:
                move[i]=0
        move_min=np.zeros(len(self.status))
        for i in range(len(self.status)):
            move_min[i]=np.median(move[i*64:(i+1)*64])
        
        for i in range(len(self.status)):
            if move_min[i]==1:
                self.status[i]+=1
        return self.status
    
    def convert_spec_feature(self, ss_unnorm):
        ss_unnorm = ss_unnorm[:200]
        A = np.copy(ss_unnorm)
        A[:25] = 0
        f = np.arange(1,len(A)+1)*60/self.fft_window_size
        if sum(A) == 0:
            return []
        
        feature_1=   np.mean(A)
        feature_2=   sum((A-feature_1)**2) / len(A)-1
        #feature_3=   sum((A-feature_1)**3) /(len(A)*pow(feature_2, 1/6))
        feature_4=   sum((A-feature_1)**4) /(len(A)*pow(feature_2, 2))
        feature_5=   sum(f*A) / sum(A)
        feature_6=   np.sqrt(sum(pow((f-feature_5),2)*A)/len(A))
        feature_7=   np.sqrt(sum(pow(f,2)*A)/sum(A))
        feature_8=   np.sqrt((sum(pow(f,4)*A))/(sum(pow(f,2)*A)))
        feature_9=   sum(pow(f,2)*A) / np.sqrt(sum(A)*sum(pow(f,4)*A))
        feature_10=  feature_6/feature_5
        feature_11=  (sum(pow(f-feature_5,3)*A)) / (len(A)*pow(feature_6,3))
        feature_12=  (sum(pow(f-feature_5,4)*A)) / (len(A)*pow(feature_6,4))
        #feature_13=  (sum(pow(f-feature_5,1/2)*A)) / (len(A)*pow(feature_6,1/2))
        feature_14=  np.sqrt((sum(pow(f-feature_5,2)*A))/sum(A))
        
        x_log=[#feature_1,
        feature_2,
        #feature_3,
        feature_4,
        feature_5,
        # feature_6,
        feature_7,
        feature_8,
        feature_9,
        feature_10,
        feature_11,
        feature_12,
        #feature_13,
        feature_14
        ]
        x_logs=np.array(x_log)
        return x_logs
        


    def get_HRV(self, HR, status):

        bpm = []
        tot_sec = len(HR)
        for i in range(len(HR)):
            if status[i] > 0 and status[i] % 2 == 1 and (HR[i] > self.bpm_search_lower *2 and HR[i] < self.bpm_search_upper *2 ):
                bpm.append(HR[i])
            
        if len(bpm) > tot_sec * 0.7:
            rri = (60/np.array(bpm))  * 1000
            mean_HR = np.mean(bpm)  
            SDNN = np.std(rri)
            RMSSD = np.sqrt(np.mean((rri[1:] - rri[:-1])**2))  
        else: 
            mean_HR = -1
            SDNN = -1
            RMSSD = -1
        
        return mean_HR, SDNN, RMSSD
        
    def get_vigor(self, mean_HR_n, SDNN_n, RMSSD_n):
        vigor = self.vigor_parameter[-1]
        vigor += mean_HR_n * self.vigor_parameter[0]
        vigor += SDNN_n * self.vigor_parameter[1]
        vigor += RMSSD_n * self.vigor_parameter[2]
        if vigor < 1: 
            vigor = 1
        elif vigor < 0:
            vigor = 0 
        return vigor


        


    def get_heart_rate(self, data, g_sensor_x, g_sensor_y, g_sensor_z):
        if self.USE_ACC_FEATURE == 0:
            g_sensor_data = (g_sensor_x ** 2 + g_sensor_y ** 2 + g_sensor_z ** 2) ** 0.5
        elif self.USE_ACC_FEATURE == 1:
            temp = np.array([g_sensor_x, g_sensor_y, g_sensor_z])
            # check the oritation of g sensor
            max_index = np.argmax(abs(np.mean(temp, axis=1)))
            # use the axis with maximun value, and fix the oritaiton
            g_sensor_data = temp[max_index] * np.sign(np.mean(temp, axis=1)[max_index])
        
        self.g_nor=(g_sensor_data-np.mean(g_sensor_data))/np.std(g_sensor_data)
        self.p_nor=(data-np.mean(data))/np.std(data)
        self.p_sensor_data, self.g_sensor_data = self._pre_process(data, g_sensor_x, g_sensor_y, g_sensor_z)

        # filtering
        self.filter_p_sensor_data = self.filtering_signal(self.p_sensor_data, self.bpm_p_cutoff_freq[0], self.bpm_g_cutoff_freq[1]) # TODO USE_REAL_DECIMATE
        self.filter_g_sensor_data = self.filtering_signal(self.g_sensor_data, self.bpm_g_cutoff_freq[0], self.bpm_g_cutoff_freq[1])

        if self.USE_EXTEND_FFT_RESOLUTION == 1 or self.USE_REAL_DECIMATE == 1:
            self.decimate_scale = 2
            self.fs = int(self.fs/self.decimate_scale)
        else:
            self.decimate_scale = 1

        # STFT (short time Fourier transform)
        _, _, s  = scipy.signal.stft(self.down_sample(self.filter_p_sensor_data,self.decimate_scale), window='hamming', nperseg=int(self.fs * self.ss_t_len), noverlap=int(self.fs * (self.ss_t_len - 1)), nfft=self.fs * self.fft_window_size, boundary=None)
        _, _, ns = scipy.signal.stft(self.down_sample(self.filter_g_sensor_data,self.decimate_scale), window='hamming', nperseg=int(self.fs * self.ss_t_len), noverlap=int(self.fs * (self.ss_t_len - 1)), nfft=self.fs * self.fft_window_size, boundary=None)

        #_, _, s  = scipy.signal.stft(self.filter_p_sensor_data, window='hamming', nperseg=int(self.fs * self.ss_t_len), noverlap=int(self.fs * (self.ss_t_len - 1)), nfft=self.fs * self.fft_window_size, boundary=None)
        #_, _, ns = scipy.signal.stft(self.filter_g_sensor_data, window='hamming', nperseg=int(self.fs * self.ss_t_len), noverlap=int(self.fs * (self.ss_t_len - 1)), nfft=self.fs * self.fft_window_size, boundary=None)

        if self.USE_REAL_DECIMATE == 1:
            self.filter_p_sensor_data = self.down_sample(self.filter_p_sensor_data,self.decimate_scale)
            self.filter_g_sensor_data = self.down_sample(self.filter_g_sensor_data,self.decimate_scale)
            self.decimate_scale = 1
        elif self.USE_EXTEND_FFT_RESOLUTION == 1:
            self.fs = int(self.fs*self.decimate_scale)
                    
        self.ss  = np.abs(s [0:self.fs *self.fft_window_size//self.decimate_scale//2,])
        self.nss = np.abs(ns[0:self.fs *self.fft_window_size//self.decimate_scale//2,])
       
        
        self.ss_unnorm  = np.abs(s [0:self.fs *self.fft_window_size//self.decimate_scale//2,])
        self.nss_unnorm = np.abs(ns[0:self.fs *self.fft_window_size//self.decimate_scale//2,])

        self.golden_harmonic_spectrum = self.get_golden_harmonic_spectrum(self.golden_bpm, self.ss.shape[0])

        # get engine noise
        self.normalization(self.nss)
        self.nss_clip = np.copy(self.nss)
        self.engine_noise = self.get_engine_noise(self.nss)
        self.normalization(self.engine_noise)

        # get scenario and status
        mean_nss = np.mean(self.nss[int(self.fft_window_size*7):,:],0)
        mean_nss[np.where(mean_nss<0.02)] = 0.02 

        self.acc_org_peak_ratio = (np.max(self.nss[int(self.fft_window_size*7/self.stft_reduce_scale):,:],0)/mean_nss)

        self.status = self.get_scenario(self.filter_p_sensor_data, self.filter_g_sensor_data, (self.acc_org_peak_ratio > 6))
        if self.USE_KNN_SCENARIO == 1  :
            status = self.get_scenario_knn(data, g_sensor_x, g_sensor_y, g_sensor_z)
            self.status = status[:len(self.status)]
        # Spectrum subtraction
        self.reserve_freq_band(self.ss,  self.reserved_freq_lower_bound, self.reserved_freq_upper_bound)
        self.reserve_freq_band(self.nss, self.reserved_freq_lower_bound, self.reserved_freq_upper_bound)
        self.normalization(self.ss)
        self.normalization(self.nss)

        self.ss_clip = np.copy(self.ss)  
        self.nss_clip = np.copy(self.nss)

        self.get_trans_func(np.copy(self.ss), np.copy(self.nss), self.status)
        self.ss_denoise, self.ss_status = self.get_ss(np.copy(self.ss), np.copy(self.nss), self.status, self.engine_noise)
        self.normalization(self.ss_denoise)
        '''double check the body up-or-down and check the no-vital by mads'''
        if self.USE_NEW_NO_VITAL_SIGN == 1 and self.USE_CHECK_STATUS_AFTER_SS == 1:
           self.status = self.update_status_other(self.ss_denoise) 
        self.status = self.check_normal_move()
            # self.status = self.move()
        if self.USE_TIME_FUSION:
            self.ss_denoise = self.get_time_fusion_spectrum(self.ss_denoise, self.p_sensor_data, self.g_sensor_data)
        
        # get overlaped spectrum and candidate peaks
        #self.ss_denoise_overlaped = self.get_overlap_spectrum(self.ss_denoise, self.overlap_weight)
        self.ss_denoise_overlaped = self.get_overlap_spectrum(np.copy(self.ss_denoise), np.array([1,1,1,1,1,1,1]))
        self.ss_denoise_overlaped = self.compensate_overlap_spectrum(np.copy(self.ss_denoise_overlaped), np.array([1,1,1,1,1,1,1]))
        self.normalization(self.ss_denoise_overlaped)
        self.bpm_idx = self.get_range_peak_idx(np.copy(self.ss_denoise_overlaped), self.bpm_search_lower, self.bpm_search_upper)

        # run iteration to get best overlap weighting and bpm results
        self.confidence_level, self.spec_peak_height = self.get_confidence_level(np.copy(self.ss_denoise_overlaped),np.copy(self.ss_denoise) , self.bpm_idx, self.status)
        if self.USE_SVM_reliability == 1 :
            self.confidence_level_svm2,self.confidence_level_svm1=self.get_confidence_level2(np.copy(self.ss_unnorm))
        else:
            self.confidence_level_svm1 = [0] * len(self.status)
            self.confidence_level_svm2 = [0] * len(self.status)
        self.overlap_weight = self.get_overlap_list(self.confidence_level, self.bpm_idx)
        self.ss_denoise_overlaped = self.get_overlap_spectrum(np.copy(self.ss_denoise), self.overlap_weight)
        self.ss_denoise_overlaped = self.compensate_overlap_spectrum(np.copy(self.ss_denoise_overlaped), self.overlap_weight)
        self.normalization(self.ss_denoise_overlaped)
        self.bpm_pre = self.get_bpm_final(np.copy(self.ss_denoise_overlaped), self.bpm_search_lower, self.bpm_search_upper, self.status)
        self.peaks_locs, self.peaks_amps = self.get_peaks_array(self.ss_denoise, int(self.fft_window_size*0.833))
        self.peaks_locs_o, self.peaks_amps_o = self.get_peaks_array(self.ss_denoise_overlaped, 0)

        # heart rate post processing
        if self.USE_BPM_POST_PROCESSING and self.USE_TRAIN_FOR_DEEPBREATH == 0:
            self.bpm = self._post_process_bpm(self.bpm_pre, self.status)
        else:
            self.bpm = self.bpm_pre

        if self.USE_REAL_DECIMATE == 1:
            self.fs = self.fs*2

        # FEWS system : vigor computing 
        if self.USE_VIGOR_COMPUTE : 
            vigor = 0
            for i in range(len(self.bpm)):
                HR = self.bpm[i]
                STATUS = self.status[i]
                self.HR_serial_HRV.append(HR)
                self.status_serial_HRV.append(STATUS)
                if len(self.HR_serial_HRV) == self.HRV_t_len:
                    MEAN_HR, SDNN, RMSSD =  self.get_HRV(self.HR_serial_HRV, self.status_serial_HRV)
                    
                    if self.vigor_learning_progress < 1:
                        if MEAN_HR > 0 :
                            self.HRV_reference_vigor.append([MEAN_HR, SDNN, RMSSD])
                            if len(self.HRV_reference_vigor) >= self.number_learning_vigor:
                                self.vigor_learning_progress = 1
                            else:
                                self.vigor_learning_progress = len(self.HRV_reference_vigor) / self.number_learning_vigor
                    else:
                        reference_HRV = np.mean(self.HRV_reference_vigor, axis = 0)
                        MEAN_HR_N = MEAN_HR / reference_HRV[0]
                        SDNN_N = SDNN / reference_HRV[1]
                        RMSSD_N = RMSSD / reference_HRV[2]
                        vigor = self.get_vigor(MEAN_HR_N, SDNN_N, RMSSD_N)
                    
                    self.HR_serial_HRV = self.HR_serial_HRV[self.HRV_t_interval :]
                    self.status_serial_HRV = self.status_serial_HRV[self.HRV_t_interval :]

                self.FEWS_vigor.append(vigor)
                
                if self.USE_FATIGUE_ALARM and self.vigor_learning_progress >= 1:
                    isfatigue = vigor > 0.5
                    fatigue_condition = 0
                    if isfatigue:
                        alarm_count += 1
                        if alarm_count >= self.number_collect_fatigue :
                            fatigue_condition = 1                                      
                    else:
                        alarm_count = 0
                    self.FEWS_alarm.append(fatigue_condition)


    #############################################################################################
    def main_func(self, data, g_sensor_x, g_sensor_y, g_sensor_z):
        self.get_heart_rate(data, g_sensor_x, g_sensor_y, g_sensor_z)
        return 0