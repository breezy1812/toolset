import numpy as np
import scipy
import math
from scipy.interpolate import interp1d
from . import filter
from scipy.signal import find_peaks, lfilter, peak_prominences,firwin, filtfilt, hilbert
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import pylab as plt

def normalize_peak(data):
    max_data = np.max(data)
    min_data = np.min(data)
    rang_data = max_data - min_data
    data_n = (data - min_data) * rang_data
    return data_n

def normalize_area(data):
    data_sum = np.sum(data)
    data_n = data/data_sum
    return data_n

class Alg_freq_domain():
    def __init__(self, fs=64, fft_window_size=30):
        self.fft_window_size = fft_window_size
        self.fs = fs
        # =========Parameters==========
        self.max_overlap = 7
        self.iteration_times = 1
        self.overlap_weight = np.array([1, 1, 1, 1, 1, 1, 1])
        self.bdt_weight = 0.4
        self.als_threshold = 4
        self.engine_threshold = 2
        self.reliable_threshold = 0.75
        self.reserved_freq_lower_bound = int(self.fft_window_size / 3)    # 10
        self.reserved_freq_upper_bound = int(self.fft_window_size * 10)   # 300
        self.NOBODY_THRESHOLD = 5000
        self.SITTING_THRESHOLD = 3000000
        self.DRIVING_THRESHOLD = 10
        self.IDLE_THRESHOLD = 0.8
        self.ss_t_len = 6  # default:5 
        # ==========Flags===============
        # 0: No Spectrum subtraction, 1: ALS, 2: Direct ss, 3: spec_sub_trans, 4: spec_sub_old
        self.USE_SPECTRUM_SUBTRACTION = 4
        # 0: No Filter, 1: FIR filter, 2: IIR filter
        self.USE_FILTER = 1 
        self.USE_TIME_FUSION = 0
        self.USE_MANUALLY_WEIGHT = 0
        self.USE_BDT = 1
        self.USE_ENGINE_DENOISE = 1
        self.USE_TRANSFER_FUNC = 1
        self.USE_CLIP = 0 
        self.USE_BPM_POST_PROCESSING = 1
        self.USE_NO_JUMP_BPM = 1
        self.USE_GET_RPM = 0
        # ==========Filter==============
        self.bpm_filter_order = 4
        self.bpm_p_cutoff_freq = [1.6, 10]
        self.bpm_g_cutoff_freq = [1.6, 31.5]
        self.rpm_filter_order = 2
        self.rpm_cutoff_freq = [0.05, 1]
        self.bpm_search_lower = 25
        self.bpm_search_upper = 55
        # ==============================
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
        self.engine_peaks = []
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
        self.bpm_no_jump_range = 10

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
    def reserve_freq_band(SS, lower_bound, upper_bound):
        SS[0:lower_bound] = 0
        SS[upper_bound:] = 0

    def spec_sub_direct(self, ss, nss):
        result = np.zeros(ss.shape[0])
        heart_beat_range = [30, 300]
        sss = normalize_area(ss[int(heart_beat_range[0]):int(heart_beat_range[1])]) - normalize_area((nss[int(heart_beat_range[0]):int(heart_beat_range[1])]**1.5))
        result[heart_beat_range[0]:heart_beat_range[1]] = sss        
        result[np.where(result<0)] = 0        
        result = normalize_peak(result)       
        
        return result

    @staticmethod
    def spec_sub_trans(ss, nss):
        heart_beat_range = [30, 300]
        result = np.zeros(ss.shape[0])

        diff_ss = np.around((ss[1:] - ss[:-1]) * 30)
        diff_nss = np.around((nss[1:] - nss[:-1]) * 30)
        spec_subtract = np.logical_or(diff_ss != diff_nss, np.logical_and(diff_ss == 0, diff_nss == 0))
        count = 0
        spec_subtract = np.append(spec_subtract, [0, 0])
        for i in range(len(spec_subtract) - 1):
            if spec_subtract[i]:
                count += 1
                result[i + int(heart_beat_range[0] + 1)] = count
                if spec_subtract[i + count] == 0:
                    count -= 2
                elif spec_subtract[i + count + 1] == 0:
                    count -= 1
            else:
                count = 0
        return result

    @staticmethod
    def spec_sub_old(ss, nss):
        heart_beat_range = [30, 300]  # [100, 240]
        result = np.zeros(ss.shape[0])
        ss = ss[int(heart_beat_range[0]):int(heart_beat_range[1])]
        nss = nss[int(heart_beat_range[0]):int(heart_beat_range[1])]
        diff_ss = np.around((ss[1:] - ss[:-1]) * 50)
        diff_nss = np.around((nss[1:] - nss[:-1]) * 50)
        spec_subtract = diff_ss > diff_nss
        count = 0
        for i in range(len(spec_subtract)):
            if spec_subtract[i]:
                count += 1
                result[i + int(heart_beat_range[0] + 1)] = count ** 0.5
            elif count > 0:
                count -= 1
                result[i + int(heart_beat_range[0] + 1)] = count ** 0.5

        return result

    def de_engine_noise(self, ss, nss, engine_noise):
   
        engine_peaks, _ = find_peaks(engine_noise[150:300], height=0.3)

        ss_window = ss[150:300]
        if np.max(ss_window) > 0:
            ss_window = ss_window / np.max(ss_window)
        nss_window = nss[150:300]
        if np.max(nss_window) > 0:
            nss_window = nss_window / np.max(nss_window)

        nss_peaks, _ = find_peaks(nss_window, height=0.3)
        ss_peaks, _ = find_peaks(ss_window, height=0.3)
        if len(engine_peaks) == 0 or len(nss_peaks) == 0 or len(ss_peaks) == 0:
            return ss

        if len(self.engine_peaks) > 0:
            engine_peaks = np.append(engine_peaks, self.engine_peaks[-1])

        peak_list = []
        upper_est = 0
        lower_est = 300
        for engine_peak in engine_peaks:
            nss_engine_peak = np.abs(nss_peaks - engine_peak)
            if np.min(nss_engine_peak) < 5:
                close_peak = nss_peaks[np.argmin(nss_engine_peak)]
                ss_engine_peak = np.abs(ss_peaks - close_peak)
                if np.min(ss_engine_peak) < 5:
                    max_peak = ss_peaks[np.argmin(ss_engine_peak)] + 150
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

                    ss[int(lower):int(upper)] = np.linspace(
                        value, top_value, num=int(upper)-int(lower))
                    peak_list.append(max_peak - 150)
                    if lower_est > lower:
                        lower_est = lower
                    if upper_est < upper:
                        upper_est = upper
        self.engine_peaks.append(peak_list)
        return ss
    
    def get_trans_func(self, ss, nss, status):
        driving_spec = ss[:, status == 5]
        driving_acc = nss[:, status == 5]
        
        driving_sum = np.sum(driving_spec, axis=1)
        driving_acc_sum = np.sum(driving_acc, axis=1)
        driving_sum[driving_acc_sum == 0] = 1
        driving_acc_sum[driving_acc_sum == 0] = 1
        self.trans_func = driving_sum / driving_acc_sum  

    def get_ss(self, ss, nss, status, engine_noise):
        ss_status = np.zeros(ss.shape[1]) 
        if self.USE_SPECTRUM_SUBTRACTION == 0:
            return ss, ss_status        

        ss = ss.T
        nss = nss.T
        engine_noise = engine_noise.T
        ss_all = np.copy(ss)

        for i in range(ss.shape[0]):
            if status[i] > self.als_threshold:
                ss_status[i] = 3
                if self.USE_TRANSFER_FUNC:
                    nss[i] = self.trans_func*nss[i]                                   
                if self.USE_SPECTRUM_SUBTRACTION == 1:
                    ss_all[i] = ss_all[i] = self.get_gd_of_one_window_opt(ss[i], nss[i],[self.fft_window_size, self.fft_window_size * 10])  
                elif self.USE_SPECTRUM_SUBTRACTION == 2:
                    ss_all[i] = self.spec_sub_direct(ss[i], nss[i])  
                elif self.USE_SPECTRUM_SUBTRACTION == 3:
                    ss_all[i] = self.spec_sub_trans(ss[i], nss[i])
                elif self.USE_SPECTRUM_SUBTRACTION == 4:
                    ss_all[i] = self.spec_sub_old(ss[i], nss[i])  
            elif status[i] > self.engine_threshold and self.USE_ENGINE_DENOISE:
                ss_status[i] = 4
                ss_all[i] = self.de_engine_noise(ss[i], nss[i], engine_noise[i])  # org

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

    def get_overlap_rpm(self, STFT, overlap_weight):
        stft_overlap1 = np.zeros((STFT.shape[0], STFT.shape[1]))
        stft_overlap = np.zeros((STFT.shape[0], STFT.shape[1]))
        for i in range(0, 120):
            for j in range(len(overlap_weight)):
                moving_max = np.mean(STFT[i * (j + 1): (i + 1) * (j + 1)], axis=0)
                stft_overlap1[i] += moving_max * overlap_weight[j]
        return stft_overlap      
        
    def get_overlap(self, STFT, overlap_weight):
        stft_overlap1 = np.zeros((STFT.shape[0], STFT.shape[1]))
        stft_overlap = np.zeros((STFT.shape[0], STFT.shape[1]))
        for i in range(0, 120):
            for j in range(len(overlap_weight)):
                moving_max = np.mean(STFT[i * (j + 1): (i + 1) * (j + 1)], axis=0)
                stft_overlap1[i] += moving_max * overlap_weight[j]

        for i in range(60):
            idx1 = int(i * 1/2 + 0.5)
            idx2 = int(i * 3/4 + 0.5)
            idx3 = int(i * 3/5 + 0.5)
            idx4 = int(i * 4/5 + 0.5)
            idx5 = int(i * 3/7 + 0.5)
            idx6 = int(i * 4/7 + 0.5)
            idx7 = int(i * 5/7 + 0.5)

            reimburse = np.zeros((8, len(stft_overlap1[0])))
            reimburse[0] = stft_overlap1[idx1] * overlap_weight[1] * overlap_weight[3]
            reimburse[1] = stft_overlap1[idx1] * overlap_weight[2] * overlap_weight[5]
            reimburse[2] = stft_overlap1[idx2] * overlap_weight[2] * overlap_weight[3]
            reimburse[3] = stft_overlap1[idx3] * overlap_weight[2] * overlap_weight[4]
            reimburse[4] = stft_overlap1[idx4] * overlap_weight[3] * overlap_weight[4]
            reimburse[5] = stft_overlap1[idx5] * overlap_weight[2] * overlap_weight[6]
            reimburse[6] = stft_overlap1[idx6] * overlap_weight[3] * overlap_weight[6]
            reimburse[7] = stft_overlap1[idx7] * overlap_weight[4] * overlap_weight[6]
            sss2 = stft_overlap1[i] + (np.sum(reimburse[0:8], axis=0) / 8)

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

    def overlap_index_2_weight(self, overlap_index):
        overlap_weight = np.zeros(self.max_overlap)
        if overlap_index == 0:
            overlap_weight[1:5] = 1        
        else:
            for mul in range(self.max_overlap):
                overlap_weight[mul] = (overlap_index >> mul) & 1
        return overlap_weight

    def get_overlap_list(self, confidence_level, bpm_spectrum):
        ss = self.ss_denoise.T
        count_overlap_index =  np.zeros(2**self.max_overlap)
        overlap_index_selected = 0
        for i in range(len(ss)):#len(ss)
            # in static and idle and confidence_level 1            
            if confidence_level[i] and (self.status[i] == 1 or self.status[i] == 3):
            #if (self.status[i] == 1 or self.status[i] == 3):
                # search harmonic peak in detected bpm in each time and get overlap_list
                overlap_index, peak_num = self.get_overlap_index(ss[i], bpm_spectrum[i] )
                if peak_num>=4:
                    count_overlap_index[overlap_index] += 1
                
                if count_overlap_index[overlap_index] >= 4:
                    overlap_index_selected = overlap_index
                    break

        overlap_weight = self.overlap_index_2_weight(overlap_index_selected)
        print('time to get overlap index:%d, overlap weight: [%d,%d,%d,%d,%d,%d,%d]' % (i, overlap_weight[0], overlap_weight[1], overlap_weight[2], overlap_weight[3], overlap_weight[4], overlap_weight[5], overlap_weight[6]))
        return overlap_weight

    def get_bpm_final(self, bpm_overlap_data, lower, upper, status=np.array([])):
        result_bpm = []
        stable_index_list = []
        self.bpm_interval = np.zeros(bpm_overlap_data.shape)
        self.bpm_interval[lower:upper] = np.copy(bpm_overlap_data[lower:upper])
        self.normalization(self.bpm_interval)
        bpm_interval = self.bpm_interval.T

        for i in range(len(bpm_interval)):


            if len(status) == 0 or self.USE_BDT == 0:
                truth_peak_index = np.argmax(bpm_interval[i])
            elif (status[i] == 1 or status[i] == 3) and self.confidence_level[i] == 1:
                self.stable_index.append(i)
                truth_peak_index = np.argmax(bpm_interval[i])

                stable_index_list.append(truth_peak_index)
                if len(stable_index_list) > 20:
                    stable_index_list.pop(0)
            elif i <= 6:
                truth_peak_index = np.argmax(bpm_interval[i])
            else:
                if len(stable_index_list) == 0:
                    truth_peak_index = np.argmax(bpm_interval[i])
                else:
                    #if self.confidence_level[i] == 1:
                    #    stable_index_list.append(truth_peak_index)
                    truth_peak_index = np.median(np.copy(stable_index_list[(len(stable_index_list) + 1) % 2:]))
                    height = 0.5
                    peaks, _ = find_peaks(bpm_interval[i], height=height)
                    peak_candidate_amp = bpm_interval[i][peaks]
                    if len(peaks) > 2:
                        truth_peak_index = self.decide_rule_bdt(peak_candidate_amp, peaks, truth_peak_index)
                    else:
                        truth_peak_index = np.argmax(bpm_interval[i])
            
                if self.USE_NO_JUMP_BPM == 1: # fix PVT_2020-11-11-18-55        
                    if i>self.bpm_no_jump_range:
                        mm_bpm = np.median(result_bpm[-(self.bpm_no_jump_range-1):])
                        if (truth_peak_index - mm_bpm)>self.bpm_no_jump_range:
                            truth_peak_index = mm_bpm + 2
                        elif (truth_peak_index - mm_bpm) < -self.bpm_no_jump_range:
                            truth_peak_index = mm_bpm - 2

                #if self.confidence_level[i] == 1:
                #    stable_index_list.append(truth_peak_index)        

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

        return peaks_locs_sort, peaks_amps_sort\

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
        peaks_mean = np.mean(down_acc_window[down_acc_window_peaks])

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

    def get_scenario(self, bcg_signal, acc_signal, algo_acc_signal, acc_org_peak_ratio):
        window = self.fs
        result = np.zeros(int((len(acc_signal) + 63) / window))
        down_acc_power = np.zeros(int((len(acc_signal) + 63) / window))
        bcg_mean_power = np.zeros(int((len(acc_signal) + 63) / window))
        start = 0
        filter_bcg_signal = self.FIR_filter(window, 0.5, 30, bcg_signal)
        filter_acc_signal = self.FIR_filter(window, 0.5, 30, acc_signal)

        while start < len(acc_signal) - window * (self.ss_t_len-1):
            i = int(start / window)
            down_acc_window = self.exclude_abnormal_peak(filter_acc_signal[start: start + window * self.ss_t_len])
            algo_acc_signal_window = self.exclude_abnormal_peak(algo_acc_signal[start: start + window * self.ss_t_len])

            down_acc_power[i] = np.mean(down_acc_window)
            algo_acc_signal_power = np.mean(algo_acc_signal_window)
            bcg_window = filter_bcg_signal[start: start + window * self.ss_t_len]
            bcg_mean_power[i] = np.mean(np.abs(bcg_window)) - np.abs(np.mean(bcg_window))
            start += window

            if down_acc_power[i] > self.DRIVING_THRESHOLD:
                if algo_acc_signal_power < self.IDLE_THRESHOLD:
                    result[i + 2] = 1
                else:
                    result[i + 2] = 5
            elif down_acc_power[i] > self.IDLE_THRESHOLD:
                if acc_org_peak_ratio[i] == 1:
                    result[i + 2] = 3
                else:
                    result[i + 2] = 1                    
            else:
                result[i + 2] = 1

        result = self.moving_median(result, 2)
        acc_max_idx = np.argmax(down_acc_power)
        while bcg_mean_power[acc_max_idx] == 0:
            down_acc_power[acc_max_idx] = 0
            acc_max_idx = np.argmax(down_acc_power)
        power_diff = bcg_mean_power / bcg_mean_power[acc_max_idx] - down_acc_power / down_acc_power[acc_max_idx]
        power_diff[power_diff < 0] = 0
        if np.sum(result == 1) == 0:
            motion_threshold_static = 1
        else:
            motion_threshold_static = np.mean(power_diff[result == 1])
            if motion_threshold_static < 0.025:
                motion_threshold_static = 0.025

        if np.sum(result == 3) == 0:
            motion_threshold_idle = 1
        else:
            motion_threshold_idle = np.mean(power_diff[result == 3])
            if motion_threshold_idle < 0.025:
                motion_threshold_idle = 0.025

        if np.sum(result == 5) == 0:
            motion_threshold_driving = 1
        else:
            motion_threshold_driving = np.mean(power_diff[result == 5])
            if motion_threshold_driving < 0.025:
                motion_threshold_driving = 0.025

        result[np.logical_and(result == 1, power_diff > (motion_threshold_static / 2))] += 1
        result[np.logical_and(result == 3, power_diff > (motion_threshold_idle / 2))] += 1
        result[np.logical_and(result == 5, power_diff > (motion_threshold_driving / 2))] += 1

        self.acc_power = down_acc_power
        self.bcg_power = bcg_mean_power
        half_t_len = round(self.ss_t_len/2)
        return np.array(result[half_t_len-1:-half_t_len])

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
        while min_index <= 300 and spec[min_index] < upper_min:
            upper_min = spec[min_index]
            min_index += 1

        return spec[int(peak_idx)] * 2 - upper_min - lower_min

    def get_confidence_level(self, spec_data, peak_index, status):
        spec_data = spec_data.T
        result = []
        
        for i in range(len(peak_index)):
            peak_height = self.find_peak_local_height(spec_data[i], peak_index[i])
            overlap_index, peak_num = self.get_overlap_index(spec_data[i], peak_index[i] )
            if status[i] <= 4:
                result.append(peak_height > self.reliable_threshold)
            else:
                result.append( (overlap_index > 2) and (peak_height > self.reliable_threshold))

        return np.array(result)

    @staticmethod
    def get_engine_noise(spec_data):
        spec_data_overlap = np.zeros((spec_data.shape[0], spec_data.shape[1]))
        overlap_weight = [0, 1, 1, 1]
        for i in range(150, 300):
            for j in range(len(overlap_weight)):
                if i * (j + 1) < 960:
                    moving_max = np.amax(spec_data[i * (j + 1): (i + 1) * (j + 1)], axis=0)
                    spec_data_overlap[i] += moving_max * overlap_weight[j]
        return spec_data_overlap

    @staticmethod
    def moving_median(ppi, half_range):
        ppi1 = np.copy(ppi)
        for i in range(half_range, len(ppi) - half_range):
            ppi1[i] = np.median(ppi[i - half_range:i + half_range+1])
        return ppi1

    @staticmethod
    def _post_process_bpm(ppi, status):
        ppi1 = np.copy(ppi)
        for i in range(2, len(ppi) - 2):
            if status[i] < 5:
                ppi1[i] = np.median(ppi[i - 2:i + 3])
        for i in range(2, len(ppi) - 2):
                ppi1[i] = np.mean(ppi[i - 2:i + 3])        
        return ppi1

    @staticmethod
    def _pre_process(data, g_sensor_data_x, g_sensor_data_y=[], g_sensor_data_z=[]):
        data = np.array(data)
        data = data - data[0]

        if len(g_sensor_data_y) != 0:
            A = np.array([np.mean(g_sensor_data_x), np.mean(g_sensor_data_y), np.mean(g_sensor_data_z)])
            B = np.array([(g_sensor_data_x), (g_sensor_data_y), (g_sensor_data_z)])
            g_sensor_data = B[np.argmax(abs(A))] * np.sign(A[np.argmax(abs(A))])
        else:
            g_sensor_data = g_sensor_data_x

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
        acc_data = np.concatenate(
            (np.array([0, 0, 0, 0, 0]), acc_data), axis=0)
        acc_data = np.concatenate(
            (acc_data, np.array([0, 0, 0, 0, 0])), axis=0)
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

    def get_heart_rate(self, data, g_sensor_data):
        self.p_sensor_data, self.g_sensor_data = self._pre_process(data, g_sensor_data)

        # filtering
        self.filter_p_sensor_data = self.filtering_signal(self.p_sensor_data, self.bpm_p_cutoff_freq[0], self.bpm_p_cutoff_freq[1])
        self.filter_g_sensor_data = self.filtering_signal(self.g_sensor_data, self.bpm_g_cutoff_freq[0], self.bpm_g_cutoff_freq[1])

        # STFT (short time Fourier transform)
        _, _, s  = scipy.signal.stft(self.filter_p_sensor_data, window='hamming', nperseg=self.fs * self.ss_t_len, noverlap=self.fs * (self.ss_t_len - 1), nfft=self.fs * self.fft_window_size, boundary=None)
        _, _, ns = scipy.signal.stft(self.filter_g_sensor_data, window='hamming', nperseg=self.fs * self.ss_t_len, noverlap=self.fs * (self.ss_t_len - 1), nfft=self.fs * self.fft_window_size, boundary=None)
        self.ss  = np.abs(s)
        self.nss = np.abs(ns)

        # get engine noise
        self.normalization(self.nss)        
        self.engine_noise = self.get_engine_noise(self.nss)
        self.normalization(self.engine_noise)

        # get scenario and status
        acc_org_peak_ratio = (np.max(self.nss[200:,:],0)/np.mean(self.nss[200:,:],0)) > 4
        self.status = self.get_scenario(self.p_sensor_data, self.g_sensor_data, self.filter_g_sensor_data, acc_org_peak_ratio)

        # Spectrum subtraction
        self.reserve_freq_band(self.ss,  self.reserved_freq_lower_bound, self.reserved_freq_upper_bound)
        self.reserve_freq_band(self.nss, self.reserved_freq_lower_bound, self.reserved_freq_upper_bound)
        self.normalization(self.ss)
        self.normalization(self.nss)
        self.get_trans_func(np.copy(self.ss), np.copy(self.nss), self.status)
        self.ss_denoise, self.ss_status = self.get_ss(np.copy(self.ss), np.copy(self.nss), self.status, self.engine_noise)
        if self.USE_TIME_FUSION:
            self.ss_denoise = self.get_time_fusion_spectrum(self.ss_denoise, self.p_sensor_data, self.g_sensor_data)
        
        # Overlap spectrum
        self.ss_denoise_overlaped = self.get_overlap(self.ss_denoise, self.overlap_weight)
        self.normalization(self.ss_denoise_overlaped)

        # get heart rate and candidate peaks
        self.peaks_locs, self.peaks_amps = self.get_peaks_array(self.ss_denoise, 25)
        self.bpm_pre = self.get_bpm_final(np.copy(self.ss_denoise_overlaped), self.bpm_search_lower, self.bpm_search_upper)

        # run iteration to get best overlap weighting and bpm results
        for _ in range(self.iteration_times):
            self.confidence_level = self.get_confidence_level(self.ss_denoise_overlaped, (self.bpm_pre - 1) * self.fft_window_size // 60, self.status)
            if self.USE_MANUALLY_WEIGHT == 0:
                self.overlap_weight = self.get_overlap_list(self.confidence_level, (self.bpm_pre - 1) * self.fft_window_size // 60)
                self.ss_denoise_overlaped = self.get_overlap(self.ss_denoise, self.overlap_weight)
                self.normalization(self.ss_denoise_overlaped)
                self.bpm_pre = self.get_bpm_final(np.copy(self.ss_denoise_overlaped), self.bpm_search_lower, self.bpm_search_upper, self.status)

        self.peaks_locs_o, self.peaks_amps_o = self.get_peaks_array(self.ss_denoise_overlaped, 0)
        
        # heart rate post processing
        if self.USE_BPM_POST_PROCESSING:
            self.bpm = self._post_process_bpm(self.bpm_pre, self.status)
        else:
            self.bpm = self.bpm_pre

    def get_respiration_rate(self, data, g_sensor_data):
        rpm_b, rpm_a = filter.butter_bandpass(self.rpm_cutoff_freq[0], self.rpm_cutoff_freq[1], self.fs, self.rpm_filter_order)
        data = data - data[0]

        self.rpm_data_out = lfilter(rpm_b, rpm_a, data)
        self.rpm_data_out = np.around(self.rpm_data_out)
        self.filter_rpm_data = np.copy(self.rpm_data_out)
        self.rpm_data_out = self.rpm_data_out[::2]
        _, _, rpm_s = scipy.signal.stft(self.rpm_data_out, window='hamming', nperseg=self.fs * 10, noverlap=self.fs * 8, nfft=self.fs * self.fft_window_size)

        self.rpm_s = np.abs(rpm_s[:, 1:-1])
        self.normalization(self.rpm_s)
        self.rpm_overlap = self.get_overlap_rpm(self.rpm_s, [1, 1])
        rpm_interval = self.dismantling(self.rpm_overlap, 3, 30)
        rpm_interval = rpm_interval.T
        self.rpm = self.get_rpm(rpm_interval, 3)
        self.rpm = self.moving_median(self.rpm, 2)
        rpm_list = self.rpm
        x = np.linspace(0, len(rpm_list) - 1, len(rpm_list))
        f = interp1d(x, rpm_list, kind='linear')
        x_new = np.linspace(0, len(rpm_list) - 1, len(rpm_list) * 5 - 4)
        self.rpm = f(x_new)

    #############################################################################################
    def main_func(self, data, g_sensor_data):
        self.get_heart_rate(data, g_sensor_data)
        if self.USE_GET_RPM:
            self.get_respiration_rate(data, g_sensor_data)
        return 0