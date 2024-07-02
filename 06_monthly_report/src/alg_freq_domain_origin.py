import numpy as np
import scipy
import math
from scipy.interpolate import interp1d
from . import filter
from scipy.signal import find_peaks, lfilter, peak_prominences,firwin, filtfilt, hilbert
from scipy.optimize import minimize
from scipy.signal import savgol_filter


class Alg_freq_domain():
    def __init__(self, fs=64, fft_window_size=30):
        self.fft_window_size = fft_window_size
        self.fs = fs
        # =========Parameters==========
        self.max_overlap = 7
        self.overlap_weight = np.array([1, 1, 1, 0, 0, 0, 0])
        self.b = 2
        self.td_remove_DC_lower = 10
        self.als_threshold = 0.8
        self.reliable_threshold = 1
        self.als_range_lower = self.fft_window_size * 1
        self.als_range_upper = self.fft_window_size * 10
        self.NOBODY_THRESHOLD = 5000
        self.SITTING_THRESHOLD = 3000000
        # ==========Flags===============
        self.USE_ALS = 0
        self.USE_IIR_FILTER = 0
        self.USE_MANUALLY_WEIGHT = 0
        self.USE_TIME_FUSION = 0
        self.USE_BDT = 1
        self.USE_SPECTRUM_SUBTRACTION = 1
        self.USE_ENGINE_DENOISE = 1
        self.USE_TIME_INHIBITION = 0
        # ==========Filter==============
        self.bpm_filter_order = 4
        self.bpm_cutoff_freq = [1.6, 8]
        self.rpm_filter_order = 2
        self.rpm_cutoff_freq = [0.05, 1]
        self.bpm_search_lower = 25
        self.bpm_search_upper = 55
        # ==============================
        self.ground_truth_bpm = np.array([])
        self.bpm_data = np.array([])
        self.stable_index = []
        self.g_sensor_data = np.array([])
        self.filter_bpm_data = np.array([])
        self.filter_g_sensor_data = np.array([])
        self.filter_time_bpm_data = np.array([])
        self.filter_time_g_sensor_data = np.array([])
        self.status = np.array([])
        self.reliability = []
        self.similarity = []
        self.engine_similarity = []
        self.nss_sum = np.array([])
        self.ss = np.array([])
        self.nss = np.array([])
        self.ass = np.array([])
        self.tss = np.array([])
        self.rpm_s = np.array([])
        self.rpm_data_out = np.array([])
        self.filter_rpm_data = np.array([])
        self.rpm = np.array([])
        self.rpm_overlap = np.array([])
        self.ss_org = np.array([])
        self.nss_org = np.array([])
        self.norm_data1 = np.array([])
        self.norm_data2 = np.array([])
        self.bpm_pre = np.array([])
        self.bpm = np.array([])
        self.bpm_result_out = np.array([])
        self.time_corrcoef = np.array([])
        self.result_ss = np.array([])
        self.bpm_interval = np.array([])
        self.engine_noise = np.array([])
        self.frequency = np.linspace(0, fs // 2, fft_window_size * fs // 2 + 1)
        self.engine_denoise_list = []
        self.stft_time = np.array([])
        # ============================== ALS ========================================
        self.USE_CLIP = 0 # Jimmy test
        self.USE_MEDIAN_POST_PROCESSING = 0 # default: 1
        self.ss_max_arr = np.array([]) # ss normaization value
        self.nss_max_arr = np.array([]) # ss normaization value
        self.ss_status = np.array([]) # ss normaization value
        self.ld = 100

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
    def remove_DC(SS, upper):
        SS[0:upper] = 0

    @staticmethod
    def overlapping_cutoff(SS, upper):
        SS[upper:] = 0

    @staticmethod
    def spec_sub(ss, nss):
        heat_beat_range = [100, 240]
        result = np.zeros(ss.shape[0])
        ss = ss[int(heat_beat_range[0]):int(heat_beat_range[1])]
        nss = nss[int(heat_beat_range[0]):int(heat_beat_range[1])]
        diff_ss = np.around((ss[1:] - ss[:-1]) * 50)
        diff_nss = np.around((nss[1:] - nss[:-1]) * 50)
        spec_subtract = diff_ss > diff_nss
        count = 0
        for i in range(len(spec_subtract)):
            if spec_subtract[i]:
                count += 1
                result[i + int(heat_beat_range[0] + 1)] = count ** 0.5
            elif count > 0:
                count -= 1
                result[i + int(heat_beat_range[0] + 1)] = count ** 0.5

        result = savgol_filter(result, 21, 7)

        return result

    @staticmethod
    def de_engine_noise(ss, engine_noise):
        max_peak = np.argmax(ss)
        engine_peak = np.argmax(engine_noise)
        median_value = np.median(engine_noise)
        if(engine_peak - median_value) < 0.7:
            return ss

        if abs(max_peak - engine_peak) < 5:
            max_value = ss[max_peak]
            upper = max_peak + 1
            lower = max_peak - 1
            value = max_value
            while ss[upper] < value:
                value = ss[upper]
                upper += 1

            value = max_value
            while ss[lower] < value:
                value = ss[lower]
                lower -= 1

            ss[lower:upper] = value

        return ss

    def get_ss(self, ss, nss, mean_spec, engine_noise):
        ss = ss.T
        nss = nss.T
        engine_noise = engine_noise.T
        ss_all = np.copy(ss)
        ss_status = np.zeros(ss.shape[0]) #Jimmy
        for i in range(ss.shape[0]):
            if mean_spec[i] > self.als_threshold:
                ss_status[i] = 1 #Jimmy
                if self.USE_ALS:
                    ss_status[i] = 2 #Jimmy
                    ss_all[i] = self.get_gd_of_one_window_opt(ss[i], nss[i],
                                                              [self.fft_window_size, self.fft_window_size * 10])
                else:
                    ss_status[i] = 3 #Jimmy
                    ss_all[i] = self.spec_sub(ss[i], nss[i])
            elif self.USE_ENGINE_DENOISE and self.engine_denoise_list[i]:
                ss_status[i] = 4 #Jimmy
                ss_all[i] = self.de_engine_noise(ss[i], engine_noise[i])

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
        return SS

    @staticmethod
    def get_overlap(STFT, overlap_weight):
        stft_overlap = np.zeros((STFT.shape[0], STFT.shape[1]))
        for i in range(0, 80):
            for j in range(len(overlap_weight)):
                moving_max = np.amax(STFT[i * (j + 1): (i + 1) * (j + 1)], axis=0)
                stft_overlap[i] += moving_max * overlap_weight[j]
        return stft_overlap

    def get_overlap_list(self, reliability, bpm):
        ss = self.result_ss.T
        count = 0
        peak_val = [0] * 3
        overlap_weight = np.zeros(self.max_overlap)
        overlap_weight[1:5] = 1
        for i in range(len(ss)):
            if reliability[i] > self.reliable_threshold and self.similarity[i] < self.als_threshold:
                bpm_peak = bpm[i] // 2
                peak_val[count] = 0
                peak_num = 0

                for mul in range(1, self.max_overlap + 1):
                    overlap_ss = ss[i][::mul]
                    peaks, _ = find_peaks(overlap_ss, height=0.5)
                    if not peaks.any():
                        continue
                    diff = np.abs(peaks - bpm_peak)
                    min_index = np.argmin(diff)
                    min_diff = diff[min_index]
                    if min_diff < 2:
                        peak_val[count] += 1 << (mul - 1)
                        peak_num += 1
                if peak_num >= 3:
                    count += 1
                else:
                    peak_val[count] = 0

                if count >= 3:
                    if peak_val[0] == peak_val[1] and peak_val[0] == peak_val[2]:
                        for mul in range(self.max_overlap):
                            overlap_weight[mul] = (peak_val[0] >> mul) & 1
                        return overlap_weight
                    elif peak_val[0] == peak_val[1] or peak_val[0] == peak_val[2]:
                        count = 1
                    elif peak_val[1] == peak_val[2]:
                        peak_val[0] = peak_val[1]
                        count = 1
                    else:
                        count = 0

        return overlap_weight

    def get_bpm_final(self, bpm_overlap_data, lower, upper, status=np.array([])):
        result_bpm = []
        stable_index_list = []
        self.bpm_interval = np.zeros(bpm_overlap_data.shape)
        self.bpm_interval[lower:upper] = np.copy(bpm_overlap_data[lower:upper])
        self.normalization(self.bpm_interval)
        bpm_interval = np.copy(self.bpm_interval.T)

        for i in range(len(bpm_interval)):
            height = 0.5
            peaks, _ = find_peaks(bpm_interval[i], height=height)
            peak_candidate_amp = bpm_interval[i][peaks]

            if len(status) == 0 or self.USE_BDT == 0:
                truth_peak_index = np.argmax(bpm_interval[i])
            elif status[i] == 1 or status[i] == 2:
                self.stable_index.append(i)
                truth_peak_index = np.argmax(bpm_interval[i])
                if truth_peak_index <= 1 or len(bpm_interval[i]) - truth_peak_index <= 1 :
                    truth_peak_index = np.median(np.copy(stable_index_list[(len(stable_index_list)+1) % 2:]))
                    
                stable_index_list.append(truth_peak_index)
                if len(stable_index_list) > 20:
                    stable_index_list.pop(0)
            else:
                if len(stable_index_list) < 3:
                    truth_peak_index = np.argmax(bpm_interval[i])
                else:
                    truth_peak_index = np.median(np.copy(stable_index_list[(len(stable_index_list)+1) % 2:]))
                    if len(peaks) != 0:
                        truth_peak_index = self.decide_rule_bdt(peak_candidate_amp, peaks, truth_peak_index)

            result_bpm.append(truth_peak_index)
        result_bpm = np.array(result_bpm)
        result_bpm = result_bpm * 60 / self.fft_window_size + 1
        return result_bpm

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
        peak_sep = np.exp(-2 * (peak_sep / 2 * self.b) ** 2)
        peak_amp = ss_candidate_amp ** 1.5
        truth_peak_index = ss_candidate_index[np.argmax(peak_sep * peak_amp)]
        return truth_peak_index

    def get_status(self, bcg_signal, similarity, reliability, engine_list):
        result = [5]
        for i in range(len(similarity)):
            if result[i] == 3 or result[i] == 1 or result[i] == 2 or result[i] == 4:
                if similarity[i] > self.als_threshold:
                    result.append(4)
                elif reliability[i] < 1:
                    result.append(3)
                elif engine_list[i]:
                    result.append(2)
                else:
                    result.append(1)
            elif result[i] == 5 or result[i] == 0:
                if similarity[i] > self.als_threshold:
                    result.append(4)
                else:
                    if reliability[i] > 1:
                        if engine_list[i]:
                            result.append(2)
                        else:
                            result.append(1)
                    elif reliability[i] > 0.5:
                        result.append(3)
                    else:
                        result.append(5)


            nobody_count = np.sum(np.abs(bcg_signal[(i+4)*self.fs:(i+5)*self.fs]) < self.NOBODY_THRESHOLD)
            sitting_count = np.sum(np.abs(bcg_signal[(i+4)*self.fs:(i+5)*self.fs]) > self.SITTING_THRESHOLD)
            if nobody_count > 60:
                result[i+1] = 0
            if sitting_count > 50:
                result[i+1] = 5
        result.pop(0)
        return np.array(result)

    def get_reliability(self, spec_data, peak_index):
        spec_data = spec_data.T
        result = []
        peak_index = (peak_index - 1) * self.fft_window_size // 60
        for i in range(len(peak_index)):
            spec = spec_data[i]
            min_index = int(peak_index[i])
            lower_min = spec[min_index]

            min_index -= 1
            while min_index >= 0 and spec[min_index] < lower_min:
                lower_min = spec[min_index]
                min_index -= 1

            min_index = int(peak_index[i])
            upper_min = spec[min_index]
            min_index += 1
            while min_index <= 300 and spec[min_index] < upper_min:
                upper_min = spec[min_index]
                min_index += 1
            result.append(spec[int(peak_index[i])] * 2 - upper_min - lower_min)
        return result

    @staticmethod
    def get_engine_noise(spec_data):
        spec_data_overlap = np.zeros((spec_data.shape[0], spec_data.shape[1]))
        spec_data_overlap[0] = 0.5
        overlap_weight = [0, 1, 0, 1]
        for i in range(150, 240):
            for j in range(len(overlap_weight)):
                moving_max = np.amax(spec_data[i * (j + 1): (i + 1) * (j + 1)], axis=0)
                spec_data_overlap[i] += moving_max * overlap_weight[j]
        return spec_data_overlap

    @staticmethod
    def get_engine_list(spec_data):
        spec_data[0:100] = 0
        spec_data = spec_data.T
        engine_list = np.zeros(len(spec_data))
        for i in range(len(spec_data)):
            engine_peak = np.argmax(spec_data[i, 750:900]) + 750
            peaks, _ = find_peaks(spec_data[i, 0:750], distance=50)
            temp = []
            for j in range(2, 5):
                temp.append(np.min(np.abs(peaks - (engine_peak / j))) < 5)
            if np.sum(temp) >= 2:
                engine_list[i] = True

        for i in range(2, len(engine_list) - 2):
            engine_list[i] = np.median(engine_list[i-2: i+3])
        return engine_list

    @staticmethod
    def _post_process_rpm(ppi):
        ppi1 = np.copy(ppi)
        for i in range(2, len(ppi) - 2):
            ppi1[i] = np.median(ppi[i - 2:i + 3])
        return ppi1

    @staticmethod
    def _post_process_bpm(bpm_pre, status):
        bpm = np.copy(bpm_pre)
        stable_bpm = -1
        for i in range(len(bpm_pre)):
            if status[i] == 1:
                if stable_bpm == -1:
                    stable_bpm = bpm[i]
                elif abs(bpm_pre[i] - stable_bpm) > 5:
                    bpm[i] = stable_bpm * 2 / 3 + bpm_pre[i] / 3
                stable_bpm = bpm[i]
            elif status[i] == 2:
                if stable_bpm == -1:
                    bpm[i] = -1
                elif abs(bpm_pre[i] - stable_bpm) > 5:
                    bpm[i] = stable_bpm * 4 / 5 + bpm_pre[i] / 5
            elif status[i] == 4:
                bpm[i] = stable_bpm
        return bpm

    @staticmethod
    def _pre_process(data, g_sensor_data_x, g_sensor_data_y=[], g_sensor_data_z=[]):
        data = np.array(data)
        data = data - data[0]

        if len(g_sensor_data_y) != 0:
            A = np.array([np.mean(g_sensor_data_x), np.mean(g_sensor_data_y), np.mean(g_sensor_data_z)])
            B = np.array([(g_sensor_data_x), (g_sensor_data_y), (g_sensor_data_z)])
            # g_sensor_data = np.sqrt(g_sensor_data_x**2 + g_sensor_data_y**2 + g_sensor_data_z**2) * np.sign(A[np.argmax(abs(A))])
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
        start = 0
        window = 320
        acc_data = np.concatenate((np.array([0, 0, 0, 0, 0]), acc_data), axis=0)
        acc_data = np.concatenate((acc_data, np.array([0, 0, 0, 0, 0])), axis=0)
        offset_list = []
        corrcoef_list = []
        scale_list = []
        while start + window < len(bcg_data):
            bcg_window = bcg_data[start: start + window]
            bcg_window = bcg_window - np.mean(bcg_window)
            max_corrcoef = -1

            for i in range(10):
                acc_window = acc_data[start + i: start + i + window]
                corrcoef = np.corrcoef(bcg_window, acc_window)[0][1]
                if corrcoef > max_corrcoef:
                    offset = i
                    max_corrcoef = corrcoef
                    scale = np.mean(np.abs(bcg_window)) / np.mean(np.abs(acc_window))
            if max_corrcoef < 0.7:
                offset = -1
                scale = -1
            start = start + self.fs
            offset_list.append(offset)
            corrcoef_list.append(max_corrcoef)
            scale_list.append(scale)

        offset_list = np.array(offset_list)
        offset_list = offset_list[offset_list != -1]
        counts = np.bincount(offset_list)
        offset = np.argmax(counts)

        scale_list = np.array(scale_list)
        scale_list = scale_list[scale_list != -1]
        scale = np.median(scale_list)

        acc_data = acc_data[offset:len(bcg_data) + offset]

        diff_ss = np.around((bcg_data[1:] - bcg_data[:-1]))
        diff_nss = np.around((acc_data[1:] - acc_data[:-1]) * scale)
        spec_subtract = np.sign(diff_ss) != np.sign(diff_nss)
        count = 0
        result = np.copy(bcg_data)[1:] * spec_subtract
        # for i in range(len(spec_subtract)):
        #     if spec_subtract[i]:
        #         count += 1
        #         result[i] = (count ** 0.5) * 10000
        #     elif count > 0:
        #         count -= 1
        #         result[i] = (count ** 0.5) * 10000

        for i in range(len(corrcoef_list)):
            if corrcoef_list[i] < 0.7:
                if i == 0:
                    # result[:5 * self.fs] = bcg_data[:5 * self.fs]
                    result[:5 * self.fs] = 0
                else:
                    # result[(i+4) * self.fs:(i+5) * self.fs] = bcg_data[(i+4) * self.fs:(i+5) * self.fs]
                    result[(i+4) * self.fs:(i+5) * self.fs] = 0

        return result

    #############################################################################################

    def main_func(self, data, g_sensor_data):

        self.bpm_data, self.g_sensor_data = self._pre_process(data, g_sensor_data)

        _, _, non_filter_spec = scipy.signal.stft(self.bpm_data, window='hamming', nperseg=self.fs * 5,
                                                  noverlap=self.fs * 4, nfft=self.fs * self.fft_window_size, boundary=None)
        temp = np.abs(non_filter_spec)
        self.normalization(temp)
        self.engine_noise = self.get_engine_noise(temp)
        self.normalization(self.engine_noise)
        self.engine_noise[0] = 0
        _, _, non_filter_g_spec = scipy.signal.stft(self.g_sensor_data, window='hamming', nperseg=self.fs * 5,
                                                    noverlap=self.fs * 4, nfft=self.fs * self.fft_window_size, boundary=None)

        temp = np.abs(non_filter_g_spec)
        self.normalization(temp)
        self.engine_denoise_list = self.get_engine_list(temp)

        '''filter'''
        if self.USE_IIR_FILTER:
            bpm_b, bpm_a = filter.butter_bandpass(self.bpm_cutoff_freq[0], self.bpm_cutoff_freq[1], self.fs,
                                                  self.bpm_filter_order)
            self.filter_bpm_data = lfilter(bpm_b, bpm_a, self.bpm_data)
            self.filter_g_sensor_data = lfilter(bpm_b, bpm_a, self.g_sensor_data)
        else:
            self.filter_bpm_data = self.FIR_filter(self.fs, self.bpm_cutoff_freq[0], self.bpm_cutoff_freq[1],
                                                   self.bpm_data)
            self.filter_g_sensor_data = self.FIR_filter(self.fs, self.bpm_cutoff_freq[0], self.bpm_cutoff_freq[1],
                                                        self.g_sensor_data)

        # Jimmy
        if self.USE_CLIP==1:
            len_p = len(self.filter_bpm_data) - 64
            clip_th = 100000
            clip_reduce = 5.0
            for i in range(len_p):
                p_value = self.filter_bpm_data[i]
                if max(self.filter_g_sensor_data[i:i+63])<10:
                    if p_value>clip_th:
                        self.filter_bpm_data[i] = (p_value-clip_reduce)/clip_reduce + clip_th
                    elif p_value<-clip_th:
                        self.filter_bpm_data[i] = (p_value-clip_reduce)/clip_reduce - clip_th
                
            #self.filter_bpm_data = self.clip_data(self.filter_bpm_data)
            #self.filter_g_sensor_data = self.clip_data(self.filter_g_sensor_data)

        '''STFT'''
        _, T, s = scipy.signal.stft(self.filter_bpm_data, window='hamming', nperseg=self.fs * 5,
                                    noverlap=self.fs * 4, nfft=self.fs * self.fft_window_size, boundary=None)
        self.ss = np.abs(s)
        self.ss_org = np.copy(self.ss)
        self.normalization(self.ss_org)

        _, _, ns = scipy.signal.stft(self.filter_g_sensor_data, window='hamming', nperseg=self.fs * 5,
                                     noverlap=self.fs * 4, nfft=self.fs * self.fft_window_size, boundary=None)
        self.stft_time = np.array(T)
        self.nss = np.abs(ns)
        self.nss_org = np.copy(self.nss)
        self.nss_max_arr = self.normalization(self.nss_org)
        self.nss_sum = np.sum(np.abs(self.nss_org[1:] - self.nss_org[:-1]), axis=0)

        '''get_similarity'''
        list_index = []
        inhibition_bpm_data = np.copy( self.filter_bpm_data)
        inhibition_g_sensor_data = np.copy(self.filter_g_sensor_data)
        flag_move = False
        
        for i in range(self.ss.shape[1]):
            self.similarity.append(self.cos_sim(self.ss[24:240, i], self.nss[24:240, i]))
            if (len(self.similarity) > 10 and i < self.ss.shape[1]-1) and self.USE_TIME_INHIBITION:
                seg_P = self.filter_bpm_data[int(self.stft_time[i] - self.fs * 2.5):int(self.stft_time[i] + self.fs * 2.5)]
                seg_g = self.filter_g_sensor_data[int(self.stft_time[i] - self.fs * 2.5):int(self.stft_time[i] + self.fs * 2.5)]
                
                
                if self.similarity[-1] - self.similarity[-2] < 0  and (self.similarity[-1] < self.als_threshold and self.similarity[-2] > self.als_threshold ):                    
                    flag_move = True
                
                if flag_move:
                    if self.similarity[-1] > self.als_threshold:
                        flag_move = False
                        continue
                    
                    seg_P = self.filter_bpm_data[int(self.stft_time[i] - self.fs * 2.5):int(self.stft_time[i] + self.fs * 2.5)]
                    seg_g = self.filter_g_sensor_data[int(self.stft_time[i] - self.fs * 2.5):int(self.stft_time[i] + self.fs * 2.5)]
                    list_index.append(i)
                    list_max = []
                    if len(seg_P) < self.fs *5:
                        break
                    for j in range(10):
                        list_max.append(max(abs(seg_P[j*int(self.fs*0.5):(j+1)*int(self.fs*0.5)])))
                        
                    if max(list_max)<100000:
                        flag_move = False
                        continue
                    
                    #method 11/4
                    '''
                    mini_bound = np.mean(list_max)
                    sum_square = np.sum((list_max - mini_bound)**2)
                    step = 1000
                    while 1:
                        if np.sum((list_max - (mini_bound + step))**2) < sum_square:
                            mini_bound = mini_bound + step
                        else:
                            break
                    
                    for j in range(10):
                        if list_max[j] >mini_bound:
                            seg_P[int(j*self.fs*0.5) : int((j+1)*self.fs*0.5)] = 0
                            seg_g[int(j*self.fs*0.5) : int((j+1)*self.fs*0.5)] = 0
                            
                    new_simi = self.cos_sim(seg_P, seg_g)
                    if new_simi > self.als_threshold*0.9:
                        self.similarity[-1] = 1
                        inhibition_bpm_data[int(self.stft_time[i] - self.fs * 2.5):int(self.stft_time[i] + self.fs * 2.5)] = seg_P
                        inhibition_g_sensor_data[int(self.stft_time[i] - self.fs * 2.5):int(self.stft_time[i] + self.fs * 2.5)] = seg_g
                        '''
                    
                    #method 11/3
                   
                        
                    list_diff = np.array(list_max)[1:] - np.array(list_max)[:-1]
                    cut_index = int(np.argmax(list_diff)*self.fs*0.5)
                    
                    new_simi_left = self.cos_sim(seg_P[:cut_index], seg_g[:cut_index])
                    new_simi_right = self.cos_sim(seg_P[cut_index:], seg_g[cut_index:])
                    
                    if new_simi_left > self.als_threshold*0.9 and new_simi_left > new_simi_right:
                        inhibition_bpm_data[int(self.stft_time[i] - self.fs * 2.5) + cut_index :int(self.stft_time[i] + self.fs * 2.5)] = 0  
                        #inhibition_g_sensor_data[int(self.stft_time[i] - self.fs * 2.5) + cut_index :int(self.stft_time[i] + self.fs * 2.5)] = 0    
                        self.similarity[-1] = 1

                        


        self.similarity = np.array(self.similarity)
        
        '''STFT for inhibition'''
        if self.USE_TIME_INHIBITION:
            _, T, s = scipy.signal.stft(inhibition_bpm_data, window='hamming', nperseg=self.fs * 5,
                                        noverlap=self.fs * 4, nfft=self.fs * self.fft_window_size, boundary=None)
            self.ss = np.abs(s)
            _, _, ns = scipy.signal.stft(inhibition_g_sensor_data, window='hamming', nperseg=self.fs * 5,
                                     noverlap=self.fs * 4, nfft=self.fs * self.fft_window_size, boundary=None)
        
            self.nss = np.abs(ns)

        self.remove_DC(self.ss, self.td_remove_DC_lower)
        self.remove_DC(self.nss, self.td_remove_DC_lower)
        self.overlapping_cutoff(self.ss, self.als_range_upper)
        self.overlapping_cutoff(self.nss, self.als_range_upper)
        self.normalization(self.ss)
        self.normalization(self.nss)
        '''Spectrum subtraction'''
        if self.USE_SPECTRUM_SUBTRACTION == 0:
            self.ass = self.ss
        else:
            self.ass, self.ss_status = self.get_ss(np.copy(self.ss), np.copy(self.nss), self.similarity, self.engine_noise)
        self.normalization(self.ass)

        '''Time_fusion'''
        if self.USE_TIME_FUSION:
            self.tss = self.time_fusion(self.filter_bpm_data, self.filter_g_sensor_data)
            self.result_ss = self.ass
            if self.USE_SPECTRUM_SUBTRACTION == 0:
                self.result_ss[:, self.similarity > self.als_threshold] = self.ss[:, self.similarity > self.als_threshold]

        else:
            self.result_ss = self.ass

        self.bpm_result_out = self.get_overlap(self.result_ss, self.overlap_weight)
        self.normalization(self.bpm_result_out)

        '''get_bpm_mix'''
        self.bpm_pre = self.get_bpm_final(np.copy(self.bpm_result_out), self.bpm_search_lower, self.bpm_search_upper)

        self.reliability = self.get_reliability(self.bpm_result_out, self.bpm_pre)
        self.status = self.get_status(self.filter_bpm_data, self.similarity, self.reliability, self.engine_denoise_list)

        if self.USE_MANUALLY_WEIGHT == 0:
            self.overlap_weight = self.get_overlap_list(self.reliability, self.bpm_pre)
            self.bpm_result_out = self.get_overlap(self.result_ss, self.overlap_weight)

        self.normalization(self.bpm_result_out)
        self.bpm_pre = self.get_bpm_final(np.copy(self.bpm_result_out), self.bpm_search_lower, self.bpm_search_upper,
                                          self.status)
        # self.bpm = self.bpm_pre
        # self.bpm = self._post_process_bpm(self.bpm_pre, self.status)
        self.bpm = self._post_process_rpm(self.bpm_pre)
        '''get rpm'''
        rpm_b, rpm_a = filter.butter_bandpass(self.rpm_cutoff_freq[0], self.rpm_cutoff_freq[1], self.fs,
                                              self.rpm_filter_order)
        data = data - data[0]
        self.rpm_data_out = lfilter(rpm_b, rpm_a, data)
        self.rpm_data_out = np.around(self.rpm_data_out)
        self.filter_rpm_data = np.copy(self.rpm_data_out)
        self.rpm_data_out = self.rpm_data_out[::2]
        _, _, rpm_s = scipy.signal.stft(self.rpm_data_out, window='hamming', nperseg=self.fs * 10, noverlap=self.fs * 8, nfft=self.fs * self.fft_window_size)

        self.rpm_s = np.abs(rpm_s[:, 1:-1])
        self.normalization(self.rpm_s)
        self.rpm_overlap = self.get_overlap(self.rpm_s, [1, 1])
        rpm_interval = self.dismantling(self.rpm_overlap, 3, 30)
        rpm_interval = rpm_interval.T
        self.rpm = self.get_rpm(rpm_interval, 3)
        self.rpm = self._post_process_rpm(self.rpm)
        rpm_list = self.rpm
        x = np.linspace(0, len(rpm_list) - 1, len(rpm_list))
        f = interp1d(x, rpm_list, kind='linear')
        x_new = np.linspace(0, len(rpm_list) - 1, len(rpm_list) * 5 - 4)
        self.rpm = f(x_new)

        return 0
