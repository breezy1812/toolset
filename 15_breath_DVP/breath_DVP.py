import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from opensignalsreader import OpenSignalsReader
from src.read_data import read_pressure_acc_result
from datetime import datetime
from scipy.signal import find_peaks

file_path='data\\20231128_breath_DVP\\'
file_name='opensignals_84ba20545f12_2023-11-28_09-45-10.txt'
file_name_raw='2023_11_28_09_45_18.log'

goldenfolder = 'golden\\'
rawfolder = 'raw\\'
column = 6



#get data and downsample
_, _, _, _, _, _,  _, resp, start_time = read_pressure_acc_result(file_path +rawfolder, file_name_raw, 64)
acq = OpenSignalsReader(file_path+ goldenfolder + file_name)
# method 1
# analytic_signal = hilbert(acq.raw(column))
# instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# instantaneous_phase = instantaneous_phase[50::100]

# method 2
analytic_signal = acq.raw(column) - 28
instantaneous_phase = []
buf = 0
for i in range(len(analytic_signal)):
    buf += analytic_signal[i]
    instantaneous_phase.append(buf)

instantaneous_phase = instantaneous_phase / max(instantaneous_phase)
instantaneous_phase = instantaneous_phase[50::100]

# compute the delta time and calibration
timeacq = datetime.strptime(acq.info['time'], "%H:%M:%S.%f")
timeraw = datetime(*start_time)
delta = (timeraw - timeacq).seconds + 1
instantaneous_phase = instantaneous_phase[delta :]
print('delta_time = ' + str(delta) + 's')

# check the deep breathing start time(start_sec)
peaks,_ = find_peaks(instantaneous_phase)
start_ind = 0
for i in peaks[1:] - peaks[:-1]:
    if abs(10 - i) <= 2 :
        break
    else:
        start_ind += 1
start_sec = peaks[start_ind]
print('start_time = ' + str(start_sec) + 's')

count = 0 
score = 0
serial = np.arange(start_sec, min(start_sec + 120, len(resp)), 1)
# for i in range(min(len(instantaneous_phase), len(resp)) - 1):
for i in serial:
    count += 1
    if instantaneous_phase[i+1] - instantaneous_phase[i] > 0:
        if resp[i] == 1:
            score += 1
    else:
        if resp[i] == -1:
            score += 1




print('percision = ' + str(score / count))

plt.figure()
# plt.plot(acq.raw(rawnum)-500)
plt.plot(instantaneous_phase)
plt.plot(resp)
plt.show()