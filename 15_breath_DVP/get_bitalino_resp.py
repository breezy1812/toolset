import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from opensignalsreader import OpenSignalsReader

file_path='data\\20240125_test\\'
file_name='opensignals_84ba20545f12_2024-01-25_11-10-15.txt'

goldenfolder = 'golden\\'
column = 2



#get data and downsample
acq = OpenSignalsReader(file_path+ goldenfolder + file_name)
# method 1
analytic_signal = hilbert(acq.raw(column))
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# instantaneous_phase = instantaneous_phase[50::100]

# method 2
analytic_signal = acq.raw(column)
analytic_signal = analytic_signal - np.mean(analytic_signal)
instantaneous_phase_2 = []
buf = 0
for i in range(len(analytic_signal)):
    buf += analytic_signal[i]
    instantaneous_phase_2.append(buf)

instantaneous_phase_2 = instantaneous_phase_2 / max(instantaneous_phase_2)
# instantaneous_phase = instantaneous_phase[::100]

plt.figure()
plt.plot(instantaneous_phase)
plt.plot(instantaneous_phase_2)
plt.show()