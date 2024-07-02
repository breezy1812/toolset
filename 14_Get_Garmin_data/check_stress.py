import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.read_data import read_csv_with_time
import numpy as np
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

file_name_raw=['2023_11_27_14_32_19_Polar_GoldenHR.csv',
               '2023_12_01_13_29_04_Polar_GoldenHR.csv'
]

goldenfolder = 'C:\\Users\\Dell\Documents\\toolset\\14_Get_Garmin_data\\'
rawfolder = 'C:\\Users\\Dell\Documents\\toolset\\14_Get_Garmin_data\\raw\\'
column = 6
time_raw = []
time_garmin = []
stress_garmin = []

time_RMSSD = []
data_RMSSD = []
data_HR = []
data_SDNN =[]
data_garmin = []



# 取得資料夾中的所有檔案名稱以及建立時間
file_info = [(f, os.path.getctime(os.path.join(goldenfolder, f))) for f in os.listdir(goldenfolder) if f.endswith('.txt')]

# 按照建立時間排序
file_info.sort(key=lambda x: x[1])

# 取得排序後的檔案名稱
file_names = [info[0] for info in file_info]

for i in range(2):
    filename = file_names[i]
    #get data and downsample
    HR, start_time = read_csv_with_time( rawfolder + file_name_raw[i] )
    timeraw = datetime(*start_time)
    timeraw = timeraw.replace(year=1900,month=1, day=1)

    filename = os.path.join(goldenfolder, filename)
    with open(filename, 'r') as garmin:
        for line in garmin.readlines():
            line = line.split(',')
            if (int(line[2]) <= 5):
                continue
            t = datetime.strptime(line[1], "%H:%M")
            stress_garmin.append(int(line[2]))
            time_garmin.append(t)

    indmin = np.argmin([abs(i - timeraw) for i in time_garmin])

    time_garmin = time_garmin[indmin:]
    stress_garmin = stress_garmin[indmin:]


    # counter = 0
    buf = []
    wind = 150
    window = timedelta(seconds = wind)

    for t in time_garmin:
        if t - window > timeraw and t + window < timeraw + timedelta(seconds = len(HR)):
            ind = int((t - timeraw).total_seconds())
            buf = HR[ind - wind : ind + wind]
            buf = buf[buf !=0]

            mean_HR = np.mean(buf)
            rri = np.array([int(60000/i) for i in buf])
            RMSSD = np.sqrt(np.mean( np.power(rri[1:] - rri[:-1],2)))
            SDNN = np.std(rri)
            data_RMSSD.append(RMSSD)
            data_HR.append(mean_HR)
            data_SDNN.append(SDNN)
            time_RMSSD.append(t)
            data_garmin.append(stress_garmin[time_garmin.index(t)])

# data_RMSSD = 60 - (np.array(data_RMSSD))/5
# for i in range(len(HR)):
#     T = timeraw + timedelta(seconds = i )
#     time_raw.append(T)
#     buf.append(60000 / HR[i])
#     if i % 60 == 0 and i > 300:
#         time_RMSSD.append (T)
#         buf = np.array(buf)
#         RMSSD = np.mean( np.power(buf[1:] - buf[:-1],2))
#         SDNN = np.std(buf)
#         # data_RMSSD.append(np.mean(HR[i: i +60]) - np.mean(HR))
#         data_RMSSD.append(60 - (RMSSD)/5)
#         buf = []

corr_coef_1 = np.corrcoef(data_HR, data_garmin)[0, 1]
slope_1, intercept_1 = np.polyfit( data_HR,data_garmin, 1)

corr_coef_2 = np.corrcoef(data_garmin, data_RMSSD)[0, 1]
slope_2, intercept_2 = np.polyfit(data_RMSSD, data_garmin, 1)

corr_coef_3 = np.corrcoef(data_garmin, data_SDNN)[0, 1]
slope_3, intercept_3 = np.polyfit(data_SDNN, data_garmin, 1)

X = np.column_stack((data_HR, data_RMSSD, data_SDNN))
coefficients, residuals, _, _ = np.linalg.lstsq(X, data_garmin)
reg = LinearRegression().fit(X, data_garmin)

# 创建一个多项式模型
# poly_model = np.poly1d(coefficients)

# 預測 Y 值

y_pred = reg.predict(X)
corr_coef_4 = np.corrcoef(data_garmin, y_pred)[0, 1]
coefficients = reg.coef_
intercept = reg.intercept_
print("coefficients:", coefficients)
print("intercept:", intercept)


X1 = np.arange(min(data_HR), max(data_HR))
X2 = np.arange(min(data_RMSSD), max(data_RMSSD))
X3 = np.arange(min(data_SDNN), max(data_SDNN))
plt.figure()

plt.subplot(2,2 ,1)
sns.scatterplot(y=data_garmin, x=data_HR)
plt.title(f'HR Correlation: {corr_coef_1:.2f}')
plt.plot(X1, slope_1 * X1 + intercept_1, color='red')

# 繪製第二張散布圖
plt.subplot(2,2 , 2)
sns.scatterplot(y=data_garmin, x=data_RMSSD)
plt.title(f'RMSSD Correlation: {corr_coef_2:.2f}')
plt.plot(X2, slope_2 * X2 + intercept_2, color='red')

# 繪製第三張散布圖
plt.subplot(2,2 , 3)
sns.scatterplot(y=data_garmin, x=data_SDNN)
plt.title(f'SDNN Correlation: {corr_coef_3:.2f}')
plt.plot(X3, slope_3 * X3 + intercept_3, color='red')

# 繪製第四張散布圖
plt.subplot(2,2 , 4)
sns.scatterplot(y=data_garmin, x=y_pred)
plt.title(f'linear model Correlation: {corr_coef_4:.2f}')
plt.plot(y_pred, y_pred, color='red')

plt.tight_layout()

plt.figure()
plt.plot(time_RMSSD, data_garmin)
plt.plot(time_RMSSD, y_pred)



plt.figure()
# coefficients = [1.146, 0.16, -0.48]
# y_pred = np.dot(X, coefficients) -46

coefficients = [1.14616865,  0.1602979,  -0.0486112]

y_pred = np.dot(X, coefficients) + 22.1
plt.plot(time_RMSSD, data_garmin)
plt.plot(time_RMSSD, y_pred)

plt.show()
