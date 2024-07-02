import os
import sys
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ACC_MAX = 65536
PRESSURE_MAX = 16777216
RAW_DATA_PLOT = 0
HEART_RATE_PLOT = 0
decode_status = 0 #(0) OK (1)1.4 to 1.6 (2) 1.6 to 1.4 


if __name__ == '__main__':
    if len(sys.argv) != 2:
        SAMPLING_RATE = 64
    else:
        SAMPLING_RATE = int(sys.argv[1])

    infile_dir = r'.\01_input\05_decode'
    outfile_dir = r'.\02_output\05_decode'
    if not os.path.exists(infile_dir):
        os.makedirs(infile_dir)
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)

    filenames = os.listdir(infile_dir)
    for filename in filenames:
        decode_status = 0
        with open(os.path.join(infile_dir, filename), 'r', encoding='utf-8') as decode_input:
            origin_data = []
            origin_text = []
            modified_data = []
            
            ### Get raw data
            for line in decode_input.readlines(): 
                origin_text.append(line)                                      
                temp = line.split(',') 
                if len(temp) > 4:
                    origin_data.append(int(temp[1]))
            origin_data = np.array(origin_data)
            
            ### check decode status
            max_diff = abs(max(origin_data[1:] - origin_data[:-1]))
            if  max_diff > 16000216:
                if max(origin_data) < PRESSURE_MAX / 2 and min(origin_data) < 0:
                    decode_status = 1
                else: 
                    decode_status = 2
            print(filename)
            print("processing status = " + str(decode_status))  
            
            
            ### fix decode 
            for data in origin_data:
                if decode_status == 1:
                    if data <= 0:
                        data += (PRESSURE_MAX-1)
                if decode_status == 2:
                    if data >= PRESSURE_MAX/2:
                        data -= (PRESSURE_MAX-1)
                modified_data.append(data)
                
                
            ### save the modified data
            with open(os.path.join(outfile_dir, filename), 'w', encoding='utf-8') as decode_output:
                i = 0
                for line in  origin_text:
                    temp = line.split(',') 
                    if len(temp) > 4:
                        temp[1] = str(modified_data[i])
                        line = ','.join(temp)
                        i += 1
                        
                        
                    decode_output.write(line + '\n')
                
            
        plt.figure()
        plt.title(filename)
        plt.subplot(2,1,1)
        plt.plot(range(len(origin_data)), origin_data, color='blue')
        plt.subplot(2,1,2)
        plt.plot(range(len(origin_data)), modified_data, color='red')
        plt.show()