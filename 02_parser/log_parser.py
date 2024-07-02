import os
import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ACC_MAX = 65536
PRESSURE_MAX = 16777216
RAW_DATA_PLOT = 0
HEART_RATE_PLOT = 0

if __name__ == '__main__':
    if len(sys.argv) != 2:
        SAMPLING_RATE = 64
    else:
        SAMPLING_RATE = int(sys.argv[1])

    HEADER_TRAILER = ['55', 'BC', '66']
    ONE_UNIT_LEN = 18

    infile_dir = r'.\01_input\01_BCG'
    outfile_dir = r'.\02_output\01_BCG'

    filenames = os.listdir(infile_dir)
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext != '.txt' and ext != '.csv' and ext != '.log':
            continue
        
        with open(os.path.join(infile_dir, filename)) as file_in:            
            lines = file_in.read()
            lines = lines.replace('\n', '')
            lines = lines.replace(' ', '')

        i = 0
        outfilename = os.path.join(outfile_dir, filename)
        with open(outfilename, 'w') as file_out:
            item_index = 0
            time_buffer = []
            pressure_buffer = []
            heart_rate_buffer = []
            resp_rate_buffer = []
            acc_x_buffer = []
            acc_y_buffer = []
            acc_z_buffer = []
            time_counter = 0
            while i <= len(lines) // 2 - ONE_UNIT_LEN:
                if lines[i * 2: (i + 1) * 2] == HEADER_TRAILER[0] and lines[(i + 1) * 2: (i + 2) * 2] == HEADER_TRAILER[1] and lines[(i + 17) * 2: (i + 18) * 2] == HEADER_TRAILER[2]:
                    pressure = lines[(i + 4) * 2: (i + 5) * 2] + lines[(i + 3) * 2: (i + 4) * 2] + lines[(i + 2) * 2: (i + 3) * 2]
                    heart_rate = lines[(i + 5) * 2: (i + 6) * 2]
                    resp_rate = lines[(i + 6) * 2: (i + 7) * 2]
                    curve_state = lines[(i + 7) * 2: (i + 8) * 2]
                    b_pressure = lines[(i + 10) * 2: (i + 11) * 2] + lines[(i + 9) * 2: (i + 10) * 2] + lines[(i + 8) * 2: (i + 9) * 2]
                    r_pressure = lines[(i + 13) * 2: (i + 14) * 2] + lines[(i + 12) * 2: (i + 13) * 2] + lines[(i + 11) * 2: (i + 12) * 2]
                    #acc_x = lines[(i + 9) * 2: (i + 10) * 2] + lines[(i + 8) * 2: (i + 9) * 2]
                    #acc_y = lines[(i + 11) * 2: (i + 12) * 2] + lines[(i + 10) * 2: (i + 11) * 2]
                    #acc_z = lines[(i + 13) * 2: (i + 14) * 2] + lines[(i + 12) * 2: (i + 13) * 2]
                    timestamp = lines[(i + 16) * 2: (i + 17) * 2] + lines[(i + 15) * 2: (i + 16) * 2] + lines[(i + 14) * 2: (i + 15) * 2]

                    pressure = int(pressure, 16)
                    heart_rate = int(heart_rate, 16)
                    resp_rate = int(resp_rate, 16)
                    curve_state = int(curve_state, 16)
                    signal_state = (curve_state >> 4) & 0x0F
                    curve_state &= 0x0F
                    b_pressure = int(b_pressure, 16)
                    #b_pressure = struct.unpack('!f', bytes.fromhex(b_pressure))[0]
                    r_pressure = int(r_pressure, 16)
                    #acc_x = int(acc_x, 16)
                    #acc_y = int(acc_y, 16)
                    #acc_z = int(acc_z, 16)
                    timestamp = int(timestamp, 16)
                
#                    if acc_x > ACC_MAX / 2 - 1:
#                        acc_x -= ACC_MAX
#                    if acc_y > ACC_MAX / 2 - 1:
#                        acc_y -= ACC_MAX
#                    if acc_z > ACC_MAX / 2 - 1:
#                        acc_z -= ACC_MAX
                        
                    if pressure > PRESSURE_MAX / 2 - 1:
                        pressure -= PRESSURE_MAX
                    if b_pressure > PRESSURE_MAX / 2 - 1:
                        b_pressure -= PRESSURE_MAX
                    if r_pressure > PRESSURE_MAX / 2 - 1:
                        r_pressure -= PRESSURE_MAX

                    time_buffer.append(time_counter / SAMPLING_RATE)
                    pressure_buffer.append(pressure)
                    heart_rate_buffer.append(heart_rate)
                    resp_rate_buffer.append(resp_rate)
                    #acc_x_buffer.append(acc_x)
                    #acc_y_buffer.append(acc_y)
                    #acc_z_buffer.append(acc_z)
                    file_out.write(str(timestamp) + ', ' + str(pressure) + ', ' + str(b_pressure) + ', ' + str(r_pressure) + ', ' + str(heart_rate) + ', ' + str(resp_rate) + ', ' + str(signal_state) + ', ' + str(curve_state) + '\n')
                    #file_out.write(str(timestamp) + ', ' + str(pressure) + ', ' + str(acc_x) + ', ' + str(acc_y) + ', ' + str(acc_z) + ', ' + str(heart_rate) + ', ' + str(resp_rate) + '\n')
                    i = i + ONE_UNIT_LEN
                    time_counter += 1
                else:
                    i = i + 1

        if RAW_DATA_PLOT == 1:
            plt.figure(figsize = (20, 20))
            ax = plt.plot(time_buffer, pressure_buffer)
            plt.xlabel('Time (s)', fontsize = 20)
            plt.ylabel('Quantization', fontsize = 20)
            plt.title(filename)
            plt.axis('tight')
            ax = plt.gca()
            fmt = '%1.2e'
            plt.xticks(fontsize = 30)
            plt.yticks(fontsize = 30)
            yticks = mtick.FormatStrFormatter(fmt)
            ax.yaxis.set_major_formatter(yticks)
            plt.savefig(os.path.splitext(outfilename)[0] + '.png')
            #plt.show()

        if HEART_RATE_PLOT == 1:
            plt.figure(figsize = (20, 20))
            ax = plt.scatter(time_buffer, heart_rate_buffer)
            plt.xlabel('Time (s)', fontsize = 20)
            plt.ylabel('Heart Rate', fontsize = 20)
            plt.title(filename, {'fontsize': 30})
            plt.axis('tight')
            plt.axis([0, 120, 55, 85])
            plt.xticks(fontsize = 30)
            plt.yticks(fontsize = 30)
            plt.savefig(os.path.splitext(outfilename)[0] + '_hr.png')