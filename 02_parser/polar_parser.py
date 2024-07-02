import os
import pandas as pd

if __name__ == '__main__':
    in_directory = r'.\01_input\02_Polar'
    out_directory = r'.\02_output\02_Polar'
    for filename in os.listdir(in_directory):
        if os.path.splitext(filename)[1] != '.txt' and os.path.splitext(filename)[1] != '.log' and os.path.splitext(filename)[1] != '.CSV':
            continue

        in_filename = os.path.join(in_directory, filename)
        out_filename = os.path.join(out_directory, filename)

        col_num = 2
        in_data = pd.read_csv(in_filename, header = None, skiprows = 3)
        out_data = in_data.iloc[:, col_num]
        with open(out_filename, 'w') as out_fid:
            for i in range(len(out_data)):
                out_fid.write(str(out_data[i]) + '\n')