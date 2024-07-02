import numpy as np


def get_threshold(RT_serial, _type = "Wilcoxon", bins = []):
    mini = max(min(RT_serial), 300)
    maxi = min(max(RT_serial), 1000)
    if len(bins) == 0:
        thresholds = np.arange(mini, maxi, (maxi - mini)/100)
        thresholds = thresholds[1:]
    else:
        thresholds = bins
    ranks = np.arange(len(RT_serial)) + 1
    list_Z = []
    for i in range(len(thresholds)):
        
        group1 = RT_serial[RT_serial < thresholds[i]]
        group2 = RT_serial[RT_serial >=thresholds[i]]
        R1_group = ranks[RT_serial < thresholds[i]]
        R2_group = ranks[RT_serial >=thresholds[i]]
        
        R1 = sum(R1_group)
        R2 = sum(R2_group)
        N1 = len(group1)
        N2 = len(group2)
        if N1 < len(RT_serial) / 100 or N2 < len(RT_serial) / 100 :
            list_Z.append(0)
            continue
        if _type == "Wilcoxon":
            u = N2*(N1 + N2 + 1) / 2
            sigma = (N1*N2*(N1 + N2 + 1)/12)**0.5
            Z = (R2 - u)/sigma
            list_Z.append(abs(Z))
        elif _type == "M-W":
            U = N1*N2 + (N2*(N2 + 1) / 2) -R2
            u = N1*N2 / 2
            sigma = (N1*N2*(N1 + N2 + 1)/12)**0.5
            Z = (U - u)/sigma
            list_Z.append(abs(Z))
    list_Z = np.array(list_Z)
    list_Z[list_Z < 1.96] = 0
    return [thresholds,list_Z], thresholds[np.argmax(list_Z)]

def get_Zvalue(data1, data2, _type = "Wilcoxon" ):
    N1 = len(data1)
    N2 = len(data2)

    buffer = [ [i , 1] for i in data1] + [ [i , 2] for i in data2]
    buffer = np.array(buffer)
    #group_all = data1 + data2
    group_all = sorted(buffer, key=lambda x: x[0])

    R1 = 0
    R2 = 0
    for i in range(len(group_all)):
        if group_all[i][1] == 1:
            R1 += i
        else:
            R2 += i
    if _type == "Wilcoxon":
        u = N1*(N1 + N2 + 1) / 2
        sigma = (N1*N2*(N1 + N2 + 1)/12)**0.5
        Z = (R1 - u)/sigma
        
    elif _type == "M-W":
        U = N1*N2 + (N1*(N1 + 1) / 2) -R1
        u = N1*N2 / 2
        sigma = (N1*N2*(N1 + N2 + 1)/12)**0.5
        Z = (U - u)/sigma
    return Z


    






