"""
作者：zhaozhijie
日期：2022年05月19日11时02分
"""
import numpy as np

data = '/Users/zhaozhijie/PycharmProjects/VRP/heterogenerous/时间修正/80需求点/data_GRASP.txt'
with open(data, 'r', encoding='utf-8') as file:
    result_data = file.readlines()
    bus = []
    dis = []
    time = []
    cpu_time = []
    GRASP = []
    GRASP_VND = []
    for line in range(1, 11):
        l = result_data[line].split(',')
        bus.append(float(l[0]))
        dis.append(float(l[1]))
        time.append(float(l[2]))
        cpu_time.append(float(l[3]))
        try:
            GRASP.append(float(l[4]))
            GRASP_VND.append(float(l[5]))
        except ValueError:
            pass


def Standard_Deviation(data: list):
    mean = np.mean(data)
    var = np.var(data)
    dev = np.std(data, ddof=1)
    print(mean, var, dev)


with open(data, 'a', encoding='utf-8') as f:
    txt = []
    txt.append(str(sum(bus)/10)+',')
    txt.append(str(sum(dis)/10)+',')
    txt.append(str(sum(time)/10)+',')
    txt.append(str(round(sum(cpu_time)/10, 2))+',')
    txt.append(str(sum(GRASP)/10)+',')
    txt.append(str(sum(GRASP_VND)/10)+',')
    try:
        txt.append(str(min(GRASP))+',')
    except ValueError:
        txt.append('nan'+',')
    try:
        txt.append(str(min(GRASP_VND))+',')
    except ValueError:
        txt.append('nan'+',')
    txt.append(str(round(np.std(bus, ddof=1), 2))+',')
    txt.append(str(round(np.std(dis, ddof=1), 2))+',')
    f.write('\n')
    f.writelines(txt)