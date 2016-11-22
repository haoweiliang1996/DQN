#生成相应表达式的数据 保存为csv形式
from matplotlib import  pyplot as plt
import csv
import os
import json

with open('../config/参数.json','r') as f:
    CONFIG=json.load(f)
arx=[i/20000 for i in range(20000)]
a1,a2,r1,r2=CONFIG['GOAL']
formulation = lambda x: ((a2 * ((r2 - x) ** 3)) + (a1 * ((r1 - x) ** 3))) * (x < r2) + a1 * ((r1 - x) ** 3) * (
    x > r2 and x < r1)
ary=list(map(formulation,arx))
arpair=[arx,ary]

if not os.path.exists('input_data'):
    os.makedirs('input_data')

with open('input_data/data.csv','w') as f:
    csvW=csv.writer(f)
    csvW.writerows(arpair)
plt.scatter(arx,ary)
plt.show()
