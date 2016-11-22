import json
import time
import os
import collections
d=collections.OrderedDict()

d['GAMMA'] = 0.9 # discount factor for target Q

d['INITIAL_EPSILON'] = 0.35 # starting value of epsilon
d['FINAL_EPSILON'] = 0.01 # final value of epsilon

d['REPLAY_SIZE'] = 50000#10000 # experience replay buffer size
d['BATCH_SIZE'] = 16#32 # size of minibatch

d['STEP'] = 1200*2 # Step limitation in an episode
d['STEP_SIZE']=0.05/2# 函数试探的步长

d['BOUND_RANGE']=[100,100]
d['TEST_BATCH_SIZE']=100# size of cal_the_distance
d['OK_EPS']=1.0 #控制何时结束

d['GBEST_USED']=False
#
d['MODEL_SAVE']=True

# Hyper Parameters for training and test
d['EPISODE'] = 20000*2#10000#5000#10000 # Episode limitation
d['TEST'] = 10 # The number of experiment test every 100 episode

d['INITIAL_VAULE']=[0,100,0,0]
d['GOAL']=  [1.52546, 35.329, 0.250, 0.700]#[1.82546, 40.329, 1.250, 0.900]
d['DATA_FILE']='data.csv'
print(str(d))
print(hash(str(d)))
training_day=time.strftime("%d",time.localtime(time.time()))
exp_log_path='log/'+training_day
if not os.path.exists(exp_log_path):
    os.makedirs(exp_log_path)
instruction=input('have instruction? if not presss enter is just ok')
save_file_name_path=exp_log_path+'/'+str(hash(str(d)))+instruction
if not os.path.exists(save_file_name_path):
    os.makedirs(save_file_name_path)
with open('config'+'/'+'参数.json','w') as f:
    json.dump(d,f)


