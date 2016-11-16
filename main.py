import tensorflow as tf
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import seaborn
import os
import json
import  collections
import logging
import sys

CONFIG=collections.OrderedDict()
start_time=''
exp_log_path='log'
this_exp_log_path=''
training_day={'month':time.strftime("%m",time.localtime(time.time())),'day':time.strftime("%d",time.localtime(time.time()))}

logger=logging.getLogger()

def init_the_exp():
  global start_time
  start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime(time.time()))
  global  this_exp_log_path
  this_exp_log_path=exp_log_path+'/'+training_day['month']+'/'+training_day['day']+'/'+start_time
  if os.path.exists(this_exp_log_path):
    print('alreday exist')
  else:
    os.makedirs(this_exp_log_path)

  congif_file_path = 'config/参数.json'
  with open(congif_file_path,'r') as f:
      CONFIG=json.load(f)

  # 将log输出至文件与控制台
  console = logging.StreamHandler(stream=sys.stdout)
  formatter = logging.Formatter('%(message)s')
  fh = logging.FileHandler(this_exp_log_path + '/log.log')
  fh.setFormatter(formatter)
  console.setFormatter(formatter)
  logger.addHandler(console)
  logger.addHandler(fh)
  logger.setLevel(logging.INFO)

  logger.info('load the config')
  logger.info("path:{}".format(this_exp_log_path))
  logger.info("start time:{}".format(start_time))

  #print("episode:{},stepTimes:{},stepSize:{},gbest: {},Ok_eps:{},init_state: {},agent_saved:{}".format(EPISODE,STEP,STEP_SIZE,GBEST_USED,OK_EPS,env.initVaule,MODEL_SAVE),"range:",0)

  return CONFIG

CONFIG=init_the_exp()
logger.info(str(CONFIG))
# Hyper Parameters for DQN
GAMMA = CONFIG['GAMMA'] # discount factor for target Q

INITIAL_EPSILON = CONFIG['INITIAL_EPSILON'] # starting value of epsilon
FINAL_EPSILON = CONFIG['FINAL_EPSILON'] # final value of epsilon

REPLAY_SIZE = CONFIG['REPLAY_SIZE']#10000 # experience replay buffer size
BATCH_SIZE = CONFIG['BATCH_SIZE']#32 # size of minibatch

STEP = CONFIG['STEP']#1200*2 # Step limitation in an episode
STEP_SIZE=CONFIG['STEP_SIZE']#0.05/2# 函数试探的步长

BOUND_RANGE=CONFIG['BOUND_RANGE']#[100,100]
TEST_BATCH_SIZE=CONFIG['TEST_BATCH_SIZE']# size of cal_the_distance
OK_EPS=CONFIG['OK_EPS'] #控制何时结束

GBEST_USED=CONFIG['GBEST_USED']
#
MODEL_SAVE=CONFIG['MODEL_SAVE']
###
INITIAL_VAULE=CONFIG['INITIAL_VAULE']

plt.ion()
fig=plt.figure(figsize=(3,4))
tmp=np.loadtxt("data.csv",dtype="double",delimiter=",")
data_set=tmp.transpose().tolist()
data_set_dis=tmp.tolist()
datax = data_set_dis[0]
logger.info('fig size: {0} DPI, size in inches {1}'.format(
  fig.get_dpi(), fig.get_size_inches()))

#res_output=[]
res_output_short=[]
res_reward_output=[]
###

def end_the_exp(agent,input_LIST):
  over_time=time.strftime("%m-%d-%H-%M-%S", time.localtime(time.time()))
  logger.info("over time:{}".format(over_time))
  global  this_exp_log_path
  with open(this_exp_log_path+'/'+'参数.json','w') as f:
    json.dump(CONFIG,f)
    logger.info('CONFIG saved')
  if MODEL_SAVE:
    agent.saver.save(agent.session, this_exp_log_path+'/'+"model.ckpt")
    logger.info('MODEL saved')
  import draw_a_gif
  draw_a_gif.save_the_pic(this_exp_log_path,input_LIST)
  logger.info('gif drawed')

  logger.info(res_output_short)
  logger.info(res_reward_output)


class myEny():
  def __init__(self):

    tmp = np.loadtxt("data.csv", dtype="double", delimiter=",")
    self.data_set = tmp.transpose().tolist()
    self.data_set_dis = tmp.tolist()

    class observation_space():
      def __init__(self):
        self.shape=[4]
    self.observation_space=observation_space()

    class action_space():
      def __init__(self):
        self.n=3**4#self.observation_space.shape[0]^3
        self.all_space=[]
        for i in range(3):
          for j in range(3):
            for k in range(3):
              for l in range(3):
                self.all_space.append(list(map(lambda x:STEP_SIZE*x,[i-1, j-1, k-1, l-1])))
    self.action_space=action_space()

    self.initVaule=INITIAL_VAULE#[0.1,0.1,0.1,0.1]#[0.4,35,0.2,0.3]#[1,35,1,1]#[0.7,35,0.7,0.7]#np.random.normal(size=self.observation_space.shape[0]).tolist()

    self.state=self.reset()#np.zeros(self.observation_space.shape[0]).tolist()
    self.formulation =lambda a1,a2,r1,r2,x: ((a2*((r2-x)**3))+(a1*((r1-x)**3)))*(x<r2)+a1*((r1-x)**3)*(x>r2 and x<r1)
    self.initDistance = self.isok()
    self.gbest_dis=self.initDistance
    logger.info("env init ok")



  def calDistance(self,state):
    test_batch=random.sample(self.data_set,TEST_BATCH_SIZE)
    tot=0
    for i in range(TEST_BATCH_SIZE):
      now_batch=test_batch[i]
      tot+=((self.formulation(state[0],state[1],state[2],state[3],now_batch[0])-now_batch[1])**2)  #state 0->3
    return tot/(2*TEST_BATCH_SIZE)


  def reset(self):
    self.state = self.initVaule
    return self.state
  def out_bound(self):
    flag=self.state[0]<0 or self.state[1]<0 or self.state[2]<0 or self.state[2]>10 or self.state[3]<0 or self.state[3]>10
    #flag=self.state[0]<0 or self.state[0]>BOUND_RANGE[0] or self.state[1]<0 or self.state[1]>BOUND_RANGE[1] or self.state[2]<0 or self.state[2]>10 or self.state[3]<0 or self.state[3]>10
    #flag=self.state[0]<0 or self.state[0]>30 or self.state[1]<30 or self.state[1]>50 or self.state[2]<0 or self.state[2]>10 or self.state[3]<0 or self.state[3]>10
    return flag

  def step(self,action):
    next_state=list(map(lambda x,y:x+y,self.state,self.action_space.all_space[action]))
    now_distance=self.calDistance(next_state)
    #reward=-1*(self.calDistance(next_state)-now_distance)
    reward = 1 * ((self.calDistance(next_state) - self.calDistance(self.state)) < 0)  #一步一分式
    self.state=next_state
    done=False
    info='No info'#只是为了和gym一样
    if now_distance>10*self.initDistance or self.out_bound():
      done=True
      reward=-5*abs(reward)
    return next_state,reward,done,info
  def isok(self):
    kount=len(self.data_set)
    tot=0

    for i in range(kount):
      now_batch = self.data_set[i]
      tot += ((self.formulation(self.state[0], self.state[1], self.state[2], self.state[3], now_batch[0]) - now_batch[1]) ** 2)

    return tot/(2*kount)


class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

    self.create_Q_network()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()
    self.saver = tf.train.Saver()

    sess=self.session
    model_load_path = "/home/haowei/PycharmProjects/DQN/model/model.ckpt"
    if not os.path.exists(model_load_path):
      sess.run(tf.initialize_all_variables())
      #print('model saved in file %s' %model_save_path)
    else:
      self.saver.restore(sess,model_load_path)
      logger.info('model restores by file %s' %model_load_path)



  def create_Q_network(self):
    # network weights
    W1 = self.weight_variable([self.state_dim,20],"W1")
    b1 = self.bias_variable([20],"b1")
    W2 = self.weight_variable([20,self.action_dim],"W2")
    b2 = self.bias_variable([self.action_dim],"b2")
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
    # hidden layers
    #h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    h_layer=tf.sigmoid(tf.matmul(self.state_input,W1) + b1)
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2

  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):

    self.time_step += 1
    minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
    for i in range(0, BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else:
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
    })

  def egreedy_action(self,state):
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
    })[0]
    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / (EPISODE*STEP)
    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)
    else:
      return np.argmax(Q_value)



  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
    })[0])

  def weight_variable(self,shape,name):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial,name=name)

  def bias_variable(self,shape,name):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial,name=name)
# ---------------------------------------------------------
# Hyper Parameters for training and test
EPISODE = CONFIG['EPISODE']#20000*2#10000#5000#10000 # Episode limitation
TEST = CONFIG['TEST']#10 # The number of experiment test every 100 episode

def main():
  env=myEny()
  agent = DQN(env)

  # init_the_exp()
  goal = [1.82546, 40.329, 1.250, 0.900]
  def drawpic(ll=[],label=""):
    a1, a2, r1, r2 = ll
    formulation = lambda x: ((a2 * ((r2 - x) ** 3)) + (a1 * ((r1 - x) ** 3))) * (x < r2) + a1 * ((r1 - x) ** 3) * (
      x > r2 and x < r1)
    datay = list(map(formulation, datax))
    a1, a2, r1, r2 = goal
    goaly = list(map(formulation, datax))
    plt.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(label)
    plt.scatter(datax, goaly, color='red')
    plt.scatter(datax, datay, color='blue')
    #plt.axis([-0.2, 1.6, -5, 50])
    plt.pause(0.05)

  over=False
  for episode in range(EPISODE):
    # initialize task
    state = env.reset()
    if over==True:
      break
    # Train
    for step in range(STEP):   #step(30)
      action = agent.egreedy_action(state) # e-greedy action for train   argmax (n=4**3)
      next_state,reward,done,_ = env.step(action)   #random2000->100 ,reward    rewardBatchSize(=100)
      # Define reward for agent

      agent.perceive(state,action,reward,next_state,done)  # train network BATHSIZE(=32)*network size 4*20+20*4**3    32*5200  1.6*10**5;   *30~=5*10**6 per episode
      state = next_state
      if done:
        break
    EPISIZE=100 # Test every 100 episodes
    if episode % EPISIZE == 0:
      total_reward = 0
      total_distance=0
      mini_distance=1000000
      mini_state=[0,0,0,0]#env.initVaule
      if over==True:
        break
      cnt=0
      cnti=0
      for i in range(TEST):
        cnti+=1
        state = env.reset()
        pre_state = state
        for j in range(STEP):
          #env.render()
          cnt+=1
          action = agent.action(state) # direct action for test
          pre_state=state
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            env.state=pre_state
            break

        temp_ok = env.isok()
        total_distance+=temp_ok
        #更新目前最优
        if GBEST_USED and temp_ok < env.gbest_dis and (not env.out_bound()):
          env.gbest_dis = temp_ok
          env.initVaule = env.state
          logger.info("gbest:{},gstatue:{}".format(env.gbest_dis, env.initVaule))
        if temp_ok<mini_distance:
          mini_distance=temp_ok
          mini_state=env.state
        if temp_ok < OK_EPS:  # env.calDistance(env.state)<10:
          logger.info("a1,a2,r1,r2:" + str(env.state) + "\n")
          over=True
          break
      ave_reward = total_reward/cnt
      ave_distance=total_distance/(cnti)
      res_reward_output.append(mini_distance)
      res_output_short.append(mini_state)

      #logger.info ('episode: ',episode,'Evaluation Average Reward:',ave_reward,"avg distance",ave_distance,"mini_state:",mini_state,"mini_distance:",mini_distance)
      logger.info('episode:{} ave_reward:{} ave_distance:{} mini_state:{} minidistance:{}'.format(episode,ave_reward,ave_distance,mini_state,mini_distance))
      drawpic(env.state,episode)
  end_the_exp(agent=agent,input_LIST=res_output_short)
if __name__ == '__main__':
  main()