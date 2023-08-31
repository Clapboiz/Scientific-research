# Deep Q Learning / Keras / OpenAI / Frozen Lake / Not Slippery / 5x5
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import os

from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
from tensorflow.keras.layers import Dropout

env = gym.make("FrozenLake-v1")
train_episodes=400
test_episodes=100
max_steps=750
state_size = env.observation_space.n
action_size = env.action_space.n
batch_size=3000

class Agent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2500)
        self.learning_rate=0.8
        self.epsilon=1
        self.max_eps=1
        self.min_eps=0.01
        self.eps_decay = 0.001/3
        self.gamma=0.9
        self.state_size= state_size
        self.action_size= action_size
        self.epsilon_lst=[]
        self.model = self.buildmodel()

    def buildmodel(self):
        model=Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, new_state, reward, done, state, action):
        self.memory.append((new_state, reward, done, state, action))

    def action(self, state):
        if np.random.rand() > self.epsilon:
            return np.random.randint(0,4)
        return np.argmax(self.model.predict(state))

    def pred(self, state):
        return np.argmax(self.model.predict(state))

    def replay(self,batch_size):
        minibatch=random.sample(self.memory, batch_size)
        for new_state, reward, done, state, action in minibatch:
            target= reward
            if not done:
                target=reward + self.gamma* np.amax(self.model.predict(new_state))
            target_f= self.model.predict(state)
            target_f[0][action]= target
            self.model.fit(state, target_f, epochs=3, verbose=0)

        if self.epsilon > self.min_eps:
            self.epsilon=(self.max_eps - self.min_eps) * np.exp(-self.eps_decay*episode) + self.min_eps

        self.epsilon_lst.append(self.epsilon)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

agent=Agent(state_size, action_size)

reward_lst=[]
for episode in range(train_episodes):
    state= env.reset()
    state_arr=np.zeros(state_size)
    state_arr[int(state[0])] = 1
    state= np.reshape(state_arr, [1, state_size])
    reward = 0
    done = False
    for t in range(max_steps):
        # env.render()
        action = agent.action(state)
        result = env.step(action)
        new_state = result[0]
        reward += result[1]  # Cộng dồn phần thưởng
        done = result[2]
        new_state_arr = np.zeros(state_size)
        new_state_arr[new_state] = 1
        new_state = np.reshape(new_state_arr, [1, state_size])
        agent.add_memory(new_state, reward, done, state, action)
        state= new_state

        if done:
            print(f'Episode: {episode:4}/{train_episodes} and step: {t:4}. Eps: {float(agent.epsilon):.2}, reward {reward}')
            break

    reward_lst.append(reward)

    if len(agent.memory)> batch_size:
        agent.replay(batch_size)

print(' Train mean % score= ', round(100*np.mean(reward_lst),1))

mean_reward = np.mean(reward_lst)
std_reward = np.std(reward_lst)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# test
test_wins=[]
for episode in range(test_episodes):
    state = env.reset()
    state_arr=np.zeros(state_size)
    state_arr[int(state[0])] = 1
    state= np.reshape(state_arr, [1, state_size])
    done = False
    reward=0
    state_lst = []
    state_lst.append(state)
    print('******* EPISODE ',episode, ' *******')

    for step in range(max_steps):
        action = agent.pred(state)
        result = env.step(action)
        new_state = result[0]
        reward += result[1]  # Cộng dồn phần thưởng
        done = result[2]
        new_state_arr = np.zeros(state_size)
        new_state_arr[new_state] = 1
        new_state = np.reshape(new_state_arr, [1, state_size])
        state = new_state
        state_lst.append(state)
        if done:
            print(reward)
            # env.render()
            break

    test_wins.append(reward)
env.close()

print(' Test mean % score= ', int(100*np.mean(test_wins)))

fig=plt.figure(figsize=(10,12))
matplotlib.rcParams.clear()
matplotlib.rcParams.update({'font.size': 16})
plt.subplot(311)
plt.scatter(list(range(len(reward_lst))), reward_lst)
plt.title('5x5 Frozen Lake Result(DQN) \n \nTrain Score')
plt.ylabel('Score')
plt.xlabel('Episode')

plt.subplot(312)
plt.scatter(list(range(len(agent.epsilon_lst))), agent.epsilon_lst)
plt.title('Epsilon')
plt.ylabel('Epsilon')
plt.xlabel('Episode')

plt.subplot(313)
plt.scatter(list(range(len(test_wins))), test_wins)
plt.title('Test Score')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.ylim((0,1.1))
plt.savefig('5x5resultdqn.png',dpi=300)
plt.show()

# import tensorflow as tf
# import numpy as np
# import gym
# import matplotlib.pyplot as plt

# #
# # from gym.envs.registration import register
# # # register(
# # #     id='FrozenLakeNotSlippery-v0',
# # #     entry_point='gym.envs.toy_text:FrozenLakeEnv',
# # #     kwargs={'map_name' : '4x4', 'is_slippery': False},
# # #     max_episode_steps=100,
# # #     reward_threshold=0.78, # optimum = .8196
# # # )


# tf.set_random_seed(1)
# np.random.seed(1)

# class DQN():
#     def __init__(self,nstate,naction):
#         self.nstate=nstate
#         self.naction=naction
#         self.sess = tf.Session()
#         self.memcnt=0
#         self.BATCH_SIZE = 64
#         self.LR = 0.0012                      # learning rate
#         self.EPSILON = 0.92                 # greedy policy
#         self.GAMMA = 0.9999                   # reward discount
#         self.MEM_CAP = 2000
#         self.mem= np.zeros((self.MEM_CAP, self.nstate * 2 + 2))     # initialize memory
#         self.updataT=150
#         self.built_net()


#     def built_net(self):
#         self.s = tf.placeholder(tf.float64, [None,self.nstate])
#         self.a = tf.placeholder(tf.int32, [None,])
#         self.r = tf.placeholder(tf.float64, [None,])
#         self.s_ = tf.placeholder(tf.float64, [None,self.nstate])

#         with tf.variable_scope('q'):                                  # evaluation network
#             l_eval = tf.layers.dense(self.s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
#             self.q = tf.layers.dense(l_eval, self.naction, kernel_initializer=tf.random_normal_initializer(0, 0.1))

#         with tf.variable_scope('q_next'):                                           # target network, not to train
#             l_target = tf.layers.dense(self.s_, 10, tf.nn.relu, trainable=False)
#             q_next = tf.layers.dense(l_target, self.naction, trainable=False)

#         q_target = self.r + self.GAMMA * tf.reduce_max(q_next, axis=1)    #q_next:  shape=(None, naction),
#         a_index=tf.stack([tf.range(self.BATCH_SIZE,dtype=tf.int32),self.a],axis=1)
#         q_eval=tf.gather_nd(params=self.q,indices=a_index)
#         loss=tf.losses.mean_squared_error(q_target,q_eval)
#         self.train=tf.train.AdamOptimizer(self.LR).minimize(loss)
#         #  q现实target_net- Q估计
#         self.sess.run(tf.global_variables_initializer())

#     def choose_action(self,status):
#         fs = np.zeros((1,self.nstate))
#         fs[0,status]=1.0  # ONE HOT
#         if  np.random.uniform(0.0,1.0)<self.EPSILON:
#             action=np.argmax( self.sess.run(self.q,feed_dict={self.s:fs}))
#         else:
#             action=np.random.randint(0,self.naction)
#         return action

#     def learn(self):
#         if(self.memcnt%self.updataT==0):
#             t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
#             e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
#             self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
#         rand_indexs=np.random.choice(self.MEM_CAP,self.BATCH_SIZE,replace=False)
#         temp=self.mem[rand_indexs]
#         bs = temp[:,0:self.nstate]#.reshape(self.BATCH_SIZE,NSTATUS)
#         ba = temp[:,self.nstate]
#         br = temp[:,self.nstate+1]
#         bs_ = temp[:,self.nstate+2:]#.reshape(self.BATCH_SIZE,NSTATUS)
#         self.sess.run(self.train, feed_dict={self.s:bs,self.a:ba,self.r:br,self.s_:bs_})


#     def storeExp(self,s,a,r,s_):
#         fs = np.zeros(self.nstate)
#         fs[s] = 1.0                       # ONE HOT
#         fs_ = np.zeros(self.nstate)
#         fs_[s_] = 1.0                          # ONE HOT
#         self.mem[self.memcnt%self.MEM_CAP]=np.hstack([fs,a,r,fs_])
#         self.memcnt+=1


#     def run(self,numsteps):
#         cnt_win =0
#         all_r=0.0
#         win_rate=[]
#         for i in range(numsteps):
#             s=env.reset()
#             done=False
#             while(not done):
#                 a=self.choose_action(s)
#                 s_,r,done,_=env.step(a)
#                 all_r+=r
#                 self.storeExp(s,a,r,s_)
#                 if(self.memcnt>self.MEM_CAP):
#                     self.learn()
#                     if(done):
#                         if(s_==self.nstate-1):
#                             cnt_win+=1.0
#                 s=s_
#             if (i % 50 == 0):
#                 print("period: ",i, ": ")
#                 if (cnt_win / 50 > 0.4):
#                      self.EPSILON += 0.01;
#                 elif (cnt_win / 50 > 0.2):
#                       self.EPSILON += 0.005;
#                 elif (cnt_win / 50 > 0.1):
#                       self.EPSILON += 0.003;
#                 elif (cnt_win / 50 > 0.05):
#                       self.EPSILON += 0.001;

#                 print("current accuracy: %.2f %%" %(cnt_win/50.0*100))
#                 win_rate.append(cnt_win / 50)
#                 cnt_win=0
#                 print("Global accuracy : %.2f %%" %(all_r / (i+1)*100))
#         print("Global accuracy : ",all_r/numsteps*100,"%")
#         plt.plot(win_rate)
#         plt.show()

# env = gym.make('FrozenLake-v0')
# #env = gym.make('FrozenLake8x8-v0')
# env = env.unwrapped
# dqn=DQN(env.observation_space.n,env.action_space.n)
# dqn.run(2000)
