import numpy as np
import datetime
import math
import random
import rospy
from env3d.settings import Scene_settings
from env3d.agent.transition import Transition

import time



class Env(object):
    def __init__(self,args,config):
        self.scene=Scene_settings(config)
        self.transition=Transition(config,self.scene)
        self.config=config
        self.pcl_size=512

        


    def reset(self):
        self.timeout=False 
        self.done=False
        self.pcl= np.zeros((3,self.pcl_size))
        
        self.scene.reset()
        self.transition.reset()
        self.old_pcl=np.zeros((1,3))
        state=[self.pcl, self.scene.quat_pose]

        return state , 0





    def step(self, unsorted_actions):
        pcl=None
        pcl_state= np.zeros((3,self.pcl_size))
        actions, i = self.transition.legal_transition(unsorted_actions)
        pcl, pose = self.transition.make_action(actions[i])
        pcl=np.swapaxes(pcl,0,1)
        pcl_state[:,:pcl.shape[1]]=pcl
        state=[pcl_state, pose]

        if self.old_pcl.shape[1] < pcl.shape[1]:
            reward=pcl.shape[1]-self.old_pcl.shape[1]
            reward = reward -  self.transition.diff
            self.old_pcl=pcl
        else:
            reward=0
        print("reward",reward)

        return state, reward, actions, i, False

    def simulate_step(self, action):
        pcl= np.zeros((3,0))
        print(action)
        if (self.transition.check_transition(action)):
            reward = self.transition.simulate_action(action)
            print(reward)
            #pcl=np.swapaxes(pcl,0,1)
            #print(self.old_pcl.shape[1], pcl.shape[1])
            #if self.old_pcl.shape[1] < pcl.shape[1]:
            #    reward=pcl.shape[1]-self.old_pcl.shape[1]
            #else:
            #    reward = 0
        else:
            reward = 0
        return reward


         
        


