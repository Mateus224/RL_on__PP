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
        self.pcl_size=600

        


    def reset(self):
        self.timeout=False 
        self.done=False
        self.pcl= np.zeros((3,self.pcl_size))
        
        self.scene.reset()
        self.transition.reset()
        self.old_pcl=np.zeros((1,3))
        state=[self.pcl, self.scene.quat_pose]
        self.points=0

        return state 





    def step(self, unsorted_actions):
        pcl=None
        terminate=False
        pcl_state= np.zeros((3,self.pcl_size))
        actions, i = self.transition.legal_transition(unsorted_actions)
        pcl, pose = self.transition.make_action(actions[i])
        pcl=np.swapaxes(pcl,0,1)
        self.points=pcl.shape[1]
        if pcl.shape[1]>599:
            terminate=True
            print("too many points !!!", pcl.shape)
        else:
            pcl_state[:,:pcl.shape[1]]=pcl
        state=[pcl_state, pose]
        #if state.shape[0]>512:

        reward = self.transition.reward
        #if self.old_pcl.shape[1] < pcl.shape[1]:
        #    reward=pcl.shape[1]-self.old_pcl.shape[1]
        #    reward = reward -  self.transition.diff
        #    self.old_pcl=pcl
        #else:
        #    reward=0
        return state, reward, actions, i, terminate


    def simulate_step(self, action):
        pcl= np.zeros((3,0))
        if (self.transition.check_transition(action)):
            reward = self.transition.simulate_action(action)
            #pcl=np.swapaxes(pcl,0,1)
            #print(self.old_pcl.shape[1], pcl.shape[1])
            #if self.old_pcl.shape[1] < pcl.shape[1]:
            #    reward=pcl.shape[1]-self.old_pcl.shape[1]
            #else:
            #    reward = 0
        else:
            reward = 0
        return reward

    """
    def simulate_step(self, action):
        pcl= np.zeros((3,0))
        if (self.transition.check_transition(action)):
            pcl = self.transition.simulate_action(action)
            pcl=np.swapaxes(pcl,0,1)
            print(self.old_pcl.shape[1], pcl.shape[1])
            if self.old_pcl.shape[1] < pcl.shape[1]:
                reward=pcl.shape[1]-self.old_pcl.shape[1]
            else:
                reward = 0
        else:
            reward = 0
        return reward
    """

         
        


