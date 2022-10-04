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

        


    def reset(self):
        self.timeout=False 
        self.done=False
        self.pcl= np.zeros((3,1024))
        
        self.scene.reset()
        self.transition.reset()
        self.old_pcl=np.zeros((1,3))
        state=[self.pcl, self.scene.quat_pose]

        return state , 0





    def step(self, sorted_actions):
        pcl=None
        pcl_state= np.zeros((3,1024))
        action, done = self.transition.legal_transition(sorted_actions)
        if done is False:
            pcl, pose = self.transition.make_action(action)
            done=False
            pcl=np.swapaxes(pcl,0,1)
        else:
            pcl, pose = self.transition.make_action(action)
            done=True
            pcl=np.swapaxes(pcl,0,1)
        pcl_state[:,:pcl.shape[1]]=pcl
        state=[pcl_state, pose]

        if self.old_pcl.shape[1] < pcl.shape[1]:
            reward=pcl.shape[1]-self.old_pcl.shape[1]
            self.old_pcl=pcl
        else:
            reward=0

        return state, reward, action, done






        return 0, 0, 0,0,0,0

    def Q_guided_action(sorted_actions,step_size,action_state,limit_length,limit_width,camera,scene):
        for j in sorted_actions:

            ok= step(reset=2,
                        step_size=step_size,
                        step_choice=j,
                        action_state=action_state,
                        limit_length=limit_length,
                        limit_width=limit_width,
                        camera=camera,
                        scene=scene)
            if (ok==1):
                print("Action "+str(i)+" :"+str(j))
                return ok
                
        return ok




        

    def save_pcl(self):
        now = datetime.datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        #end_time = time.time()

        #print(end_time-start_time)


        path_file="./Sim_output/out_"+str(current_time)+".pcd"


        f=open(path_file,"w") #File location to modify
        i = 0; #Storage point cloud number
        for item in bpy.data.objects:
            if item.type == 'MESH' and item.name.startswith('Scan'):
                print('write once')
                for sp in item.data.vertices:
                    #print('X=%+#5.3f\tY=%+#5.3f\tZ=%+#5.3f' % (sp.co[0], sp.co[1],sp.co[2]));
                    if(  (np.isnan(sp.co[0])==0) and(np.isnan(sp.co[1])==0) and (np.isnan(sp.co[2])==0) ):
                        string_out='%#5.3f\t%#5.3f\t%#5.3f \n' % (sp.co[0], sp.co[1],sp.co[2])
                        i = i+1
                        f.write(string_out);  
        f.close()          
        f=open(path_file,"r+") #File location to modify
        new_data = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n" %(i,i))
        old = f.read()
        f.seek(0)
        f.write(new_data)
        f.write(old)


         
        


