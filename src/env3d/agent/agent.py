from env3d.agent.lut_actions import Actions
import multiprocessing as mp
import numpy as np
import ctypes as c
import json



class Pose:
    def __init__(self):
        self.init_R= matrix_from_axis_angle([0,0,1,-np.pi/2])
        self.pose_matrix=None

    def reset(self, x=0, y=0, z=0, orientation=[0,0,1,np.pi/2]):
        self.pose_matrix= transform_from(self.init_R,[x,y,z])
        self.Sim_pose_matrix= transform_from(self.init_R,[x,y,z])

class AgentDispatcher():
    def __init__(self, config):
        self.b_multiagent=False
        self.uav=Agent(config, "UAV", xn, yn, zn)

    
    def reset(self):
        
        self.uav.reset_agent()
        #self.uuv.reset_agent()



class Agent():
    def __init__(self,config, agent_name):
        self.agent_key=agent_name
        self.config=config
        self.random_pose=config.getboolean(self.agent_key,"random_pose")
        rot_speed_degr=config.getint(self.agent_key,"rot_speed_degr")
        init_pose_json=config.get(self.agent_key,'init_pose')
        init_pose=json.loads(init_pose_json)
        self.init_pose=np.array(init_pose)

        self.pose = Pose()


        self.action_space=1
        self.rad=np.deg2rad(rot_speed_degr)
        self.actions= Actions(self.rad)


    def reset_agent(self):
        if self.random_pose:
            self.x0, self.y0 = np.random.randint(1, self.xn-2), np.random.randint(1, self.yn-2)
            min_z = self.real_2_D_map[self.x0][self.y0][0]
            assert min_z!=self.zn or min_z+1!=self.zn or min_z+1!=self.zn-1

            self.z0= 0 
            self.rotation=random_axis_angle()
        else:
            self.x0, self.y0, self.z0 = self.init_pose[0], self.init_pose[1], self.init_pose[2]
            self.rotation=random_axis_angle()
        self.pose.reset(x=self.x0, y=self.y0, z=self.z0)
        self.sensor_model.reset(self.map, self.pose, self.hashmap)
 

    def sim_actions(self,actionArr):
        if self.action_space==0:
            action_set = self.actions.ACTIONS2D
        elif self.action_space==1:
            action_set = self.actions.ACTIONS3D 
        elif self.action_space==2:
            action_set = self.actions.ACTIONS_cont_3D
        if type(actionArr).__module__=="numpy":
            sorted_actions=np.argsort(actionArr)[::-1]
        else:
            sorted_actions=np.argsort(actionArr.cpu().detach().numpy())[::-1]
        for i in range(sorted_actions.shape[0]):
            action=sorted_actions[i]
            R_t=np.matmul(np.matmul(matrix_from_axis_angle([1, 0, 0, action_set[action][3]]), \
                    matrix_from_axis_angle([0, 1, 0, action_set[action][4]])), \
                    matrix_from_axis_angle([0, 0, 1, action_set[action][5]]))
            self.pose.Sim_pose_matrix=np.matmul(self.pose.pose_matrix[:3,:3],R_t[:3,:3])   
            new_position=np.matmul(self.pose.Sim_pose_matrix[:3,:3],action_set[action][:3])
            new_position=self.pose.pose_matrix[:3,3]+new_position
            if self.legal_change_in_pose(new_position, _2D=False):
                return sorted_actions, i
        print('ff')
        assert True, f"no legal action chosen"
                





    def make_action(self, action, np_pose_matrix, np_sensor_matrix=None, action_space=0):        
        """Is implemented gitfor multiprocess if np_sensor_matrix is not None we store there the new matrix

        PARAMETERS
        ----------

        np_sensor_matrix: we store there the sesnor matrix in case we are using multiprozesses
        action_space: defines which action set is takes from the Action Class.
        """
        R_t=np.eye(4)
        if self.action_space==0:
            action_set = self.actions.ACTIONS2D
        elif self.action_space==1:
            action_set = self.actions.ACTIONS3D 
        elif self.action_space==2:
            action_set = self.actions.ACTIONS_cont_3D
        R_t=np.matmul(np.matmul(matrix_from_axis_angle([1, 0, 0, action_set[action][3]]), \
                matrix_from_axis_angle([0, 1, 0, action_set[action][4]])), \
                matrix_from_axis_angle([0, 0, 1, action_set[action][5]]))
            
        np_pose_matrix[:3,:3]=np.matmul(np_pose_matrix[:3,:3],R_t[:3,:3])   
        new_position=np.matmul(np_pose_matrix[:3,:3],action_set[action][:3])
        new_position=np_pose_matrix[:3,3]+new_position
        np_pose_matrix[:3,3]= new_position
        #np_pose_matrix[:3,:3]= np.matmul(np_pose_matrix[:3,:3],R_t[:3,:3])
        np_sensor_matrix[:,:,:3,3]= new_position
        np_sensor_matrix[:,:,:3,:3]= np.matmul(np_pose_matrix[:3,:3],self.sensor_model.sensor_matrix_init[:,:,:3,:3])
        
        return True

    
    def in_map(self, new_pos):
        return new_pos[0] >= 1 and new_pos[1] >= 1 and new_pos[0] < (self.xn-1) and new_pos[1] < (self.yn-1) and new_pos[2] >= 0 and new_pos[2] <= (self.zn)#(self.zn/2-1)

  
    def _2Dcollision(self, new_pos):
        if self.real_2_D_map[int(new_pos[0]),int(new_pos[1]),0]>=new_pos[2]:
            #print("COLLISION ! ! !", new_pos[0],int(new_pos[1]),int(new_pos[2]))
            return True
        else:
            return False


    def no_collision(self,new_pos):
        x = int(np.rint(new_pos[0]))
        y = int(np.rint(new_pos[1]))
        z = int(np.rint(new_pos[2]))
        hashkey = 1000000*x+1000*y+z
        if hashkey in self.hashmap:
            #print("COLLISION ! ! !", x,y,z)
            #print(self.real_2_D_map)
            return False
        else:
            return True

    def legal_rotation(self, new_pos):
        angleX,angleY, _ = (180/math.pi)*euler_xyz_from_matrix(self.pose.Sim_pose_matrix[:3,:3])
        if (abs(angleX)>46) or (abs(angleY)>46):
            return False
        else:
            return True




    def legal_change_in_pose(self, new_position, sub_map_border=None, _2D=True): 
        if sub_map_border!=None:
            in_sub_map=self.in_sub_map(new_position, sub_map_border)
        if _2D:
            return self.in_map(new_position) and not self._2Dcollision(new_position) and in_sub_map
        else:
            return self.in_map(new_position) and self.no_collision(new_position) and self.legal_rotation(new_position)
