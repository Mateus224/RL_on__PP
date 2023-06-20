
import json
import numpy as np
import rospy
import roslib

from gazebo_msgs.msg import ModelState , ContactsState
from gazebo_msgs.srv import SetModelState, SetPhysicsProperties
from pcg_gazebo.generators import WorldGenerator
from pcg_gazebo.simulation import SimulationModel
from pcg_gazebo.task_manager import Server
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64   
import gazebo_msgs.msg
import geometry_msgs.msg
import open3d as o3d
import tf
import tf2_ros

class Scene_settings():

    def __init__(self,config):
        json_sensor_rays=config.get('Agent01','sensor_rays')
        list_sensor_rays=json.loads(json_sensor_rays)
        self.sensor_rays=np.array(list_sensor_rays)

        json_pose=config.get('Agent01', 'init_pose')
        list_json_pose=json.loads(json_pose)
        self.pose_=np.array(list_json_pose)

        random_pose=config.getboolean('Agent01', 'random_pose')
        self.create_environment()

        rospy.init_node('RL_agent_test',log_level=rospy.INFO)

        self.initial_pose = Pose()
        if random_pose:
            self.pose_[0]= np.random.randint(13)+1
            self.pose_[1]= np.random.randint(13)+1
            self.pose_[2]= 1
            self.quat = tf.transformations.quaternion_from_euler(
                   float(0),float(0),float(0))

        self.quat_pose=np.zeros((7))
        self.quat_pose[:3]=self.pose_[:3]
        self.quat_pose[3:]=self.quat[:4]
            
        self.initial_pose.position.x =self.pose_[0]
        self.initial_pose.position.y =self.pose_[1]
        self.initial_pose.position.z =self.pose_[2]

        self.__init__worldCoordinates()
        self.__init__gravity()



        
    def __init__worldCoordinates(self):
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "UAV"


        static_transformStamped.transform.translation.x = float(self.pose_[0])
        static_transformStamped.transform.translation.y = float(self.pose_[1])
        static_transformStamped.transform.translation.z = float(self.pose_[2])

        
        static_transformStamped.transform.rotation.x = self.quat[0]
        static_transformStamped.transform.rotation.y = self.quat[1]
        static_transformStamped.transform.rotation.z = self.quat[2]
        static_transformStamped.transform.rotation.w = self.quat[3]

        broadcaster.sendTransform(static_transformStamped)


    def __init__gazebo(self):
        server = Server()  
        server.create_simulation('pcg-example2')
        simulation = server.get_simulation('pcg-example2')
        tf2_ros.StaticTransformBroadcaster()
        #rospy.init_node('insert_object',log_level=rospy.INFO)
        simulation.create_gazebo_empty_world_task()
        simulation.run_all_tasks()
        self.gazebo_proxy = simulation.get_gazebo_proxy()
        pos=zeros((7))
        pos[:3]=1
        self.set_pose("UAV", pos)


    def __init__gravity(self):
        set_gravity = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        time_step = Float64(0.001)
        max_update_rate = Float64(1000.0)
        gravity = geometry_msgs.msg.Vector3()
        gravity.x = 0.0
        gravity.y = 0.0
        gravity.z = 0.0
        ode_config = gazebo_msgs.msg.ODEPhysics()
        ode_config.auto_disable_bodies = False
        ode_config.sor_pgs_precon_iters = 0
        ode_config.sor_pgs_iters = 50
        ode_config.sor_pgs_w = 1.3
        ode_config.sor_pgs_rms_error_tol = 0.0
        ode_config.contact_surface_layer = 0.001
        ode_config.contact_max_correcting_vel = 0.0
        ode_config.cfm = 0.0
        ode_config.erp = 0.2
        ode_config.max_contacts = 20
        set_gravity(time_step.data, max_update_rate.data, gravity, ode_config)


    def __init__spawnAgent(self):
        f = open('/home/matthias/catkin_ws/src/my_drone/urdf/my_drone.urdf','r')
        urdff = f.read()
        rospy.wait_for_service('gazebo/spawn_urdf_model')
        spawn_model_proxy = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
        spawn_model_proxy("UAV", urdff, "robotos_name_space", self.initial_pose, "world")    


    def __init__spawnObject(self, i):
        _obj=Pose()
        _obj.position.x = 50+i
        _obj.position.y = 50
        _obj.position.z = 1
        spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        spawn_model_client(
        model_name='zylinder'+str(i),
            model_xml=open('/home/matthias/catkin_ws/src/my_drone/urdf/pyr.urdf', 'r').read(),
            robot_namespace='/zylinder',
            initial_pose=_obj,
            reference_frame='world'
        )

    def __init__spawnWall_x(self, i):
        _obj=Pose()
        _obj.position.x = 6.5
        _obj.position.y = -0.5+(i*14)
        _obj.position.z = 1
        spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_client(
        model_name='wall_x'+str(i),
            model_xml=open('/home/matthias/catkin_ws/src/my_drone/urdf/wall_x.sdf', 'r').read(),
            robot_namespace='/wall_x',
            initial_pose=_obj,
            reference_frame='world'
        )

    def __init__spawnWall_y(self, i):
        _obj=Pose()
        _obj.position.x = -0.5+(i*14)
        _obj.position.y =  6.5
        _obj.position.z = 1
        spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_client(
        model_name='wall_y'+str(i),
            model_xml=open('/home/matthias/catkin_ws/src/my_drone/urdf/wall_y.sdf', 'r').read(),
            robot_namespace='/wall_x',
            initial_pose=_obj,
            reference_frame='world'
        )

    def __init__spawnWall_in_b(self):
        _obj=Pose()
        _obj.position.x = 6
        _obj.position.y =  6
        _obj.position.z =1
        spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_client(
        model_name='wall_in_b',
            model_xml=open('/home/matthias/catkin_ws/src/my_drone/urdf/wall_in_b.sdf', 'r').read(),
            robot_namespace='/wall_x',
            initial_pose=_obj,
            reference_frame='world'
        )

    def __init__spawnWall_in_s(self):
        _obj=Pose()
        _obj.position.x = 6
        _obj.position.y =  3
        _obj.position.z = 1
        spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_client(
        model_name='wall_in_s',
            model_xml=open('/home/matthias/catkin_ws/src/my_drone/urdf/wall_in_s.sdf', 'r').read(),
            robot_namespace='/wall_x',
            initial_pose=_obj,
            reference_frame='world'
        )

    def set_pose(self, name, position):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException :
            print("Service call failed")
        
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = position[0]
        state_msg.pose.position.y = position[1]
        state_msg.pose.position.z = position[2]
        state_msg.pose.orientation.w = position[3]
        state_msg.pose.orientation.x = position[4]
        state_msg.pose.orientation.y = position[5]
        state_msg.pose.orientation.z = position[6]

        resp = set_state(state_msg)

    

    def create_environment(self):
        #env_2D= np.random.choice(2, 100, p=(0.98, 0.02))
        #env_2D[np.random.randint(40)]=1
        #self.env_2D= env_2D.reshape((10,10))
        
        #self.env_2D=np.zeros((13,13))
        #x_y = np.random.randint(1,9, size=2)
        #x_y1 = np.random.randint(1,9, size=2)
        #if np.array_equal(x_y, x_y1):
        #     x_y1=np.random.randint(1,9, size=2)
        #     if np.array_equal(x_y, x_y1):
        #        x_y1=np.random.randint(1,9, size=2)
        #self.env_2D[x_y[0],x_y[1]]=1
        #self.env_2D[x_y1[0],x_y1[1]]=1
        for i in range(35):
            self.__init__spawnObject(i) 
        for i in range(2):
            self.__init__spawnWall_x(i) 
            self.__init__spawnWall_y(i)
        self.__init__spawnWall_in_b()
        self.__init__spawnWall_in_s()
        return 

    def env_walls(self):
        self.env_2D[0,:]=2
        self.env_2D[13,:]=2
        self.env_2D[:,0]=2
        self.env_2D[:,13]=2
        for i in range(7):
            self.env_2D[i+3,6]=2
            self.env_2D[6,i]=2
            import sys
            np.set_printoptions(threshold=sys.maxsize)
        

    def reset(self, pose=None):
        self.env_2D=np.zeros((14,14))
        self.env_walls()
        position=np.zeros((7))
        zylinder="zylinder"
        for i in range(35):
            position[0] = 50+i
            position[1] = 50
            position[2] = 1
            position[3] = 1
            self.set_pose(zylinder+str(i),position)            
        #env_2D= np.random.choice(2, 100, p=(0.98, 0.02))
        
        occupied=True
        while occupied:
            x_y0 = np.random.randint(2,11, size=2)
            if self.env_2D[x_y0[0],x_y0[1]] == 0:
                self.env_2D[x_y0[0],x_y0[1]]=1
                occupied=False
        occupied=True 
        while occupied:
            x_y1 = np.random.randint(2,11, size=2)
            if self.env_2D[x_y1[0],x_y1[1]] == 0:
                self.env_2D[x_y1[0],x_y1[1]]=1
                occupied=False     
        occupied=True  
        while occupied:
            x_y2 = np.random.randint(2,11, size=2)
            if self.env_2D[x_y2[0],x_y2[1]] == 0:
                self.env_2D[x_y2[0],x_y2[1]]=1
                occupied=False 
        occupied=True  
        #while occupied:
        #    x_y3 = np.random.randint(2,11, size=2)
        #    if self.env_2D[x_y3[0],x_y3[1]] == 0:
        #        self.env_2D[x_y3[0],x_y3[1]]=1
        #        occupied=False   

        o_place=np.where(self.env_2D==1)
        o_place=np.array(o_place)
        self.num_obj=o_place.shape[1]
        
        position = np.zeros((7))
        for i in range (self.num_obj):
            position[0] = o_place[0][i]
            position[1] = o_place[1][i]
            position[2] = 1
            position[3] = 1
            self.set_pose(zylinder+str(i),position)
        agent="UAV"
        if pose== None:
            check=True
            while(check):
                self.pose_[0]= np.random.randint(1,12)
                self.pose_[1]= np.random.randint(1,12)
                if self.env_2D[self.pose_[0]][self.pose_[1]] == 0:
                    check=False
                else: 
                    check=True
        else:
            self.pose_[0]=pose[0]
            self.pose_[1]=pose[1]
        self.pose_[2]= 0.5
        
        self.quat = tf.transformations.quaternion_from_euler(
                   float(0),float(0),float(0))

        self.quat_pose=np.zeros((7))
        self.quat_pose[:3]=self.pose_[:3]
        self.quat_pose[3:]=self.quat[:4]
        self.set_pose(agent, self.quat_pose)
        return
        
