#!/usr/bin/env python 
print("aa")
import rospy
from gazebo_msgs.msg import ModelState , ContactsState
from gazebo_msgs.srv import SetModelState, SetPhysicsProperties
from pcg_gazebo.generators import WorldGenerator
import numpy as np
from pcg_gazebo.simulation import SimulationModel
from pcg_gazebo.task_manager import Server
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64   
import gazebo_msgs.msg
import geometry_msgs.msg


class Node:

    def __init__(self):

        server = Server()  
        server.create_simulation('pcg-example')
        simulation = server.get_simulation('pcg-example')
        rospy.init_node('insert_object',log_level=rospy.INFO)
        simulation.create_gazebo_empty_world_task()
        simulation.run_all_tasks()
        self.gazebo_proxy = simulation.get_gazebo_proxy()

        initial_pose = Pose()
        initial_pose.position.x = 0
        initial_pose.position.y = 1
        initial_pose.position.z = 1
             
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
        f = open('/home/matthias/catkin_ws/src/myDrone/urdf/sjtu_drone.urdf','r')
        #f = open('/home/matthias/catkin_ws/src/myDrone/urdf/zylinder.sdf','r')
        urdff = f.read()
        
        rospy.wait_for_service('gazebo/spawn_urdf_model')
        spawn_model_proxy = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
        spawn_model_proxy("UAV", urdff, "robotos_name_space", initial_pose, "world")

        rospy.init_node('insert_object',log_level=rospy.INFO)
        spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_client(
        model_name='zylinder01',
            model_xml=open('/home/matthias/catkin_ws/src/myDrone/urdf/zylinder.sdf', 'r').read(),
            robot_namespace='/foo',
            initial_pose=initial_pose,
            reference_frame='world'
        )

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException :
            print("Service call failed")
        
        state_msg = ModelState()
        state_msg.model_name = 'zylinder01'


        state_msg.pose.position.x = 1
        state_msg.pose.position.y = 5
        state_msg.pose.position.z =  0
        state_msg.pose.orientation.w = 1
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0

        resp = set_state(state_msg)




    def new_pose(self,i):
    #GAZEBO SERVICE
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException :
            print("Service call failed")
        
        state_msg = ModelState()
        state_msg.model_name = 'sjtu_drone'


        state_msg.pose.position.x = 1+(i/10)
        state_msg.pose.position.y = 0
        state_msg.pose.position.z =  1.48
        state_msg.pose.orientation.w = 1
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = .3
        state_msg.pose.orientation.z = 0

        resp = set_state(state_msg)
        
        #points=self.get_pcl()

if __name__=="__main__":
    try:
        test=Node()
        test.new_pose(2)
        while True:
            a=1
    except rospy.ROSInterruptException:
        pass