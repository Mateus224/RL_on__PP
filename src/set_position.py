import rospy
from gazebo_msgs.msg import ModelState , ContactsState
from gazebo_msgs.srv import SetModelState
#from gazebo_msgs.ContactsState.msg import ContactsState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from std_msgs.msg import Bool
from std_msgs.msg import Empty
import ros_numpy
import numpy as np



class Node:

    def __init__(self):
        #rospy.init_node('get_pcl', anonymous=True)
        self.pub_takeOff=rospy.Publisher('/drone/takeoff', Empty, queue_size=10)
        self.pub_vel=rospy.Publisher('/drone/vel_mode', Bool, queue_size=10)
        #self.pub_vel=rospy.Publisher('/drone/posctrl', Bool, queue_size=10
        self.pub_vel2=rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.init_node('setVel', anonymous=True)
        rate=rospy.Rate(100)
        empty=Empty()
        move_cmd=Twist()
        move_cmd.linear.x=100.0
        move_cmd.angular.z=0.0
        b=Bool()
        b.data=True
        self.pub_vel.publish(b)
        rospy.sleep(0.5)
        self.pub_takeOff.publish(empty)
        #rospy.sleep(1.5)
        while True:
            
            self.pub_takeOff.publish(empty)
            self.pub_vel2.publish(move_cmd)
            rate.sleep()
        self.sub= None
        self.pc=None
        self.np_points_old=None


    def update_pose():
        rospy.wait_for_service('/gazebo/set_model_state')



        
        


    def get_pcl(self):
        wait_for_message=True
        while(wait_for_message):
            msg = rospy.wait_for_message("/camera/depth/points", PointCloud2, timeout=None)
            pc = ros_numpy.numpify(msg)
            height = pc.shape[0]
            width = pc.shape[1]
            np_points = np.zeros((height * width, 3), dtype=np.float32)
            np_points[:, 0] = np.resize(pc['x'], height * width)
            np_points[:, 1] = np.resize(pc['y'], height * width)
            np_points[:, 2] = np.resize(pc['z'], height * width)
            wait_for_message=False
            #if np.array_equal(self.np_points_old, np_points)
        msgs = rospy.wait_for_message("/gripper_contact_sensor_state", ContactsState, timeout=None)
        print(msgs)
        pc = ros_numpy.numpify(msg)
        height = pc.shape[0]
        width = pc.shape[1]
        np_points = np.zeros((height * width, 3), dtype=np.float32)
        np_points[:, 0] = np.resize(pc['x'], height * width)
        np_points[:, 1] = np.resize(pc['y'], height * width)
        np_points[:, 2] = np.resize(pc['z'], height * width)
        print(np_points,'---------------')
        #print(self.pc,'llll')
        

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
        
        points=self.get_pcl()

if __name__=="__main__":
    try:
        for i in range (30):
            test=Node()
            #test.new_pose(i)
    except rospy.ROSInterruptException:
        pass
