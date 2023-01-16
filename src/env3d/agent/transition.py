import json
import numpy as np

 

import tf2_ros
#import tf2_py as tf2
#import tf2_geometry_msgs
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import std_msgs.msg
import rospy
from env3d.agent.lut_actions import Actions
#import utils.math_f as math_
import open3d as o3d
import pytransform3d.rotations as rot
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import ros_numpy
import geometry_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import tf
import pcl



#rospy.init_node('point_polygon_scaler')



class Transition():
    def __init__(self, config, scene):
        self.step=0
        json_env_shape=config.get('ENV','shape')
        list_env_shape=json.loads(json_env_shape)
        self.env_shape=np.array(list_env_shape)
        self.step_size= config.getint('Agent01', 'step_size')
        actions=Actions(1.57079633)
        self.action_set=actions.ACTIONS2D
        
        self.x = self.env_shape[0]
        self.y = self.env_shape[1]
        self.z = self.env_shape[2]
        self.scene=scene
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          #PointField('rgb', 12, PointField.UINT32, 1),
          ]

        self.header = std_msgs.msg.Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = 'world'

  

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.reset()
        
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException :
            print("Service call failed to set agents position")

    def reset(self):
        self.step=0
        pose = self.scene.pose_
        self.global_pcl=np.zeros((1,3))
        self.position= np.zeros((7))
        self.position[:3]=pose[:3]
        self.position[2]=0.65
        self.position[3]=1
        self.old_cloud=np.zeros((1,3))
        self.allT_objects=np.zeros((1,3))
        self.comulativ_reward = 0
        self.comulativ_old_reward = 0
        self.exp_reward = 0
        self.exp_old_reward = 0



        self.pcd = o3d.geometry.PointCloud()
        self.ground_points = o3d.geometry.PointCloud()
        self.wall_points = o3d.geometry.PointCloud()
        self.objs = o3d.geometry.PointCloud()

        self.pcd_sim = o3d.geometry.PointCloud()
        self.ground_points_sim = o3d.geometry.PointCloud()


        self.pub = rospy.Publisher("/pcl_topic", PointCloud2)
        self.pub_agent = rospy.Publisher("/pcl_topic_agent", PointCloud2)

    def legal_transition(self,actions):
        actions_arg_sorted=np.argsort(-1*actions)
        for i in range(actions_arg_sorted.shape[0]):
            if self.check_transition(actions_arg_sorted[i]):
                return actions_arg_sorted, i



    def check_transition(self, action):
        
        x=self.position[0]+self.action_set[action][0]
        y=self.position[1]+self.action_set[action][1]
        if self.scene.env_2D[int(x),int(y)]>0:
            return False
        return True
        

    def transform_cloud(self,msg):

        transform = self.tf_buffer.lookup_transform("world", msg.header.frame_id, rospy.Time(0),rospy.Duration(4.0)) #target frame / source frame
        transformed_cloud = do_transform_cloud(msg, transform)
        transformed_cloud.header.frame_id = 'world'
        #self.pub.publish(transformed_cloud)
        return transformed_cloud

    def transform_cloud_toAgent(self, msg):
        
        transform = self.tf_buffer.lookup_transform("UAV", msg.header.frame_id, rospy.Time(0),rospy.Duration(4.0)) #target frame / source frame
        transformed_cloud = do_transform_cloud(msg, transform)
        transformed_cloud.header.frame_id = 'UAV'
        return transformed_cloud


    def get_pcl(self):
        wait_for_pcl=True
        i=0
        if self.step==0:
            rospy.sleep(0.2)
        while wait_for_pcl:
            msg_ = rospy.wait_for_message("/camera/depth/points", PointCloud2, timeout=None)
            msg=self.transform_cloud(msg_)
            pc_np_ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg_, remove_nans = True)
            pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans = True)
            if (np.array_equal(pc_np_,self.old_cloud)):
                i=i+1
                if i>10:
                    wait_for_pcl=False
                    print('-----------')
            else:
                wait_for_pcl=False
        



        self.old_cloud=np.copy(pc_np_)
        self.global_pcl=np.concatenate((self.global_pcl, np.copy(pc_np)), axis=0)
        self.global_pcl=np.where(self.global_pcl<0.01,0.01,  self.global_pcl)
        pc_pcl = pcl.PointCloud(np.array(self.global_pcl, dtype = np.float32))

        self.pcd.points = o3d.utility.Vector3dVector(pc_pcl)

        o_place=np.where(self.scene.env_2D==1)
        o_place=np.array(o_place)
        #num_obj=o_place.shape[1]
        obj1_points = np.copy(np.asarray(self.pcd.points))

        object1_mask0 = obj1_points[:,0] > o_place[0][0]-.35
        object1_mask1 = obj1_points[:,0] < o_place[0][0]+.35
        object1_mask2 = obj1_points[:,1] > o_place[1][0]-.35
        object1_mask3 = obj1_points[:,1] < o_place[1][0]+.35

        object2_mask0 = obj1_points[:,0] > o_place[0][1]-0.35
        object2_mask1 = obj1_points[:,0] < o_place[0][1]+0.35
        object2_mask2 = obj1_points[:,1] > o_place[1][1]-0.35
        object2_mask3 = obj1_points[:,1] < o_place[1][1]+0.35

        object3_mask0 = obj1_points[:,0] > o_place[0][2]-0.35
        object3_mask1 = obj1_points[:,0] < o_place[0][2]+0.35
        object3_mask2 = obj1_points[:,1] > o_place[1][2]-0.35
        object3_mask3 = obj1_points[:,1] < o_place[1][2]+0.35

       # object4_mask0 = obj1_points[:,0] > o_place[0][3]-0.35
       # object4_mask1 = obj1_points[:,0] < o_place[0][3]+0.35
       # object4_mask2 = obj1_points[:,1] > o_place[1][3]-0.35
       # object4_mask3 = obj1_points[:,1] < o_place[1][3]+0.35

        object1_mask=np.all([object1_mask0,object1_mask1], axis=0)
        object1_mask=np.all([object1_mask,object1_mask2], axis=0)
        object1_mask=np.all([object1_mask,object1_mask3], axis=0)

        object2_mask=np.all([object2_mask0,object2_mask1], axis=0)
        object2_mask=np.all([object2_mask,object2_mask2], axis=0)
        object2_mask=np.all([object2_mask,object2_mask3], axis=0)

        object3_mask=np.all([object3_mask0,object3_mask1], axis=0)
        object3_mask=np.all([object3_mask,object3_mask2], axis=0)
        object3_mask=np.all([object3_mask,object3_mask3], axis=0)

        #object4_mask=np.all([object4_mask0,object4_mask1], axis=0)
        #object4_mask=np.all([object4_mask,object4_mask2], axis=0)
        #object4_mask=np.all([object4_mask,object4_mask3], axis=0)

        self.np_objects=np.concatenate((obj1_points[object1_mask],obj1_points[object2_mask]),axis=0)
        self.np_objects=np.concatenate((self.np_objects,obj1_points[object3_mask]),axis=0)
        #self.np_objects=np.concatenate((self.np_objects,obj1_points[object4_mask]),axis=0)
        self.objs.points = o3d.utility.Vector3dVector(self.np_objects)
        #self.obj2.points = o3d.utility.Vector3dVector(obj2_points[object2_mask])

        """
        points_1 = np.copy(np.asarray(self.pcd.points))
        points_2 = np.copy(np.asarray(self.pcd.points))
        points_wall = np.copy(np.asarray(self.pcd.points))
        mask_glob = points_1[:,2] > 0.02
        mask_globx0 = points_1[:,0] > 0.1
        mask_globx1 = points_1[:,0] < 11.9
        mask_globy0 = points_1[:,1] > 0.1
        mask_globy1 = points_1[:,1] < 11.9        


        mask_ground = points_2[:,2] < 0.02 
        #mask_groundx0 = points_2[:,0] > -0.01
        #mask_groundx1 = points_2[:,0] < 12.01
        #mask_groundy0 = points_2[:,1] > -0.01
        #mask_groundy1 = points_2[:,1] < 12.01

        mask_wallx0 = points_wall[:,0] <= 0.1
        mask_wallx1 = points_wall[:,0] >= 11.9
        mask_wally0 = points_wall[:,1] <= 0.1
        mask_wally1 = points_wall[:,1] >= 11.9
        

        mask_glob=np.all([mask_glob,mask_globx0], axis=0)
        mask_glob=np.all([mask_glob,mask_globx1], axis=0)
        mask_glob=np.all([mask_glob,mask_globy0], axis=0)
        mask_glob=np.all([mask_glob,mask_globy1], axis=0)
        

        #mask_ground=np.all([mask_ground,mask_groundx0], axis=0)
        #mask_ground=np.all([mask_ground,mask_groundx1], axis=0)
        #mask_ground=np.all([mask_ground,mask_groundy0], axis=0)
        #mask_ground=np.all([mask_ground,mask_groundy1], axis=0)

        mask_wall=np.any([mask_wallx0,mask_wallx1], axis=0)
        mask_wall=np.any([mask_wall,mask_wally0], axis=0)
        mask_wall=np.any([mask_wall,mask_wally1], axis=0)

        self.pcd.points = o3d.utility.Vector3dVector(points_1[mask_glob])
        self.ground_points.points = o3d.utility.Vector3dVector(points_2[mask_ground])
        self.wall_points.points = o3d.utility.Vector3dVector(points_wall[mask_wall])

        """
        #self.ground_points = self.ground_points.voxel_down_sample(voxel_size=0.9)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.9)
        self.pcd_np=np.copy(np.asarray(self.pcd.points))
        #self.wall_points = self.wall_points.voxel_down_sample(voxel_size=1.4)
        #self.objs = self.objs.voxel_down_sample(voxel_size=0.3)
        #self.obj2 = self.pcd.voxel_down_sample(voxel_size=0.1)
        #print("points in object:", np.asarray(self.obj1.points).shape[0]+np.asarray(self.obj2.points).shape[0])



        self.allT_objects=np.concatenate((self.np_objects, self.allT_objects),axis=0)
        self.objs.points = o3d.utility.Vector3dVector(self.allT_objects)
        self.objs = self.objs.voxel_down_sample(voxel_size=0.25)
        self.allT_objects=np.copy(np.asarray(self.objs.points))
        self.global_pcl=np.concatenate((self.allT_objects,np.asarray(self.pcd.points)),axis=0)

        self.comulativ_reward= self.allT_objects.shape[0]
        self.reward = self.comulativ_reward - self.comulativ_old_reward
        self.comulativ_old_reward = self.comulativ_reward

        self.exp_reward = self.pcd_np.shape[0]
        self.exp_reward_diff = self.exp_reward - self.exp_old_reward
        self.exp_old_reward = self.exp_reward
        

        self.reward=self.reward + (self.exp_reward_diff/10)


        self.pcd.points = o3d.utility.Vector3dVector(self.global_pcl)
        #self.objs = self.objs.voxel_down_sample(voxel_size=0.25)
        self.header.stamp = rospy.Time.now()
        pc2 = point_cloud2.create_cloud(self.header, self.fields, self.pcd.points)
        
        self.pub.publish(pc2)
        
        transformedPc2 = self.transform_cloud_toAgent(pc2)
        transformedPc2.header.stamp = rospy.Time.now()
        transformed_pcl = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(transformedPc2, remove_nans = True)

        self.step+=1
        return transformed_pcl
        


    def make_action(self, action):
        
        action_vector=self.action_set[action]
        #print(action_vector,'actionvector')

        if action<4:
            self.position[:2]=self.position[:2]+action_vector[:2]
        else:
            self.position[3:]=rot.concatenate_quaternions(self.position[3:],rot.quaternion_from_axis_angle([0, 0, 1, action_vector[4]]))
            self.position[3:]=rot.concatenate_quaternions(self.position[3:],rot.quaternion_from_axis_angle([0, 0, 1, action_vector[5]]))

        

        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "UAV"


        static_transformStamped.transform.translation.x = float(self.position[0])
        static_transformStamped.transform.translation.y = float(self.position[1])
        static_transformStamped.transform.translation.z = float(self.position[2])

        quat = tf.transformations.quaternion_from_euler(
                   float(0),float(0),float(0))
        static_transformStamped.transform.rotation.x = self.position[4]
        static_transformStamped.transform.rotation.y = self.position[5]
        static_transformStamped.transform.rotation.z = self.position[6]
        static_transformStamped.transform.rotation.w = self.position[3]

        broadcaster.sendTransform(static_transformStamped)
        self.scene.set_pose("UAV",self.position)
        pcl=self.get_pcl()
        pcl2 = point_cloud2.create_cloud(self.header, self.fields, self.pcd.points)
        transformedPc2 = self.transform_cloud_toAgent(pcl2)
        transformedPc2.header.stamp = rospy.Time.now()
        #self.pub_agent.publish(transformedPc2)
        #rospy.sleep(0.2)
        transformed_pcl = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(transformedPc2, remove_nans = True)
        return transformed_pcl, self.position




    def get_simulated_pcl(self):
        #self.pcd_sim=self.pcd
        #self.ground_points_sim=self.ground_points
        wait_for_pcl=True
        i=0
        if self.step==0:
            rospy.sleep(0.2)
        while wait_for_pcl:
            msg_ = rospy.wait_for_message("/camera/depth/points", PointCloud2, timeout=None)
            msg=self.transform_cloud(msg_)
            pc_np_ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg_, remove_nans = True)
            pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans = True)
            if (np.array_equal(pc_np_,self.old_cloud)):
                i=i+1
                if i>10:
                    wait_for_pcl=False
                    #print('i bigger 8')
            else:
                wait_for_pcl=False
        #self.old_cloud=np.copy(pc_np_)
        global_pcl=np.concatenate((self.global_pcl, np.copy(pc_np)), axis=0)
        global_pcl=np.where(global_pcl<0.01,0.01,  global_pcl)
        pc_pcl = pcl.PointCloud(np.array(global_pcl, dtype = np.float32))

        self.pcd_sim.points = o3d.utility.Vector3dVector(pc_pcl)
        points_1 = np.copy(np.asarray(self.pcd_sim.points))
        points_2 = np.copy(np.asarray(self.pcd_sim.points))
        mask_glob = points_1[:,2] > 0.02
        mask_ground = points_2[:,2] < 0.02 
        mask_groundx0 = points_2[:,0] > 0.0
        mask_groundx1 = points_2[:,0] < 10.0
        mask_groundy0 = points_2[:,1] > 0.0
        mask_groundy1 = points_2[:,1] < 10.0
        mask_ground=np.all([mask_ground,mask_groundx0], axis=0)
        mask_ground=np.all([mask_ground,mask_groundx1], axis=0)
        mask_ground=np.all([mask_ground,mask_groundy0], axis=0)
        mask_ground=np.all([mask_ground,mask_groundy1], axis=0)
        self.pcd_sim.points = o3d.utility.Vector3dVector(points_1[mask_glob])
        self.ground_points_sim.points = o3d.utility.Vector3dVector(points_2[mask_ground])
        self.ground_points_sim = self.ground_points_sim.voxel_down_sample(voxel_size=0.9)
        self.pcd_sim = self.pcd_sim.voxel_down_sample(voxel_size=0.2)
        global_pcl=np.concatenate((np.copy(np.asarray(self.ground_points_sim.points)),np.copy(np.asarray(self.pcd_sim.points))),axis=0)
        #self.pcd_sim.points = o3d.utility.Vector3dVector(self.global_pcl)
        #self.header.stamp = rospy.Time.now()
        #pc2 = point_cloud2.create_cloud(self.header, self.fields, self.pcd_sim.points)
        #self.pub.publish(pc2)
        return global_pcl #, self.global_pcl


    def simulate_action(self, action):
        
        action_vector=self.action_set[action]
        position=np.copy(self.position)
        if action<4:
            position[:2]=self.position[:2]+action_vector[:2]
        else:
            position[3:]=rot.concatenate_quaternions(self.position[3:],rot.quaternion_from_axis_angle([0, 0, 1, action_vector[4]]))
            position[3:]=rot.concatenate_quaternions(self.position[3:],rot.quaternion_from_axis_angle([0, 0, 1, action_vector[5]]))

        broadcaster = tf2_ros.StaticTransformBroadcaster()
        rospy.sleep(0.002)
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "UAV"


        static_transformStamped.transform.translation.x = float(position[0])
        static_transformStamped.transform.translation.y = float(position[1])
        static_transformStamped.transform.translation.z = float(position[2])

        quat = tf.transformations.quaternion_from_euler(
                   float(0),float(0),float(0))
        static_transformStamped.transform.rotation.x = position[4]
        static_transformStamped.transform.rotation.y = position[5]
        static_transformStamped.transform.rotation.z = position[6]
        static_transformStamped.transform.rotation.w = position[3]

        broadcaster.sendTransform(static_transformStamped)
        rospy.sleep(0.002)
        #R_t= np.matmul(pytrans.quaternion_from_axis_angle([1, 0, 0, action_vector[5]]),
        #            pytrans.quaternion_from_axis_angle[0, 1, 0, action_vector[6]])

        #print(self.position)
        self.scene.set_pose("UAV",position)
        rospy.sleep(0.002)
        pcl=self.get_simulated_pcl()
        self.scene.set_pose("UAV",self.position)
        return pcl
        

