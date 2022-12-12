import numpy as np



class Actions():

    def __init__(self, rad):



	    self.ACTIONS2D=np.array([

		[1,0,0,0,0,0], #Up

		[-1,0,0,0,0,0], #Down

		[0,-1,0,0,0,0], #Left

		[0,1,0,0,0,0], #Right



		[0,0,0,0,0,-rad], #Rotate right

		[0,0,0,0,0,rad]]) #Rotate left





	# action_state=np.array([[1,0,0,0,0,-angle_camera* (PI/ 180.0)],

	#                        [-1,0,0,0,0,-angle_camera* (PI/ 180.0)],

	#                        [0,-1,0,0,0,-angle_camera* (PI/ 180.0)],

	#                        [0,1,0,0,0,-angle_camera* (PI/ 180.0)],



	#                        [1,0,0,0,0,0* (PI/ 180.0)],

	#                        [-1,0,0,0,0,0* (-PI/ 180.0)],

	#                        [0,-1,0,0,0,0* (-PI/ 180.0)],

	#                        [0,1,0,0,0,0* (-PI/ 180.0)],



	#                        [1,0,0,0,0,angle_camera* (PI/ 180.0)],

	#                        [-1,0,0,0,0,angle_camera* (PI/ 180.0)],

	#                        [0,-1,0,0,0,angle_camera* (PI/ 180.0)],

	#                        [0,1,0,0,0,angle_camera* (PI/ 180.0)]])

