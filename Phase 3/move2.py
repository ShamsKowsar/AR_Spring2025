#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion

def set_robot_position():
    rospy.init_node('set_robot_position_node', anonymous=True)
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        model_state = ModelState()
        model_state.model_name = 'vector'  # Change 'vector' to your robot's model name
        
        pose = Pose()
        pose.position = Point(x=0, y=0.2, z=0)
        pose.orientation = Quaternion(x=0, y=0, z=0, w=1) # No rotation
        
        model_state.pose = pose
        
        resp = set_state(model_state)
        
        rospy.loginfo("Robot position set to (0, 0.2, 0)")
        
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

if __name__ == '__main__':
    set_robot_position()

