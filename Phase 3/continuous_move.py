#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
import rosgraph_msgs.msg
import random
import math
DEFINED_CONSTANT=0.493
class MazeNavigator:
    def __init__(self):
        rospy.init_node('maze_navigator')
        rospy.wait_for_message('/clock', rosgraph_msgs.msg.Clock)

        self.pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/vector/laser', Range, self.callback)

        self.state = "forward"
        self.turn_direction = 1

        self.safe_distance = 0.1  
        self.turn_speed = math.pi/3  

    def callback(self, scan):
        cmd = Twist()
        distance = scan.range

        if self.state == "forward":
            if distance < self.safe_distance:
                rospy.loginfo("Obstacle detected. Initiating 90° turn...")
                self.state = "turn"
                self.turn_start_time = rospy.Time.now()

                
                angle_rad = math.radians(90/DEFINED_CONSTANT)
                self.turn_duration = rospy.Duration(angle_rad / self.turn_speed)

            else:
                cmd.linear.x = 0.02  

        elif self.state == "turn":
            cmd.angular.z = self.turn_speed * self.turn_direction
            if rospy.Time.now() - self.turn_start_time > self.turn_duration:
                rospy.loginfo("Finished 90° turn. Moving forward.")
                self.state = "forward"

        self.pub.publish(cmd)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    navigator = MazeNavigator()
    navigator.run()
