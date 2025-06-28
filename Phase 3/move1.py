#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
import rosgraph_msgs.msg
import random

class MazeNavigator:
    def __init__(self):
        rospy.init_node('maze_navigator')
        rospy.wait_for_message('/clock', rosgraph_msgs.msg.Clock)

        self.pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/vector/laser', Range, self.callback)

        self.state = "forward"
        self.turn_direction = 1 

    def callback(self, scan):
        cmd = Twist()
        distance = scan.range

        if self.state == "forward":
            if distance < 0.2:
                # Obstacle ahead → turn
                rospy.loginfo("Obstacle detected. Turning...")
                self.state = "turn"
                self.turn_start_time = rospy.Time.now()
                self.turn_duration = rospy.Duration(random.uniform(1.5, 2.5))  # seconds
            else:
                cmd.linear.x = 0.2  # Move forward

        elif self.state == "turn":
            # Rotate for a short time
            cmd.angular.z = 0.5 * self.turn_direction
            if rospy.Time.now() - self.turn_start_time > self.turn_duration:
                rospy.loginfo("Finished turning. Going forward.")
                self.state = "forward"
                # Switch turn direction randomly to avoid loops
                self.turn_direction = random.choice([-1, 1])

        self.pub.publish(cmd)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    navigator = MazeNavigator()
    navigator.run()
