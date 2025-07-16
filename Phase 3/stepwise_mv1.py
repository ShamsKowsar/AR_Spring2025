#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
import rosgraph_msgs.msg
import math

DEFINED_CONSTANT = 0.493

class MazeNavigator:
    def __init__(self):
        rospy.init_node('maze_navigator')
        rospy.wait_for_message('/clock', rosgraph_msgs.msg.Clock)

        self.pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/vector/laser', Range, self.callback)

        self.state = "forward"
        self.turn_direction = 1

        self.safe_distance = 0.01  # Distance threshold to move forward
        self.turn_speed = math.pi / 3  # 60 deg/s

        self.forward_speed = 0.02  # Linear speed forward (m/s)
        self.forward_duration = rospy.Duration(5.0)  # 5 seconds forward

        self.forward_start_time = None  # When started moving forward
        self.turn_start_time = None
        self.turn_duration = None

    def callback(self, scan):
        cmd = Twist()
        distance = scan.range
        current_time = rospy.Time.now()

        if self.state == "forward":
            # If we just started moving forward, set the start time
            if self.forward_start_time is None:
                self.forward_start_time = current_time
                rospy.loginfo("Starting forward movement for up to 5 seconds.")

            # Check if obstacle is closer than safe distance
            if distance < self.safe_distance:
                rospy.loginfo("Obstacle detected. Initiating 90° turn...")
                self.state = "turn"
                self.turn_start_time = current_time

                # Turn duration = angle / angular speed
                # Correct the angle to 90 degrees in radians (no DEFINED_CONSTANT here)
                angle_rad = math.radians(90)
                self.turn_duration = rospy.Duration(angle_rad / self.turn_speed)

                # Stop forward timer
                self.forward_start_time = None

                # Stop forward movement and start turning
                cmd.angular.z = self.turn_speed * self.turn_direction

            else:
                # Continue moving forward if time not exceeded
                elapsed = current_time - self.forward_start_time
                if elapsed < self.forward_duration:
                    cmd.linear.x = self.forward_speed
                else:
                    rospy.loginfo("Finished 5 seconds forward movement without obstacles.")
                    # Optionally, you can stop or do something else here
                    cmd.linear.x = 0.0
                    # Reset forward_start_time to None to start counting again if needed
                    self.forward_start_time = None

        elif self.state == "turn":
            cmd.angular.z = self.turn_speed * self.turn_direction/DEFINED_CONSTANT
            if current_time - self.turn_start_time > self.turn_duration:
                rospy.loginfo("Finished 90° turn. Moving forward.")
                self.state = "forward"
                self.forward_start_time = None  # Reset forward timer for next move

        self.pub.publish(cmd)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    navigator = MazeNavigator()
    navigator.run()
