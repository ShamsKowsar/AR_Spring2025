#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math

class StepwiseNavigatorOdom:
    def __init__(self):
        rospy.init_node('stepwise_navigator_odom')

        self.cmd_pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/vector/laser', Range, self.sensor_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.obstacle_detected = False
        self.min_safe_distance = 0.25
        self.current_yaw = 0.0

        rospy.sleep(1)  # Let subscribers start
        self.run()

    def sensor_callback(self, msg):
        self.obstacle_detected = msg.range < self.min_safe_distance

    def odom_callback(self, msg):
        # Convert quaternion to yaw
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.current_yaw = yaw

    def move_forward(self, speed=0.02, duration=5):
        twist = Twist()
        twist.linear.x = speed
        self.cmd_pub.publish(twist)
        rospy.sleep(duration)
        self.stop()

    def stop(self):
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.1)

    def normalize_angle(self, angle):
        """ Normalize angle to [-pi, pi] """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def turn_left_90_degrees(self, angular_speed=0.3):
        initial_yaw = self.current_yaw
        target_yaw = self.normalize_angle(initial_yaw + math.radians(90))

        twist = Twist()
        twist.angular.z = angular_speed

        rospy.loginfo("Turning 90 degrees...")

        rate = rospy.Rate(20)  # 20 Hz
        while not rospy.is_shutdown():
            error = self.normalize_angle(target_yaw - self.current_yaw)

            if abs(error) < math.radians(2):  # within 2 degrees
                break

            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop()
        rospy.loginfo("Turn complete.")

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.move_forward(speed=0.02, duration=5)

            if self.obstacle_detected:
                rospy.loginfo("Obstacle detected. Turning...")
                self.turn_left_90_degrees()
            else:
                rospy.loginfo("No obstacle. Continuing forward.")

            rate.sleep()

if __name__ == "__main__":
    StepwiseNavigatorOdom()
