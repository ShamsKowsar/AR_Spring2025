#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math

DEFINED_CONSTANT = 0.493

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
        """Moves the robot forward for a specified duration."""
        twist = Twist()
        twist.linear.x = speed
        self.cmd_pub.publish(twist)
        rospy.sleep(duration)
        self.stop()

    def stop(self):
        """Stops the robot's movement."""
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.1)

    def normalize_angle(self, angle):
        """Normalizes an angle to the range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def turn_left_90_degrees(self, angular_speed=0.3):
        """
        Turns the robot approximately 90 degrees to the left,
        adjusting the angular speed by the DEFINED_CONSTANT,
        using odometry feedback.
        """
        initial_yaw = self.current_yaw
        
        # Target yaw is now exactly 90 degrees from the initial yaw,
        # as the DEFINED_CONSTANT is applied to the speed, not the target angle.
        target_yaw = self.normalize_angle(initial_yaw + math.radians(90))

        twist = Twist()
        twist.angular.z = angular_speed / DEFINED_CONSTANT # Positive for left turn

        rospy.loginfo(f"Turning 90 degrees with adjusted speed (current wheel speed: {angular_speed / DEFINED_CONSTANT:.2f} rad/s)...")

        rate = rospy.Rate(20)  # 20 Hz loop rate for smooth turning
        while not rospy.is_shutdown():
            # Calculate the error between target and current yaw
            error = self.normalize_angle(target_yaw - self.current_yaw)

            # If the error is small enough, the turn is complete
            if abs(error) < math.radians(2):  # within 2 degrees tolerance
                break

            # Publish the turning command
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop() # Stop the robot after the turn
        rospy.loginfo("Turn complete.")

    def run(self):
        """Main loop for the navigator."""
        rate = rospy.Rate(1) # Run the main loop at 1 Hz
        while not rospy.is_shutdown():
            self.move_forward(speed=0.02, duration=5) # Move forward for 5 seconds

            if self.obstacle_detected:
                rospy.loginfo("Obstacle detected. Initiating turn...")
                self.turn_left_90_degrees() # Perform the adjusted turn
            else:
                rospy.loginfo("No obstacle. Continuing forward.")

            rate.sleep() # Wait for the next loop iteration

if __name__ == "__main__":
    StepwiseNavigatorOdom()

