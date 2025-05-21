#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import math
import csv
import time

PI = 3.1415926535897

# Global pose variables
x = 0.0
y = 0.0
theta = 0.0

def odom_callback(msg):
    global x, y, theta
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    rot_q = msg.pose.pose.orientation
    (_, _, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

def reset_robot_pose():
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        model_state = ModelState()
        model_state.model_name = 'vector'  # ⚠️ Replace with your actual model name
        model_state.pose.position.x = 0.0
        model_state.pose.position.y = 0.0
        model_state.pose.position.z = 0.0
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 1.0  # Identity quaternion
        model_state.reference_frame = 'world'
        set_state(model_state)
        rospy.sleep(1.0)
    except rospy.ServiceException as e:
        rospy.logerr("Reset failed: %s" % e)

def rotate_and_log(csv_writer, trial_num, speed_deg, angle_deg, clockwise):
    global x, y, theta

    vel_pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    speed_rad = speed_deg * 2 * PI / 360
    angle_rad = angle_deg * 2 * PI / 360
    vel_msg.angular.z = -abs(speed_rad) if clockwise else abs(speed_rad)

    vel_msg.linear.x = vel_msg.linear.y = vel_msg.linear.z = 0
    vel_msg.angular.x = vel_msg.angular.y = 0

    rospy.sleep(1.0)  # Wait for pose to settle
    x0, y0, theta0 = x, y, theta
    t0 = rospy.Time.now().to_sec()

    current_angle = 0
    rate = rospy.Rate(100)
    while current_angle < angle_rad:
        vel_pub.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_angle = speed_rad * (t1 - t0)
        rate.sleep()

    vel_msg.angular.z = 0
    vel_pub.publish(vel_msg)
    rospy.sleep(1.0)

    x1, y1, theta1 = x, y, theta
    t1 = rospy.Time.now().to_sec()
    dt = t1 - t0

    # === Compute variables
    mu = 0.5 * ((x0 - x1) * math.cos(theta0) + (y0 - y1) * math.sin(theta0))
    x_star = (x0 + x1) / 2 + mu * (y0 - y1)
    y_star = (y0 + y1) / 2 + mu * (x1 - x0)
    r_star = math.sqrt((x0 - x_star)**2 + (y0 - y_star)**2)

    angle1 = math.atan2(y1 - y_star, x1 - x_star)
    angle0 = math.atan2(y0 - y_star, x0 - x_star)
    delta_theta = angle1 - angle0
    delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))  # normalize

    v_hat = delta_theta / dt
    v_expected = delta_theta / (dt * r_star) if r_star != 0 else 0
    gamma = ((theta1 - theta0) / dt) - v_hat

    # === Log to CSV
    csv_writer.writerow([
        trial_num, clockwise, speed_deg, angle_deg,
        x0, y0, theta0, x1, y1, theta1,
        x_star, y_star, r_star, delta_theta, v_hat, v_expected, gamma
    ])

    # === Reset robot
    reset_robot_pose()

if __name__ == '__main__':
    rospy.init_node("rotation_logger", anonymous=True)
    rospy.Subscriber("/odom", Odometry, odom_callback)

    with open('rotation_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial', 'Clockwise', 'InputSpeed(deg/s)', 'InputAngle(deg)',
            'x0', 'y0', 'theta0', 'x1', 'y1', 'theta1',
            'x*', 'y*', 'r*', 'Δtheta', 'v_hat', 'v_expected', 'gamma'
        ])

        rospy.loginfo("Running 20 CW trials...")
        for i in range(20):
            rotate_and_log(writer, i+1, speed_deg=20, angle_deg=90, clockwise=True)
            rospy.sleep(1.0)

        rospy.loginfo("Running 20 CCW trials...")
        for i in range(20):
            rotate_and_log(writer, i+21, speed_deg=20, angle_deg=90, clockwise=False)
            rospy.sleep(1.0)

    rospy.loginfo("✅ All trials completed and saved to 'rotation_data.csv'")
