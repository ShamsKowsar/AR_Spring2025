#!/usr/bin/env python3
import csv
import math
import time
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

PI = 3.1415926535897
measured_factor_from_check_speed = 0.493

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
        model_state.model_name = 'robot'
        model_state.pose.position.x = 0.0
        model_state.pose.position.y = 0.0
        model_state.pose.position.z = 0.0
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 1.0
        model_state.twist.linear.x = 0.0
        model_state.twist.linear.y = 0.0
        model_state.twist.linear.z = 0.0
        model_state.twist.angular.x = 0.0
        model_state.twist.angular.y = 0.0
        model_state.twist.angular.z = 0.0
        model_state.reference_frame = 'world'
        set_state(model_state)
        rospy.sleep(1.0)
    except rospy.ServiceException as e:
        rospy.logerr("Reset failed: %s" % e)

def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))

def rotate_by_time_and_log(csv_writer, trial_num, speed_deg, angle_deg, clockwise):
    global x, y, theta

    vel_pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    speed_rad = speed_deg * 2 * PI / 360 / measured_factor_from_check_speed
    duration = angle_deg / speed_deg

    vel_msg.linear.x = vel_msg.linear.y = vel_msg.linear.z = 0
    vel_msg.angular.x = vel_msg.angular.y = 0
    vel_msg.angular.z = -abs(speed_rad) if clockwise else abs(speed_rad)

    rospy.sleep(1.0)
    x0, y0, theta0 = x, y, theta

    real_t0 = time.time()
    sim_t0 = rospy.Time.now().to_sec()
    print(f"[Trial {trial_num}] Start Real Time: {real_t0:.3f} s, Simulation Time: {sim_t0:.3f} s")

    rate = rospy.Rate(10)
    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time).to_sec() < duration:
        vel_pub.publish(vel_msg)
        rate.sleep()

    vel_msg.angular.z = 0
    vel_pub.publish(vel_msg)

    rospy.sleep(1.0)
    x1, y1, theta1 = x, y, theta

    time.time()
    sim_t1 = rospy.Time.now().to_sec()

    dt = sim_t1 - sim_t0

    print(f"Trial {trial_num} | Final yaw: {theta1 * 180 / PI:.2f} degrees")

    mu = (
        0.5
        * ((x0 - x1) * math.cos(theta0) + (y0 - y1) * math.sin(theta0))
        / ((y0 - y1) * math.cos(theta0) - (x0 - x1) * math.sin(theta0))
    )
    x_star = (x0 + x1) / 2 + mu * (y0 - y1)
    y_star = (y0 + y1) / 2 + mu * (x1 - x0)
    r_star = math.sqrt((x0 - x_star) ** 2 + (y0 - y_star) ** 2)

    delta_theta = math.atan2(y1 - y_star, x1 - x_star) - math.atan2(y0 - y_star, x0 - x_star)

    omega_hat = delta_theta / dt
    v_hat = omega_hat * r_star
    gamma = ((theta1 - theta0) / dt) - omega_hat

    csv_writer.writerow([
        trial_num,
        clockwise,
        speed_deg,
        angle_deg,
        x0,
        y0,
        theta0,
        x1,
        y1,
        theta1,
        x_star,
        y_star,
        r_star,
        delta_theta,
        omega_hat,
        v_hat,
        gamma,
    ])

    reset_robot_pose()

if __name__ == '__main__':
    speed = int(input('Speed (in degrees/sec): '))
    rospy.init_node("rotation_logger", anonymous=True)
    rospy.Subscriber("/odom", Odometry, odom_callback)
    result_path = f'rotation_data_{speed}.csv'

    with open(result_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial',
            'Clockwise',
            'Speed(deg/s)',
            'Inputtheta(deg)',
            'x0',
            'y0',
            'theta0',
            'x1',
            'y1',
            'theta1',
            'x*',
            'y*',
            'r*',
            'delta_theta',
            'omega_hat',
            'v_hat',
            'gamma',
        ])
        reset_robot_pose()
        rospy.sleep(1.0)

        print('Clockwise Rotations:')
        for i in range(20):
            rotate_by_time_and_log(writer, i + 1, speed_deg=speed, angle_deg=90, clockwise=True)
            rospy.sleep(1.0)

        print('Counter-Clockwise Rotations:')
        for i in range(20):
            rotate_by_time_and_log(writer, i + 21, speed_deg=speed, angle_deg=90, clockwise=False)
            rospy.sleep(1.0)

