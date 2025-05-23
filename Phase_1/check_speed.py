#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
import time
import statistics  

PI = math.pi

theta = 0.0

def odom_callback(msg):
    global theta
    rot_q = msg.pose.pose.orientation
    (_, _, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))

def perform_rotation_test(speed_deg, duration, trial_num):
    global theta

    speed_rad = speed_deg * 2 * PI / 360  
    vel_msg = Twist()
    vel_msg.linear.x = 0
    vel_msg.angular.z = speed_rad

    theta_start = theta
    rospy.loginfo(f"[Trial {trial_num}] Start rotation: {speed_deg} deg/s for {duration} s")

    rate = rospy.Rate(10)
    start_time = rospy.Time.now()

    while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
        vel_pub.publish(vel_msg)
        rate.sleep()

    
    vel_msg.angular.z = 0
    vel_pub.publish(vel_msg)

    rospy.sleep(1.0)  

    theta_end = theta
    delta_theta = angle_diff(theta_end, theta_start)
    delta_deg = delta_theta * 180 / PI
    expected_deg = speed_deg * duration
    ratio = delta_deg / expected_deg if expected_deg != 0 else 0

    rospy.loginfo(f"[Trial {trial_num}] Expected rotation: {expected_deg:.2f} deg, Actual rotation: {delta_deg:.2f} deg, Ratio: {ratio:.3f}")

    return ratio

if __name__ == '__main__':
    rospy.init_node('rotation_speed_validation')
    rospy.Subscriber('/odom', Odometry, odom_callback)
    vel_pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)

    rospy.sleep(1.0)  

    speed_deg = 45.0    
    duration = 2.0      
    num_trials = 10      

    ratios = []
    for i in range(1, num_trials+1):
        ratio = perform_rotation_test(speed_deg, duration, i)
        ratios.append(ratio)
        rospy.sleep(2.0)  

    avg_ratio = statistics.mean(ratios)
    std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0.0

    rospy.loginfo(f"Average ratio actual/expected over {num_trials} trials: {avg_ratio:.3f}")
    rospy.loginfo(f"Standard deviation of ratio over {num_trials} trials: {std_ratio:.3f}")

    if abs(avg_ratio - 1.0) > 0.1:
        rospy.logwarn(f"You probably need to multiply commanded speed by ~{1/avg_ratio:.2f} to get actual rotation speed.")
    else:
        rospy.loginfo("Commanded speed matches actual rotation speed well.")
