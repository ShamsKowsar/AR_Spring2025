#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict


final_position = {"x": 0.0, "y": 0.0}

def odom_callback(data):
    global final_position
    final_position["x"] = data.pose.pose.position.x
    final_position["y"] = data.pose.pose.position.y
    rospy.logdebug(f"Odom update: x={final_position['x']:.6f}, y={final_position['y']:.6f}")



def reset_robot_position():
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state_msg = ModelState()
        state_msg.model_name = 'robot'  
        state_msg.pose.position.x = 0
        state_msg.pose.position.y = 0
        state_msg.pose.position.z = 0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1
        state_msg.twist.linear.x = 0
        state_msg.twist.linear.y = 0
        state_msg.twist.linear.z = 0
        set_state(state_msg)
        rospy.sleep(1.0)  # wait for reset to take effect
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

def move():
    global final_position
    rospy.init_node('vector_controller', anonymous=True)
    velocity_publisher = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber("/odom", Odometry, odom_callback)
    vel_msg = Twist()

    speed = 0.02  # m/s
    duration_times = [2.5, 5.0, 7.5]  # seconds
    distances = [speed * t for t in duration_times]
    axis = input("Move along X or Y axis? (x/y): ").strip().lower()

    if axis == "x":
        direction = input("Forward? (yes/no): ").strip().lower()
        move_speed_x = speed if direction == "yes" else -speed
        move_speed_y = 0
    elif axis == "y":
        direction = input("Rightward? (yes/no): ").strip().lower()
        move_speed_y = speed if direction == "yes" else -speed
        move_speed_x = 0
    else:
        rospy.logerr("Invalid axis input.")
        return

    data = []
    rate = rospy.Rate(10)  # 10 Hz

    for duration, expected_distance in zip(duration_times, distances):
        rospy.loginfo(f"Starting 50 runs for duration {duration}s")

        for trial in range(50):
            # Reset position
            reset_robot_position()
            rospy.sleep(2.0)

            # Set direction
            vel_msg.linear.x = move_speed_x
            vel_msg.linear.y = move_speed_y
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0

            # Move robot using your preferred time-distance loop
            t0 = rospy.Time.now().to_sec()
            current_distance = 0
            d = expected_distance
            while current_distance < d:
                velocity_publisher.publish(vel_msg)
                t1 = rospy.Time.now().to_sec()
                current_distance = speed * (t1 - t0)
                rate.sleep()

            # Stop robot
            vel_msg.linear.x = 0
            vel_msg.linear.y = 0
            velocity_publisher.publish(vel_msg)

            # Allow odometry to update
            rospy.sleep(2.0)

            # Measure final displacement
            dx = final_position["x"]
            dy = final_position["y"]
            dt = t1 - t0
            final_speed = abs(dx / dt) if axis == "x" else abs(dy / dt)
            
            rospy.sleep(2.0)

            rospy.loginfo(f"[{duration}s] Trial {trial+1}/50: Speed = {final_speed*1000:.4f} (mm/s), X = {dx*1000:.4f} (mm), Y = {dy*1000:.4f} (mm)")

            data.append([duration, trial + 1, dx*1000, dy*1000, final_speed*1000])

    return data




def analyze_speed_errors(results):

    
    grouped = defaultdict(list)
    
    for row in results:
        duration, trial, x, y, measured_speed = row
        expected_speed = 20
        error = measured_speed - expected_speed
        grouped[duration].append(error)

    for duration, errors in grouped.items():
        errors = np.array(errors, dtype=np.float64)
        mean_error = np.mean(errors)
        var_error = np.var(errors)

        print(f"\n[Duration: {duration}s]")
        print(f"Mean speed error: {mean_error:.6f}")
        print(f"Variance of speed error: {var_error:.6f}")

        # plotting histogram
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(errors, kde=False, bins=15, color='skyblue')
        plt.title(f"Histogram of Speed Error (Duration {duration}s)")
        plt.xlabel("Speed Error (m/s)")
        plt.ylabel("Frequency")

        # plotting PDF
        plt.subplot(1, 2, 2)
        sns.kdeplot(errors, color='red', fill=True)
        plt.title(f"PDF of Speed Error (Duration {duration}s)")
        plt.xlabel("Speed Error (m/s)")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    try:
        results = move()

        with open("results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["duration", "trial", "x_position (mm)", "y_position (mm)", "measured_speed (mm/s)"])
            for row in results:
                writer.writerow(row)

        rospy.loginfo("Results saved to results.csv")
        
        analyze_speed_errors(results)

    except rospy.ROSInterruptException:
        pass



