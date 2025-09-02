#!/usr/bin/env python3
import rospy
import math
import random
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import tf
from std_msgs.msg import String
import numpy as np
# Initialize ROS node
rospy.init_node("moving_box_controller")

# Wait for the Gazebo set_model_state service
rospy.wait_for_service("/gazebo/set_model_state")
set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

# Create a publisher for the box's mode
mode_pub = rospy.Publisher("box_mode", String, queue_size=10)

# Create a ModelState object for the box
state = ModelState()
state.model_name = "moving_box"
state.reference_frame = "world"

# Simulation parameters
rate = rospy.Rate(250)  # 250 Hz
dt = 1.0 / 250.0

# Initial state
x, y, theta = 0.2, 0.0, 0.0
v = 0.1      # initial velocity
a = 0.05     # acceleration
j=0.05
omega = 0.4  # angular speed for ARC

# Bounds
x_min, x_max = -5.0, 5.0
y_min, y_max = -5.0, 5.0

# Mode switching
modes = ['CA','CV']
mode = modes[0]
mode_duration = np.random.uniform(1,5)
last_switch = rospy.Time.now().to_sec()
mode_start_time = last_switch
rospy.loginfo(f"Starting mode: {mode}, duration: {mode_duration:.2f}s")
mode_pub.publish(mode) # Publish initial mode

# ARC mode state
arc_direction = random.choice(["left", "right"])
count=0
while not rospy.is_shutdown():
    tnow = rospy.Time.now().to_sec()
    elapsed = tnow - mode_start_time
    mode_pub.publish(mode) # Publish the new mode

    # Switch mode after random duration
    if tnow - last_switch > mode_duration:
        mode = "CA" if mode=="CV"  else "CV"
        count+=1
        if count >= len(modes):
            count = 0

        mode_duration = np.random.uniform(1,5)
        last_switch = tnow
        mode_start_time = tnow
        rospy.loginfo(f"Switched to mode: {mode}, next duration: {mode_duration:.2f}s")
        mode_pub.publish(mode) # Publish the new mode
        if mode == "ARC":
            arc_direction = random.choice(["left", "right"])
        elif mode == "CA":
            v = 0.0  # reset velocity for CA
        elif mode == "CJ":
            v = 0.0  # reset velocity for CA
            a=0.0

    # Motion models
    if mode == "CV":
        # Constant velocity in X direction
        x += v * dt

    elif mode == "CA":
        # Constant acceleration in X direction
        x += v * dt + 0.5 * a * dt**2
        v += a * dt  # update velocity
    elif mode == "CJ":
        # Constant acceleration in X direction
        x += v * dt + 0.5 * a * dt**2+1.0/6.0*j*dt**3
        v += a * dt+0.5*j*dt**2  # update velocity
        a += j * dt  # update velocity

    elif mode == "ARC":
        v=-0.03
        omega=0.1
        # Arc turn: curve left or right, reduce x
        if arc_direction == "left":
            theta += omega * dt
        else:
            theta -= omega * dt

        # Ensure heading reduces x
        if math.cos(theta) > 0:
            theta += math.pi

        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt

    # Fill ModelState for Gazebo
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = 0.0

    quat = tf.transformations.quaternion_from_euler(0, 0, theta)
    state.pose.orientation.x = quat[0]
    state.pose.orientation.y = quat[1]
    state.pose.orientation.z = quat[2]
    state.pose.orientation.w = quat[3]

    try:
        set_state(state)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

    rate.sleep()


