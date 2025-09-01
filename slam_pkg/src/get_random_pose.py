#!/usr/bin/env python3
import rospy
import random
import os

# Maze bounds (adjust to your maze size)
X_MIN, X_MAX = -0.5, 0.5
Y_MIN, Y_MAX = 0.0, 0.5

x = random.uniform(X_MIN, X_MAX)
y = random.uniform(Y_MIN, Y_MAX)
yaw = random.uniform(-3.14, 3.14)

os.system(f"rosrun gazebo_ros spawn_model -param robot_description -urdf -model vector -x {x} -y {y} -z 0.0 -Y {yaw}")


