#!/usr/bin/env python3
"""Sample program to make Vector move and log rotation data."""

from time import sleep
import rospy
from rospy import Publisher
from anki_vector_ros.msg import RobotStatus
from anki_vector_ros.msg import Drive
from calculate_hyperparams import compute_rotation_parameters

import csv
import os
from datetime import datetime

def log_to_csv(file_path, omega, direction, time_needed, left_speed, right_speed):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Omega (rad/s)', 'Direction', 'Time (s)', 'Left Speed (cm/s)', 'Right Speed (cm/s)'])

        writer.writerow([
            datetime.now().isoformat(),
            omega,
            direction,
            round(time_needed, 3),
            round(left_speed, 3),
            round(right_speed, 3)
        ])

def main():
    print("Setting up publishers")
    move_pub = Publisher("/vector/motors/wheels", Drive, queue_size=1)

    omega = float(input('Angular Speed (rad/s): '))
    direction = input("Rotation direction (CW/CCW): ").strip().upper()

    delta_t, speed_l, speed_r = compute_rotation_parameters(omega, direction)

    # Log this run to CSV
    csv_file = 'rotation_data.csv'
    log_to_csv(csv_file, omega, direction, delta_t, speed_l, speed_r)

    # Wait for publishers to be ready
    sleep(0.5)

    print("Executing commands")
    move_pub.publish(speed_l * 10, speed_r * 10, 0.0, 0.0)
    sleep(delta_t)
    move_pub.publish(0.0, 0.0, 0.0, 0.0)
    print("Rotation complete and data logged.")

if __name__ == "__main__":
    rospy.init_node("vector_hello_world")
    rospy.wait_for_message("/vector/status", RobotStatus)

    main()
