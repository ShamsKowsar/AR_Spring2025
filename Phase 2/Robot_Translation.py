#!/usr/bin/env python3

from time import sleep

import rospy
from rospy import Publisher
from anki_vector_ros.msg import RobotStatus
from anki_vector_ros.msg import Drive


def main():
    print("Setting up publishers")
    move_pub = Publisher("/motors/wheels", Drive, queue_size=1)

    sleep(0.5)

    speed = float(input("what is the speed (mm/sec)?"))
    dx = float(input("what is the distance? 50, 100 or 150 mm?"))
    dt = dx/speed    # sec


    print(f"Moving {dx} mm")

    move_pub.publish(speed, speed, 0.0, 0.0)
    sleep(dt)
    move_pub.publish(0.0, 0.0, 0.0, 0.0)



if __name__ == "__main__":
    rospy.init_node("vector_translation")
    rospy.wait_for_message("/status", RobotStatus)

    main()
