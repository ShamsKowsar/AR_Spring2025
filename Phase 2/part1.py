#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Range
import csv

class ProximityDataCollector:
    def __init__(self, trials=500):
        self.trials = trials
        self.count = 0
        self.data_list = []
        self.subscriber = rospy.Subscriber('/vector/proximity', Range, self.callback)
        rospy.loginfo(f"Collecting {self.trials} proximity sensor readings...")

    def callback(self, data):
        if self.count < self.trials:
            distance = data.range  # Capture proximity measurement
            rospy.loginfo(f"Reading {self.count + 1}: {distance} meters")
            self.data_list.append(distance)
            self.count += 1
        else:
            self.subscriber.unregister()
            rospy.signal_shutdown("Finished collecting data.")

    def save_to_csv(self, filename="proximity_data.csv"):
        rospy.loginfo(f"Saving collected data to {filename}...")
        with open(filename, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Distance (meters)"])
            for entry in self.data_list:
                writer.writerow([entry])

if __name__ == '__main__':
    rospy.init_node('proximity_data_collector', anonymous=True)
    collector = ProximityDataCollector()
    rospy.spin()
    collector.save_to_csv()
    rospy.loginfo("Data collection completed and saved!")
