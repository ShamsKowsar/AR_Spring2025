#! /usr/bin/env python3
 
import rospy
from sensor_msgs.msg import Range
import csv
import os


    

class MeasureClass:
    def __init__(self, limit):
        self.count = 0
        self.limit = limit
        self.measures_list = []  # List to store incoming messages
        self.subscriber = None # To store the subscriber object


        rospy.loginfo(f"Subscribing to laser. Will process and collect {self.limit} messages.")
        self.subscriber = rospy.Subscriber('/vector/laser', Range, self.callback)

    def callback(self, data):
        """
        This function is called for every message received.
        It stores the message data until the limit is reached.
        """
        if not self.subscriber: 
            return

        if self.count < self.limit:
            measure = data.range 
            rospy.loginfo(f"Message {self.count + 1}/{self.limit}: Received '{measure}'")
            self.measures_list.append(measure) # Store the message content
            self.count += 1

            if self.count >= self.limit:

                if self.subscriber:
                    self.subscriber.unregister()
                    self.subscriber = None 

                rospy.signal_shutdown("Finished collecting Measures.")

    def get_collected_messages(self):
        """
        Returns the list of collected messages.
        """
        return self.measures_list

if __name__ == '__main__':
    try:
        distance_in_gazabo = input("What is the distance: ")
        rospy.init_node('laser_data_capture', anonymous=True)
        message_limit = 500
        
        collector = MeasureClass(message_limit)

        rospy.loginfo("Node started. Waiting for messages or shutdown signal...")

        rospy.spin()

    finally:

        if 'collector' in locals() and collector: 
            collected = collector.get_collected_messages()

        collected.insert(0, distance_in_gazabo)
        file_name = f"measures_{distance_in_gazabo}.csv"

        with open(file_name,"w",newline='') as file:
            writer = csv.writer(file)
            for item in collected:
                writer.writerow([item])


        
        rospy.loginfo("Node shutdown complete.")