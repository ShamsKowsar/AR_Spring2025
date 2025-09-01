#!/usr/bin/env python3
"""
ROS node for creating a map from odometry and laser scans.
- Subscribes: /odom (nav_msgs/Odometry), /scan (sensor_msgs/LaserScan)
- Publishes:  /map (nav_msgs/OccupancyGrid), /map_pose (geometry_msgs/PoseStamped),
            /map_path (nav_msgs/Path)
"""
import rospy
import numpy as np
import tf
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from mapping import OccupancyGridMap
from utils import wrap_to_pi


# ---------------- Parameters ----------------
MAP_SIZE_X = 500
MAP_SIZE_Y = 500
MAP_RESOLUTION = 0.02
MAP_ORIGIN = (-MAP_SIZE_X * MAP_RESOLUTION / 2.0, -MAP_SIZE_Y * MAP_RESOLUTION / 2.0)
LASER_MAX_RANGE = 4.0
MAP_PUB_HZ = 2.0
MAP_SAVE_PATH = '/tmp/odometry_map.pgm'
# -------------------------------------------


class OdometryMappingNode:
   def __init__(self):
       rospy.init_node('odometry_mapping_node', anonymous=True)


       self.map = OccupancyGridMap(
           size_x=MAP_SIZE_X, size_y=MAP_SIZE_Y,
           resolution=MAP_RESOLUTION, origin=MAP_ORIGIN
       )


       self.br = tf.TransformBroadcaster()


       odom_topic = rospy.get_param('~odom_topic', '/odom')
       scan_topic = rospy.get_param('~scan_topic', '/scan')
       rospy.Subscriber(odom_topic, Odometry, self.odom_cb, queue_size=50)
       rospy.Subscriber(scan_topic, LaserScan, self.scan_cb, queue_size=5)


       self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1, latch=True)
       self.pose_pub = rospy.Publisher('/map_pose', PoseStamped, queue_size=1)
       self.path_pub = rospy.Publisher('/map_path', Path, queue_size=1)


       # State variables
       self.last_corrected_odom = None
       self.last_raw_odom = None
       self.initial_yaw_offset = None


       self.path_msg = Path()
       self.path_msg.header.frame_id = 'map'


       self.map_timer = rospy.Timer(rospy.Duration(1.0 / MAP_PUB_HZ), self.publish_map_timer)


       rospy.on_shutdown(self.on_shutdown)
       rospy.loginfo("Odometry Mapping node initialized.")


   def odom_cb(self, msg: Odometry):
       p = msg.pose.pose
       x, y = p.position.x, p.position.y

       orientation_list = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
       (roll, pitch, current_yaw) = tf.transformations.euler_from_quaternion(orientation_list)


       # Store the raw, original odometry for TF publishing
       self.last_raw_odom = (x, y, current_yaw)


       if self.initial_yaw_offset is None:
           self.initial_yaw_offset = current_yaw


       # Create the corrected, world-aligned odometry for mapping
       corrected_yaw = wrap_to_pi(current_yaw - self.initial_yaw_offset)
       self.last_corrected_odom = (x, y, corrected_yaw)


   def scan_cb(self, msg: LaserScan):
       if self.last_corrected_odom is None:
           return


       # The robot's pose is now trusted directly from the corrected odometry
       odom_pose = self.last_corrected_odom

       ranges = np.array(msg.ranges, dtype=np.float32)
       angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment


       # Update the map and publish everything using the odometry pose
       self.map.update_by_scan(odom_pose, ranges, angles, LASER_MAX_RANGE, beam_step=2)
       self.publish_tf(odom_pose)
       self.publish_pose_and_path(odom_pose)


   def publish_tf(self, mapping_pose):
       if self.last_raw_odom is None:
           return


       map_x, map_y, map_th = mapping_pose
       odom_x, odom_y, odom_th = self.last_raw_odom


       T_map_baselink = tf.transformations.euler_matrix(0, 0, map_th)
       T_map_baselink[0, 3] = map_x
       T_map_baselink[1, 3] = map_y


       T_odom_baselink = tf.transformations.euler_matrix(0, 0, odom_th)
       T_odom_baselink[0, 3] = odom_x
       T_odom_baselink[1, 3] = odom_y


       T_baselink_odom = tf.transformations.inverse_matrix(T_odom_baselink)
       T_map_odom = np.dot(T_map_baselink, T_baselink_odom)


       trans = tf.transformations.translation_from_matrix(T_map_odom)
       quat = tf.transformations.quaternion_from_matrix(T_map_odom)


       self.br.sendTransform(trans, quat, rospy.Time.now(), "odom", "map")


   def publish_pose_and_path(self, pose_xyz):
       x, y, th = pose_xyz
       ps = PoseStamped()
       ps.header.stamp = rospy.Time.now()
       ps.header.frame_id = 'map'
       ps.pose.position.x = float(x)
       ps.pose.position.y = float(y)
       qx, qy, qz, qw = tf.transformations.quaternion_from_euler(0.0, 0.0, float(th))
       ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
       self.pose_pub.publish(ps)
       self.path_msg.header.stamp = ps.header.stamp
       self.path_msg.poses.append(ps)
       self.path_pub.publish(self.path_msg)


   def publish_map_timer(self, _evt):
       """
       Converts the OccupancyGridMap object into a nav_msgs/OccupancyGrid message
       and publishes it.
       """
       prob = self.map.get_prob_map() # Get the probability map (a NumPy array)

       og = OccupancyGrid()
       og.header.stamp = rospy.Time.now()
       og.header.frame_id = 'map'

       # Set the map metadata
       og.info.resolution = self.map.resolution
       og.info.width = self.map.size_x
       og.info.height = self.map.size_y
       og.info.origin.position.x = self.map.origin[0]
       og.info.origin.position.y = self.map.origin[1]

       # Convert the probability values (0.0 to 1.0) into the required
       # integer format (-1 for unknown, 0 for free, 100 for occupied).
       data = np.full(prob.shape, -1, dtype=np.int8)
       data[prob < 0.35] = 0   # Free space threshold
       data[prob > 0.65] = 100 # Occupied space threshold

       # The data needs to be a flat list for the message
       og.data = list(data.flatten())

       self.map_pub.publish(og)

   def on_shutdown(self):
       try:
           self.map.to_pgm(MAP_SAVE_PATH)
           rospy.loginfo("Saved map to %s", MAP_SAVE_PATH)
       except Exception as e:
           rospy.logerr("Failed to save map: %s", e)


   def run(self):
       rospy.spin()


if __name__ == '__main__':
   try:
       node = OdometryMappingNode()
       node.run()
   except rospy.ROSInterruptException:
       pass


