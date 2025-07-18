#!/usr/bin/env python3
import rospy
import random
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point as ShapelyPoint # Renamed to avoid conflict
from shapely.affinity import rotate, translate
import tf.transformations

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion # Point here refers to geometry_msgs.msg.Point
DX, DY        = +0.03914 ,+0.521493  # metres to shift (+x right, +y up)

def parse_world_file(world_file_path):
    """
    Parses a Gazebo .world file to extract the root element.
    """
    try:
        tree = ET.parse(world_file_path)
        return tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        rospy.logerr(f"Error reading or parsing world file: {e}")
        return None

def extract_walls_from_model(root, model_name="vector_world_4"):
    """
    Extracts wall polygons from a specific model within the world file.
    """
    wall_polygons = []
    if root is None:
        return wall_polygons
    model = root.find(f".//model[@name='{model_name}']")
    if model is None:
        rospy.logwarn(f"Model '{model_name}' not found.")
        return wall_polygons
    
    # Define a small buffer for robot size to avoid spawning too close to walls
    # This should be slightly larger than half the robot's width/length to prevent overlap
    # Adjust this value based on your robot's dimensions. A good starting point is
    # half of your robot's diagonal size to ensure the entire robot fits.
    robot_buffer = 0.15 # Increased slightly for more clearance

    for link in model.findall("link"):
        pose_tag, collision = link.find("pose"), link.find("collision")
        if collision is None:
            continue
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        box = geometry.find("box")
        if box is None:
            continue
        size_tag = box.find("size")
        if pose_tag is None or size_tag is None:
            continue
        
        # Parse pose: x, y, z, roll, pitch, yaw
        pose_values = list(map(float, pose_tag.text.strip().split()))
        x, y, _, _, _, yaw = pose_values

        length, width, _ = map(float, size_tag.text.strip().split())
        
        # Adjust dimensions by adding buffer to make the "wall" larger for collision checking.
        # This effectively shrinks the valid free space.
        buffered_length = length + 2 * robot_buffer
        buffered_width = width + 2 * robot_buffer

        dx, dy = buffered_length / 2, buffered_width / 2
        
        rect = Polygon([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]])
        rect = rotate(rect, np.degrees(yaw), origin=(0, 0))
        rect = translate(rect, xoff=x, yoff=y)
        wall_polygons.append(rect)
    return wall_polygons

def is_position_valid(x, y, wall_polygons):
    """
    Checks if a given (x, y) position is within any of the wall polygons.
    Returns True if valid (not on a wall), False otherwise.
    """
    # Use ShapelyPoint for geometry calculations
    point = ShapelyPoint(x, y)
    for wall in wall_polygons:
        # Check if the point is inside the buffered wall polygon or too close to its boundary
        if wall.contains(point) or wall.exterior.distance(point) < 1e-6:
            return False
    return True

def set_robot_position_randomly():
    rospy.init_node('set_robot_position_node', anonymous=True)
    rospy.wait_for_service('/gazebo/set_model_state')

    # Define the path to your world file
    # This should match the world file loaded in Gazebo
    world_file_path = rospy.get_param('~world_file', "/home/kowsar/catkin_ws/src/anki_description/world/sample1.world")
    robot_model_name = rospy.get_param('~robot_model_name', 'vector') # Your robot's model name

    root = parse_world_file(world_file_path)
    if root is None:
        rospy.logerr("Could not parse world file. Exiting.")
        return

    wall_polygons = extract_walls_from_model(root)
    if not wall_polygons:
        rospy.logwarn("No walls extracted from the specified model. Robot might spawn on unexpected areas or outside the maze bounds.")
        rospy.logwarn("Please ensure 'model_name' in world file corresponds to 'vector_world_4' or update the 'model_name' parameter in the script.")
        # If no walls are found, define a default large search area as a fallback
        min_x, max_x = -5.0, 5.0
        min_y, max_y = -5.0, 5.0
    else:
        # Determine the overall bounds of the maze from the walls' outer limits
        all_x_coords = []
        all_y_coords = []
        for poly in wall_polygons:
            all_x_coords.extend(list(poly.exterior.xy[0]))
            all_y_coords.extend(list(poly.exterior.xy[1]))

        # Add a margin to ensure the random points are well within the maze's playable area
        # This margin should be at least the 'robot_buffer' to allow for robot size.
        margin = 0.2 # A bit larger than robot_buffer for extra safety

        min_x = min(all_x_coords) + margin
        max_x = max(all_x_coords) - margin
        min_y = min(all_y_coords) + margin
        max_y = max(all_y_coords) - margin

        # Basic check to ensure bounds are not inverted due to small maze or large margin
        if min_x > max_x: min_x, max_x = max_x, min_x
        if min_y > max_y: min_y, max_y = max_y, min_y

        rospy.loginfo(f"Determined maze bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")


    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        found_valid_position = False
        attempts = 0
        max_attempts = 1000

        while not found_valid_position and attempts < max_attempts:
            # Generate random x, y within the calculated map bounds
            random_x = random.uniform(min_x, max_x)+DX
            random_y = random.uniform(min_y, max_y)+DY

            # Check if the generated point is valid (not on a wall)
            if is_position_valid(random_x, random_y, wall_polygons):
                # Generate a random yaw (orientation)
                random_yaw = random.uniform(-np.pi, np.pi)
                
                pose = Pose()
                pose.position = Point(x=random_x, y=random_y, z=0) # Assuming Z=0 for a 2D maze on the ground
                
                # Convert yaw to quaternion using tf.transformations
                q = tf.transformations.quaternion_from_euler(0, 0, random_yaw)
                pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                
                model_state = ModelState()
                model_state.model_name = robot_model_name
                model_state.pose = pose
                
                resp = set_state(model_state)
                if resp.success:
                    rospy.loginfo(f"Robot position set to ({random_x:.2f}, {random_y:.2f}, 0) with yaw {np.degrees(random_yaw):.2f} degrees after {attempts + 1} attempts.")
                    found_valid_position = True
                else:
                    rospy.logwarn(f"Failed to set model state for ({random_x:.2f}, {random_y:.2f}): {resp.status_message}. Retrying...")
            
            attempts += 1

        if not found_valid_position:
            rospy.logerr(f"Could not find a valid random position after {max_attempts} attempts. This might indicate issues with map bounds or an overly restrictive 'robot_buffer'.")
            
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    try:
        set_robot_position_randomly()
    except rospy.ROSInterruptException:
        pass
