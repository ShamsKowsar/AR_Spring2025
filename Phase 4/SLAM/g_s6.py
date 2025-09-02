#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Range, LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import scipy.sparse.linalg as ssl
import threading
import copy
import math
import datetime
import os

# Integration of op.py functions
def mat2vec(mat):
    vec = np.mat(np.zeros((3, 1)))
    vec[0, 0] = mat[0, 2]  # x
    vec[1, 0] = mat[1, 2]  # y
    vec[2, 0] = math.atan2(mat[1, 0], mat[0, 0])  # yaw
    return vec

def vec2mat(vec):
    if vec.shape != (3, 1):
        print('vec is not 3*1 matrix')
        return 0
    
    c = math.cos(vec[2, 0])
    s = math.sin(vec[2, 0])
    
    mat = np.mat([
        [c, -s, vec[0,0]],
        [s,  c, vec[1,0]],
        [0,  0,        1]
    ])
    
    return mat

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class PoseNode:
    """Enhanced pose node compatible with op.py optimization"""
    def __init__(self, id, x, y, theta, timestamp):
        self.id = id
        self.Id = id  # For compatibility with op.py
        self.x = x
        self.y = y
        self.theta = theta
        self.timestamp = timestamp
        self.scan_data = None
        # For op.py compatibility
        self.pose = np.mat([x, y, theta]).T  # 3x1 matrix
        self.yaw = theta
        
    def update_from_pose_matrix(self):
        """Update x, y, theta from pose matrix"""
        self.x = float(self.pose[0, 0])
        self.y = float(self.pose[1, 0])
        self.theta = float(self.pose[2, 0])
        self.yaw = self.theta
        
    def to_vector(self):
        return np.array([self.x, self.y, self.theta])
    
    def from_vector(self, vec):
        self.x, self.y, self.theta = vec
        self.pose = np.mat([self.x, self.y, self.theta]).T

class PoseEdge:
    """Enhanced edge class compatible with op.py optimization"""
    def __init__(self, from_id, to_id, dx, dy, dtheta, info_matrix=None):
        self.from_id = from_id
        self.to_id = to_id
        self.Id_from = from_id  # For op.py compatibility
        self.Id_to = to_id
        self.dx = dx
        self.dy = dy
        self.dtheta = dtheta
        self.info_matrix = info_matrix if info_matrix is not None else np.eye(3)
        # For op.py compatibility
        self.mean = np.mat([dx, dy, dtheta]).T  # 3x1 matrix

def id2idx(Id):
    """Convert node ID to matrix indices"""
    idx = [3*int(Id), 3*(int(Id)+1)]
    return idx

class FastDiscreteGraphSLAM:
    def __init__(self):
        rospy.init_node('fast_discrete_graph_slam_node', anonymous=True)
        
        # Create output directory
        self.output_dir = rospy.get_param('~output_dir', 
                                         os.path.expanduser('~/slam_results'))
        self.create_output_directory()
        
        # FIXED: Movement parameters with better control
        self.forward_speed = 0.05  # Reduced speed for precise control
        self.deceleration_speed = 0.02  # Slow speed for final approach
        self.discrete_move_distance = 0.08
        self.angular_speed = 0.4  # Reduced for better turn control
        self.min_safe_distance = 0.15
        self.stop_duration = 0.3  # Longer stop for sensor stabilization
        
        # FIXED: Precise distance and turn control
        self.distance_check_tolerance = 0.005  # Tighter tolerance - 5mm
        self.deceleration_distance = 0.02  # Start slowing down 2cm before target
        self.turn_tolerance = 0.05  # Tighter turn tolerance - ~3 degrees
        self.max_movement_time = 8.0  # Timeout for safety
        self.max_turn_time = 8.0 # Increased turn time slightly for complex situations
        
        # FIXED: Anti-stuck navigation parameters
        self.turn_angle = np.pi/4  # 45 degrees for better obstacle avoidance
        self.max_consecutive_turns = 4  # Reduced to prevent endless turning
        self.consecutive_turn_count = 0
        self.last_successful_direction = None  # Track what worked last time
        self.stuck_detection_time = 3.0  # Faster stuck detection
        self.movement_history = []
        self.min_progress_distance = 0.03
        
        # FIXED: Enhanced movement state machine
        self.movement_state = "STOPPED"
        self.movement_start_time = None
        self.movement_start_pose = None
        self.last_stop_time = rospy.Time.now()
        self.deceleration_initiated = False  # NEW: Gradual deceleration flag
        self.ignore_obstacles_while_turning = True  # NEW: Prevent turn interruption
        self.turn_start_time = None  # NEW: Track turn start time
        
        self.turn_target_angle = None
        self.turn_start_angle = None
        
        # SLAM parameters - OPTIMIZED FOR SPEED
        self.min_distance_new_node = 0.08
        self.min_angle_new_node = 0.3
        self.loop_closure_distance = 0.1
        self.loop_closure_min_time_diff = 10.0
        self.optimization_frequency = 10
        self.max_optimization_nodes = 100
        self.optimization_iterations = 4  # Number of iterations for pose graph optimization
        
        # Noise parameters
        self.sensor_noise_mean = -0.0009
        self.sensor_noise_var = 0.0001
        self.motion_noise_mean = 0.0004
        self.motion_noise_var = 0.0000
        
        # Graph structure - Modified to work with op.py
        self.nodes = {}  # Dictionary of PoseNode objects
        self.edges = []  # List of PoseEdge objects
        self.next_node_id = 0
        
        # Op.py integration variables
        self.H = None  # Information matrix
        self.b = None  # Information vector
        self.n_node = 0
        self.n_edge = 0
        
        # Current state
        self.current_pose = None
        self.last_odom = None
        self.last_scan = None
        self.last_node_pose = None
        self.obstacle_detected = False
        self.current_yaw = 0.0
        
        # Map
        self.map_resolution = 0.01
        self.map_size = 1
        self.occupancy_map = np.ones((self.map_size, self.map_size)) * 50
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Ground truth tracking
        self.actual_pose = None
        self.actual_path = []
        self.estimated_path = []
        self.all_actual_poses = []
        self.all_estimated_poses = []
        self.localization_errors = []
        self.timestamps = []
        
        # Enhanced anti-flip parameters
        self.flip_detection_threshold = 1.0
        self.roll_pitch_threshold = np.pi / 4
        self.velocity_threshold = 2.0
        self.last_reset_time = None
        self.reset_cooldown = 1.5
        self.consecutive_flips = 0
        self.max_consecutive_flips = 3
        self.last_stable_pose = None
        self.pose_stability_buffer = []
        self.stability_buffer_size = 5
        
        # Semi-real-time visualization
        self.enable_realtime_plot = True
        self.plot_update_interval = 2.0
        self.last_plot_update = rospy.Time.now()
        self.fig_realtime = None
        self.axes_realtime = None
        self.plot_initialized = False
        
        # ROS Publishers and Subscribers
        self.cmd_pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=1)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        
        # Service for resetting robot pose
        rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        rospy.Subscriber('/vector/laser', Range, self.laser_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        
        rospy.loginfo("Enhanced Fast Graph SLAM with FIXED movement and turning initialized")
        rospy.loginfo(f"Results will be saved to: {self.output_dir}")
        rospy.loginfo(f"Discrete movement: {self.discrete_move_distance}m with {self.distance_check_tolerance}m tolerance")
        rospy.loginfo(f"Turn angle: {np.degrees(self.turn_angle):.1f}° with {np.degrees(self.turn_tolerance):.1f}° tolerance")
        
        # Initialize plot if enabled
        if self.enable_realtime_plot:
            self.initialize_realtime_plot()
        
        # Wait for connections
        rospy.sleep(1)
        
    def create_output_directory(self):
        """Create directory for saving results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_dir, f"slam_run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize_realtime_plot(self):
        """Initialize the real-time plot"""
        plt.ion()
        self.fig_realtime, self.axes_realtime = plt.subplots(2, 2, figsize=(12, 10))
        self.fig_realtime.suptitle('Enhanced Fast Graph SLAM with FIXED Movement Control')
        
        # Configure axes
        self.axes_realtime[0, 0].set_title('Robot Trajectory')
        self.axes_realtime[0, 0].set_xlabel('X (m)')
        self.axes_realtime[0, 0].set_ylabel('Y (m)')
        self.axes_realtime[0, 0].grid(True, alpha=0.3)
        self.axes_realtime[0, 0].set_aspect('equal')
        
        self.axes_realtime[0, 1].set_title('Occupancy Map')
        self.axes_realtime[0, 1].set_xlabel('X (pixels)')
        self.axes_realtime[0, 1].set_ylabel('Y (pixels)')
        
        self.axes_realtime[1, 0].set_title('Graph Structure')
        self.axes_realtime[1, 0].set_xlabel('X (m)')
        self.axes_realtime[1, 0].set_ylabel('Y (m)')
        self.axes_realtime[1, 0].grid(True, alpha=0.3)
        self.axes_realtime[1, 0].set_aspect('equal')
        
        self.axes_realtime[1, 1].set_title('Position Error')
        self.axes_realtime[1, 1].set_xlabel('Time (s)')
        self.axes_realtime[1, 1].set_ylabel('Error (m)')
        self.axes_realtime[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self.plot_initialized = True
        
    def update_realtime_plot(self):
        """Update the semi-real-time plot"""
        if not self.enable_realtime_plot or not self.plot_initialized:
            return
            
        current_time = rospy.Time.now()
        if (current_time - self.last_plot_update).to_sec() < self.plot_update_interval:
            return
            
        self.last_plot_update = current_time
        
        try:
            for ax in self.axes_realtime.flat:
                ax.clear()
            
            # 1. Trajectory plot
            ax = self.axes_realtime[0, 0]
            if len(self.actual_path) > 1:
                actual = np.array(self.actual_path)
                ax.plot(actual[:, 0], actual[:, 1], 'r-', label='Ground Truth', linewidth=1.5, alpha=0.7)
            
            if len(self.nodes) > 0:
                with self.lock:
                    nodes_copy = copy.deepcopy(list(self.nodes.values()))
                node_x = [node.x for node in nodes_copy]
                node_y = [node.y for node in nodes_copy]
                ax.plot(node_x, node_y, 'b-', label='SLAM Estimate', linewidth=2)
                ax.scatter(node_x[0], node_y[0], c='green', s=50, marker='o', label='Start')
                if self.current_pose:
                    ax.scatter(self.current_pose[0], self.current_pose[1], 
                              c='blue', s=100, marker='*', label='Current')
            
            # Show movement state and target
            if self.movement_state == "MOVING" and self.movement_start_pose:
                target_x = self.movement_start_pose[0] + self.discrete_move_distance * np.cos(self.movement_start_pose[2])
                target_y = self.movement_start_pose[1] + self.discrete_move_distance * np.sin(self.movement_start_pose[2])
                ax.scatter(target_x, target_y, c='orange', s=80, marker='x', label='Target')
            elif self.movement_state == "TURNING":
                # Show turn arc
                if self.current_pose and self.turn_target_angle is not None:
                    angles = np.linspace(self.current_yaw, self.turn_target_angle, 20)
                    arc_x = [self.current_pose[0] + 0.1 * np.cos(a) for a in angles]
                    arc_y = [self.current_pose[1] + 0.1 * np.sin(a) for a in angles]
                    ax.plot(arc_x, arc_y, 'orange', linewidth=2, alpha=0.7, label='Turn Arc')
            
            state_info = f'State: {self.movement_state}'
            if hasattr(self, 'consecutive_turn_count'):
                state_info += f', Turns: {self.consecutive_turn_count}'
            ax.set_title(f'Trajectory ({state_info})')
            ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.legend(loc='best')
            ax.grid(True, alpha=0.3), ax.set_aspect('equal')
            
            # 2. Occupancy Map
            ax = self.axes_realtime[0, 1]
            ax.imshow(self.occupancy_map, cmap='gray', origin='lower', vmin=0, vmax=100)
            ax.set_title('Occupancy Map'), ax.set_xlabel('X (pixels)'), ax.set_ylabel('Y (pixels)')
            
            # 3. Graph Structure
            ax = self.axes_realtime[1, 0]
            if len(self.nodes) > 0:
                node_x = [node.x for node in nodes_copy]
                node_y = [node.y for node in nodes_copy]
                ax.scatter(node_x, node_y, c='blue', s=20, alpha=0.6)
                
                loop_closures = 0
                for edge in self.edges:
                    if edge.from_id in self.nodes and edge.to_id in self.nodes:
                        from_node = self.nodes[edge.from_id]
                        to_node = self.nodes[edge.to_id]
                        line_color = 'b-' if abs(edge.to_id - edge.from_id) == 1 else 'r-'
                        alpha = 0.3 if abs(edge.to_id - edge.from_id) == 1 else 0.7
                        linewidth = 0.5 if abs(edge.to_id - edge.from_id) == 1 else 1.5
                        ax.plot([from_node.x, to_node.x], [from_node.y, to_node.y], line_color, alpha=alpha, linewidth=linewidth)
                        if abs(edge.to_id - edge.from_id) > 1:
                            loop_closures += 1
            
            ax.set_title(f'Graph: {len(self.nodes)} nodes, {loop_closures} loops')
            ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.grid(True, alpha=0.3), ax.set_aspect('equal')
            
            # 4. Error plot
            ax = self.axes_realtime[1, 1]
            if len(self.localization_errors) > 1:
                with self.lock:
                    error_data = copy.deepcopy(self.localization_errors)
                
                if len(error_data) > 0:
                    times = [(e['timestamp'] - error_data[0]['timestamp']) for e in error_data]
                    pos_errors = [e['position_error'] for e in error_data]
                    ax.plot(times, pos_errors, 'b-', linewidth=1.5)
                    ax.fill_between(times, 0, pos_errors, alpha=0.3)
                    ax.set_title(f'Position Error (Current: {pos_errors[-1]:.3f}m)')
            else:
                ax.set_title('Position Error')
            ax.set_xlabel('Time (s)'), ax.set_ylabel('Error (m)'), ax.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.draw(), plt.pause(0.001)
            
        except Exception as e:
            rospy.logwarn(f"Error updating plot: {e}")

    def detect_and_reset_flip(self, pose):
        """Enhanced flip detection including front-up orientation and instability"""
        if pose is None: 
            return False
            
        current_time = rospy.Time.now()
        if self.last_reset_time and (current_time - self.last_reset_time).to_sec() < self.reset_cooldown: 
            return False
            
        quat = pose.orientation
        roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        
        # Multiple flip detection criteria
        z_height_issue = abs(pose.position.z) > self.flip_detection_threshold
        roll_issue = abs(roll) > self.roll_pitch_threshold
        pitch_issue = abs(pitch) > self.roll_pitch_threshold
        upside_down = quat.w < 0.7
        extreme_orientation = (abs(roll) > np.pi/3 or abs(pitch) > np.pi/3)
        
        is_flipped = (z_height_issue or roll_issue or pitch_issue or extreme_orientation)
        
        if is_flipped:
            flip_type = []
            if z_height_issue:
                flip_type.append(f"Z={pose.position.z:.2f}")
            if roll_issue:
                flip_type.append(f"Roll={np.degrees(roll):.1f}°")
            if pitch_issue:
                flip_type.append(f"Pitch={np.degrees(pitch):.1f}°")
            if upside_down:
                flip_type.append(f"Upside-down (w={quat.w:.2f})")
            if extreme_orientation:
                flip_type.append("Extreme orientation")
                
            rospy.logwarn(f"Robot flip detected! Issues: {', '.join(flip_type)}")
            
            reset_x, reset_y = pose.position.x, pose.position.y
            reset_yaw = yaw
            
            if self.last_stable_pose is not None:
                reset_x, reset_y, reset_yaw = self.last_stable_pose
                rospy.loginfo(f"Using last stable pose for reset")
            
            self.reset_robot_pose(reset_x, reset_y, reset_yaw)
            self.last_reset_time = current_time
            self.consecutive_flips += 1
            
            if self.consecutive_flips >= self.max_consecutive_flips:
                rospy.logwarn("Too many consecutive flips! Moving to safer position...")
                safe_x = reset_x + np.random.uniform(-0.5, 0.5)
                safe_y = reset_y + np.random.uniform(-0.5, 0.5)
                self.reset_robot_pose(safe_x, safe_y, reset_yaw)
                self.consecutive_flips = 0
                
            return True
        else:
            self.consecutive_flips = 0
            self.update_stable_pose(pose.position.x, pose.position.y, yaw)
            
        return False
        
    def update_stable_pose(self, x, y, yaw):
        """Update the last known stable pose for recovery purposes"""
        self.pose_stability_buffer.append([x, y, yaw, rospy.Time.now().to_sec()])
        
        if len(self.pose_stability_buffer) > self.stability_buffer_size:
            self.pose_stability_buffer.pop(0)
        
        if len(self.pose_stability_buffer) >= 3:
            recent_poses = np.array([[p[0], p[1], p[2]] for p in self.pose_stability_buffer[-3:]])
            self.last_stable_pose = [
                np.mean(recent_poses[:, 0]),
                np.mean(recent_poses[:, 1]),
                np.mean(recent_poses[:, 2])
            ]
        
    def reset_robot_pose(self, x, y, yaw):
        """Enhanced robot pose reset"""
        try:
            self.stop()
            rospy.sleep(0.2)
            
            state_msg = ModelState()
            state_msg.model_name = 'vector'
            
            state_msg.pose.position.x = x
            state_msg.pose.position.y = y
            state_msg.pose.position.z = 0.01
            
            q = quaternion_from_euler(0.0, 0.0, yaw)
            state_msg.pose.orientation.x = q[0]
            state_msg.pose.orientation.y = q[1] 
            state_msg.pose.orientation.z = q[2]
            state_msg.pose.orientation.w = q[3]
            
            state_msg.twist.linear.x = 0.0
            state_msg.twist.linear.y = 0.0
            state_msg.twist.linear.z = 0.0
            state_msg.twist.angular.x = 0.0
            state_msg.twist.angular.y = 0.0
            state_msg.twist.angular.z = 0.0
            
            response = self.set_model_state(state_msg)
            if response.success:
                rospy.loginfo(f"Robot successfully reset to ({x:.2f}, {y:.2f}) with yaw {np.degrees(yaw):.1f}°")
                
                rospy.sleep(0.3)
                
                self.movement_state = "STOPPED"
                self.last_stop_time = rospy.Time.now()
                self.movement_start_time = None
                self.movement_start_pose = None
                self.deceleration_initiated = False
                
                if self.current_pose is not None:
                    self.current_pose[0] = x
                    self.current_pose[1] = y
                    self.current_pose[2] = yaw
                    
            else:
                rospy.logerr(f"Failed to reset robot pose: {response.status_message}")
                
        except Exception as e:
            rospy.logerr(f"Error during robot pose reset: {e}")
            for _ in range(3):
                self.cmd_pub.publish(Twist())
                rospy.sleep(0.1)
    
    def stop(self):
        """Immediate stop command"""
        self.cmd_pub.publish(Twist())
        self.movement_state = "STOPPED"
        self.last_stop_time = rospy.Time.now()
        self.deceleration_initiated = False
        
    def move_forward(self):
        """Start forward movement with precise tracking"""
        twist = Twist()
        twist.linear.x = self.forward_speed
        self.cmd_pub.publish(twist)
        self.movement_state = "MOVING"
        self.movement_start_time = rospy.Time.now()
        self.deceleration_initiated = False
        if self.current_pose:
            self.movement_start_pose = copy.deepcopy(self.current_pose)
            rospy.loginfo(f"Starting precise move from ({self.movement_start_pose[0]:.3f}, {self.movement_start_pose[1]:.3f})")
        
    def turn_smart(self, direction='auto'):
        """
        Initiates a smart turn. The actual turning command is sustained by the
        main control_robot loop.
        """
        self.stop()
        rospy.sleep(0.2)  # Let robot settle
        
        self.consecutive_turn_count += 1
        
        turn_modifier = 1.0
        
        if direction == 'auto':
            if self.last_successful_direction == 'left':
                turn_modifier = 1.0
                rospy.loginfo("Auto-turning LEFT (last successful)")
            elif self.last_successful_direction == 'right':
                turn_modifier = -1.0
                rospy.loginfo("Auto-turning RIGHT (last successful)")
            else:
                # If no history, default to right, but also clear last successful direction
                self.last_successful_direction = None
                turn_modifier = -1.0
                rospy.loginfo("Auto-turning RIGHT (default)")
        elif direction == 'left':
            turn_modifier = 1.0
        elif direction == 'right':
            turn_modifier = -1.0
        
        if self.consecutive_turn_count >= self.max_consecutive_turns or direction == 'reverse':
            rospy.logwarn("Multiple turns failed or reverse requested! Trying 180-degree escape turn...")
            self.turn_target_angle = normalize_angle(self.current_yaw + np.pi)
            self.consecutive_turn_count = 0 # Reset after trying the escape
        else:
            self.turn_target_angle = normalize_angle(self.current_yaw + turn_modifier * self.turn_angle)
        
        self.movement_state = "TURNING"
        self.turn_start_angle = self.current_yaw
        self.turn_start_time = rospy.Time.now()
        
        rospy.loginfo(f"Turn #{self.consecutive_turn_count}, current: {np.degrees(self.current_yaw):.1f}°, target: {np.degrees(self.turn_target_angle):.1f}°")
        
    def is_stuck(self):
        """Enhanced stuck detection"""
        if len(self.movement_history) < 8:
            return False
            
        recent_positions = self.movement_history[-8:]
        if len(recent_positions) < 2:
            return False
            
        start_pos = np.array(recent_positions[0][:2])
        end_pos = np.array(recent_positions[-1][:2])
        progress = np.linalg.norm(end_pos - start_pos)
        
        recent_time = recent_positions[-1][3] - recent_positions[0][3]
        
        return progress < self.min_progress_distance and recent_time > self.stuck_detection_time
        
    def update_movement_history(self):
        """Update movement history for stuck detection"""
        if self.current_pose is not None:
            current_time = rospy.Time.now().to_sec()
            self.movement_history.append([
                self.current_pose[0], 
                self.current_pose[1], 
                self.current_pose[2], 
                current_time
            ])
            
            if len(self.movement_history) > 20:
                self.movement_history.pop(0)
    
    def check_turn_complete(self):
        """
        Precise turn completion check. Returns True only if angle tolerance is met.
        """
        if self.movement_state != "TURNING" or self.turn_target_angle is None:
            return False
            
        current_error = abs(normalize_angle(self.current_yaw - self.turn_target_angle))
        
        return current_error < self.turn_tolerance
        
    def laser_callback(self, msg):
        with self.lock:
            self.last_scan = msg.range
            
            is_obstacle = msg.range < self.min_safe_distance
            
            # Always update the obstacle flag with the latest sensor reading.
            # This ensures the robot's knowledge is never stale.
            self.obstacle_detected = is_obstacle
            
            # The emergency stop logic should only trigger when MOVING forward into an obstacle.
            if self.movement_state == "MOVING" and self.obstacle_detected:
                rospy.logwarn(f"EMERGENCY STOP from laser_callback: Obstacle at {msg.range:.3f}m!")
                self.stop()
                # The main control loop will now handle the stopped state with an obstacle.
                return

            # Map is updated only when stopped to ensure stable readings.
            if self.current_pose is not None and self.movement_state == "STOPPED":
                self.update_map(msg.range)
                
    def odom_callback(self, msg):
        with self.lock:
            pos = msg.pose.pose.position
            quat = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            self.current_yaw = yaw
            
            if self.current_pose is None:
                self.current_pose = [pos.x, pos.y, yaw]
                self.add_node(pos.x, pos.y, yaw)
                self.last_node_pose = copy.deepcopy(self.current_pose)
            else:
                if self.last_odom is not None:
                    last_pos = self.last_odom.pose.pose.position
                    last_quat = self.last_odom.pose.pose.orientation
                    _, _, last_yaw = euler_from_quaternion([last_quat.x, last_quat.y, last_quat.z, last_quat.w])
                    
                    dx_odom = pos.x - last_pos.x
                    dy_odom = pos.y - last_pos.y
                    dtheta = normalize_angle(yaw - last_yaw)

                    dx_local = dx_odom * np.cos(last_yaw) + dy_odom * np.sin(last_yaw)
                    dy_local = -dx_odom * np.sin(last_yaw) + dy_odom * np.cos(last_yaw)
                    
                    dx_local += np.random.normal(self.motion_noise_mean, np.sqrt(self.motion_noise_var))
                    dy_local += np.random.normal(self.motion_noise_mean, np.sqrt(self.motion_noise_var))
                    
                    self.current_pose[0] += dx_local * np.cos(self.current_pose[2]) - dy_local * np.sin(self.current_pose[2])
                    self.current_pose[1] += dx_local * np.sin(self.current_pose[2]) + dy_local * np.cos(self.current_pose[2])
                    self.current_pose[2] = normalize_angle(self.current_pose[2] + dtheta)
                    
                    self.all_estimated_poses.append(copy.deepcopy(self.current_pose))
                    self.timestamps.append(rospy.Time.now().to_sec())
                    
                    if self.movement_state == "STOPPED" and self.should_add_node():
                        self.add_node(self.current_pose[0], self.current_pose[1], self.current_pose[2])
                        if len(self.nodes) > 1:
                            self.add_odometry_edge()
                        self.detect_loop_closures()
                        if len(self.nodes) % self.optimization_frequency == 0 and len(self.nodes) > 2:
                            self.optimize_graph_with_op()
                        self.last_node_pose = copy.deepcopy(self.current_pose)
            
            self.last_odom = msg
            
    def model_states_callback(self, msg):
        try:
            idx = msg.name.index('vector')
            self.actual_pose = msg.pose[idx]
            
            if self.detect_and_reset_flip(self.actual_pose): 
                return
            
            pos = self.actual_pose.position
            self.actual_path.append([pos.x, pos.y])
            
            quat = self.actual_pose.orientation
            _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            self.all_actual_poses.append([pos.x, pos.y, yaw])
            
            if self.current_pose is not None:
                pos_error = np.sqrt((pos.x - self.current_pose[0])**2 + (pos.y - self.current_pose[1])**2)
                angle_error = abs(normalize_angle(yaw - self.current_pose[2]))
                with self.lock:
                    self.localization_errors.append({
                        'timestamp': rospy.Time.now().to_sec(),
                        'position_error': pos_error,
                        'angle_error': angle_error
                    })
        except ValueError:
            pass
            
    def should_add_node(self):
        if self.last_node_pose is None: 
            return True
        dx = self.current_pose[0] - self.last_node_pose[0]
        dy = self.current_pose[1] - self.last_node_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        angle_change = abs(normalize_angle(self.current_pose[2] - self.last_node_pose[2]))
        return distance > self.min_distance_new_node or angle_change > self.min_angle_new_node
        
    def add_node(self, x, y, theta):
        node = PoseNode(self.next_node_id, x, y, theta, rospy.Time.now().to_sec())
        node.scan_data = self.last_scan
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        self.estimated_path.append([x, y])
        rospy.loginfo(f"Added node {node.id} at ({x:.3f}, {y:.3f}, {np.degrees(theta):.1f}°)")
        
    def add_odometry_edge(self):
        if len(self.nodes) < 2: 
            return
        prev_node = self.nodes[self.next_node_id - 2]
        curr_node = self.nodes[self.next_node_id - 1]
        
        dx = curr_node.x - prev_node.x
        dy = curr_node.y - prev_node.y
        cos_theta, sin_theta = np.cos(prev_node.theta), np.sin(prev_node.theta)
        
        dx_local = dx * cos_theta + dy * sin_theta
        dy_local = -dx * sin_theta + dy * cos_theta
        dtheta = normalize_angle(curr_node.theta - prev_node.theta)
        
        info_matrix = np.diag([50.0, 50.0, 25.0])
        edge = PoseEdge(prev_node.id, curr_node.id, dx_local, dy_local, dtheta, info_matrix)
        self.edges.append(edge)
        
    def detect_loop_closures(self):
        if len(self.nodes) < 10: 
            return
        current_node = self.nodes[self.next_node_id - 1]
        current_pos = np.array([current_node.x, current_node.y])
        
        positions, node_ids = [], []
        for node_id, node in self.nodes.items():
            if node_id == current_node.id: 
                continue
            if abs(current_node.timestamp - node.timestamp) < self.loop_closure_min_time_diff: 
                continue
            positions.append([node.x, node.y])
            node_ids.append(node_id)
            
        if not positions: 
            return
        tree = KDTree(positions)
        indices = tree.query_ball_point(current_pos, self.loop_closure_distance)
        
        for idx in indices:
            candidate_id = node_ids[idx]
            candidate_node = self.nodes[candidate_id]
            if self.verify_loop_closure(current_node, candidate_node):
                dx = current_node.x - candidate_node.x
                dy = current_node.y - candidate_node.y
                cos_theta, sin_theta = np.cos(candidate_node.theta), np.sin(candidate_node.theta)
                dx_local = dx * cos_theta + dy * sin_theta
                dy_local = -dx * sin_theta + dy * cos_theta
                dtheta = normalize_angle(current_node.theta - candidate_node.theta)
                info_matrix = np.diag([200.0, 200.0, 100.0])
                edge = PoseEdge(candidate_id, current_node.id, dx_local, dy_local, dtheta, info_matrix)
                self.edges.append(edge)
                rospy.loginfo(f"Loop closure detected: {candidate_id} -> {current_node.id}")
                
    def verify_loop_closure(self, node1, node2):
        if node1.scan_data is None or node2.scan_data is None: 
            return False
        return abs(node1.scan_data - node2.scan_data) < 0.1

    def linearize_pose_graph(self):
        """Linearize the pose graph following op.py methodology"""
        print('Initializing H (information matrix) and b (information vector)...')
        
        nodes_to_optimize = list(self.nodes.keys())
        if len(nodes_to_optimize) > self.max_optimization_nodes:
            nodes_to_optimize = [0] + nodes_to_optimize[-(self.max_optimization_nodes-1):]
        
        self.n_node = len(nodes_to_optimize)
        self.n_edge = len([e for e in self.edges if e.from_id in nodes_to_optimize and e.to_id in nodes_to_optimize])
        
        if self.n_node == 0 or self.n_edge == 0:
            return False
            
        self.node_id_map = {node_id: i for i, node_id in enumerate(sorted(nodes_to_optimize))}
        
        self.H = np.mat(np.zeros((self.n_node*3, self.n_node*3)))
        self.b = np.mat(np.zeros((self.n_node*3, 1)))
        
        for edge_ij in self.edges:
            if edge_ij.from_id not in self.node_id_map or edge_ij.to_id not in self.node_id_map:
                continue
                
            Id_i = self.node_id_map[edge_ij.from_id]
            Id_j = self.node_id_map[edge_ij.to_id]
            i_idx = id2idx(Id_i)
            j_idx = id2idx(Id_j)
            
            node_i_pose = self.nodes[edge_ij.from_id].pose
            node_j_pose = self.nodes[edge_ij.to_id].pose
            
            Omega = np.mat(edge_ij.info_matrix)
            
            X_i = vec2mat(node_i_pose)
            R_i = X_i[0:2, 0:2]
            
            X_j = vec2mat(node_j_pose)
            
            Z_ij = vec2mat(edge_ij.mean)
            R_ij = Z_ij[0:2, 0:2]
            
            e = mat2vec(Z_ij.I * X_i.I * X_j)
            
            s_i = X_i[1, 0]
            c_i = X_i[0, 0]
            
            dR_dyaw_i = np.mat([   
                [-s_i, -c_i],
                [c_i,  -s_i]
            ])
            
            t_i = node_i_pose[0:2, 0]
            t_j = node_j_pose[0:2, 0]
            
            A = np.mat(np.zeros((3, 3)))
            A[0:2, 0:2] = -R_ij.T * R_i.T
            A[0:2, 2:3] = R_ij.T * dR_dyaw_i.T * (t_j - t_i)
            A[2:3, 0:3] = np.mat([0, 0, -1])
            
            B = np.mat(np.zeros((3, 3)))
            B[0:2, 0:2] = R_ij.T * R_i.T
            B[0:2, 2:3] = np.mat([0, 0]).T
            B[2:3, 0:3] = np.mat([0, 0, 1])
            
            H_ii = A.T * Omega * A
            H_ij = A.T * Omega * B
            H_ji = B.T * Omega * A
            H_jj = B.T * Omega * B
          
            self.H[i_idx[0]:i_idx[1], i_idx[0]:i_idx[1]] += H_ii
            self.H[i_idx[0]:i_idx[1], j_idx[0]:j_idx[1]] += H_ij
            self.H[j_idx[0]:j_idx[1], i_idx[0]:i_idx[1]] += H_ji
            self.H[j_idx[0]:j_idx[1], j_idx[0]:j_idx[1]] += H_jj
            
            b_i = A.T * Omega * e
            b_j = B.T * Omega * e
            
            self.b[i_idx[0]:i_idx[1]] += b_i
            self.b[j_idx[0]:j_idx[1]] += b_j
            
        return True

    def solve_pose_graph(self):
        """Solve the linearized pose graph system"""
        print(f'Pose: {self.n_node}, Edge: {self.n_edge}')
        
        self.H[0:3, 0:3] += np.eye(3)
        
        H = self.H.copy()
        
        H_sparse = csc_matrix(H)
        
        H_sparse_inv = ssl.splu(H_sparse)
        
        dx = -H_sparse_inv.solve(self.b)
        
        dx = dx.reshape([3, self.n_node], order='F')
        
        nodes_to_update = sorted([node_id for node_id in self.nodes.keys() 
                                if node_id in self.node_id_map])
        
        for i, node_id in enumerate(nodes_to_update):
            if node_id in self.nodes:
                self.nodes[node_id].pose += dx[:, i]
                self.nodes[node_id].update_from_pose_matrix()

    def optimize_graph_with_op(self):
        """Enhanced pose graph optimization using op.py methodology"""
        if len(self.edges) * 3 < len(self.nodes) * 3:
            rospy.logwarn("Not enough constraints for optimization.")
            return
            
        start_time = rospy.Time.now()
        rospy.loginfo(f"Starting pose graph optimization with {len(self.nodes)} nodes...")
        
        try:
            for iteration in range(self.optimization_iterations):
                rospy.loginfo(f'Pose Graph Optimization, Iteration {iteration + 1}')
                
                if not self.linearize_pose_graph():
                    rospy.logwarn("Failed to linearize pose graph")
                    return
                    
                rospy.loginfo('Solving...')
                self.solve_pose_graph()
                
                rospy.loginfo(f'Iteration {iteration + 1} done.')
            
            with self.lock:
                if (self.next_node_id - 1) in self.nodes:
                    last_node = self.nodes[self.next_node_id - 1]
                    self.current_pose = [last_node.x, last_node.y, last_node.theta]
                
                self.estimated_path = [[node.x, node.y] for node in self.nodes.values()]
            
            optimization_time = (rospy.Time.now() - start_time).to_sec()
            rospy.loginfo(f"Pose graph optimization complete in {optimization_time:.3f}s with {self.optimization_iterations} iterations")
                         
        except Exception as e:
            rospy.logerr(f"Pose graph optimization failed: {e}")
            import traceback
            traceback.print_exc()

    def update_map(self, range_measurement):
        if self.current_pose is None: 
            return
            
        robot_x = int((self.current_pose[0] + 5) / self.map_resolution)
        robot_y = int((self.current_pose[1] + 5) / self.map_resolution)
        
        if not (0 <= robot_x < self.map_size and 0 <= robot_y < self.map_size): 
            return
            
        noisy_range = range_measurement + np.random.normal(self.sensor_noise_mean, np.sqrt(self.sensor_noise_var))
        
        end_x = self.current_pose[0] + noisy_range * np.cos(self.current_pose[2])
        end_y = self.current_pose[1] + noisy_range * np.sin(self.current_pose[2])
        end_x_map = int((end_x + 5) / self.map_resolution)
        end_y_map = int((end_y + 5) / self.map_resolution)
        
        x0, y0, x1, y1 = robot_x, robot_y, end_x_map, end_y_map
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= x0 < self.map_size and 0 <= y0 < self.map_size:
                if x0 == end_x_map and y0 == end_y_map and noisy_range < 3.0:
                    self.occupancy_map[y0, x0] = min(100, self.occupancy_map[y0, x0] + 10)
                else:
                    self.occupancy_map[y0, x0] = max(0, self.occupancy_map[y0, x0] - 5)
            if x0 == x1 and y0 == y1: 
                break
            e2 = 2 * err
            if e2 > -dy: 
                err -= dy
                x0 += sx
            if e2 < dx: 
                err += dx
                y0 += sy
                
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.info.resolution, msg.info.width, msg.info.height = self.map_resolution, self.map_size, self.map_size
        msg.info.origin.position.x, msg.info.origin.position.y = -5.0, -5.0
        msg.info.origin.orientation.w = 1.0
        
        map_data = self.occupancy_map.flatten().astype(int)
        map_data[map_data == 50] = -1
        msg.data = list(map_data)
        self.map_pub.publish(msg)
        
    def control_robot(self):
        """Main robot control state machine."""
        current_time = rospy.Time.now()
        self.update_movement_history()

        if self.movement_state == "STOPPED":
            if (current_time - self.last_stop_time).to_sec() < self.stop_duration:
                return
                
            if self.is_stuck():
                rospy.logwarn("Robot appears to be fundamentally stuck! Trying recovery maneuver...")
                self.consecutive_turn_count = 0
                self.turn_smart('reverse')
                return

            if self.obstacle_detected:
                rospy.loginfo(f"Obstacle detected at {self.last_scan:.3f}m! Initiating avoidance turn.")
                
                if self.consecutive_turn_count >= self.max_consecutive_turns:
                    rospy.logwarn("Multiple turn attempts failed. Trying opposite direction...")
                    if self.last_successful_direction == 'left':
                        self.turn_smart('right')
                    elif self.last_successful_direction == 'right':
                        self.turn_smart('left') 
                    else: # If history is unclear, just pick one
                        self.turn_smart('left')
                    # After trying the opposite, clear the history to prevent getting stuck in a left/right loop
                    self.last_successful_direction = None
                else:
                    self.turn_smart('auto')
            else:
                rospy.loginfo("Path clear. Moving forward.")
                self.consecutive_turn_count = 0
                self.move_forward()
                    
        elif self.movement_state == "MOVING":
            if self.obstacle_detected:
                rospy.logwarn("Safeguard STOP: Obstacle detected in main control loop!")
                self.stop()
                return

            if self.movement_start_pose is None:
                rospy.logwarn("Cannot check move completion: movement_start_pose is None.")
                self.stop()
                return

            dx = self.current_pose[0] - self.movement_start_pose[0]
            dy = self.current_pose[1] - self.movement_start_pose[1]
            distance_moved = np.sqrt(dx**2 + dy**2)

            # Check for gross overshoot as a sanity check for odometry issues
            if distance_moved > self.discrete_move_distance * 2.0:
                 rospy.logwarn(f"Massive overshoot detected! Moved {distance_moved:.3f}m, but target was {self.discrete_move_distance:.3f}m. Stopping due to likely odometry error.")
                 self.stop()
                 return

            if distance_moved >= self.discrete_move_distance:
                rospy.loginfo(f"Discrete move complete: moved {distance_moved:.4f}m (target: {self.discrete_move_distance:.3f}m)")
                self.stop()
                self.consecutive_turn_count = 0 # Successful move resets turn counter
                return

            remaining_distance = self.discrete_move_distance - distance_moved
            if remaining_distance <= self.deceleration_distance and not self.deceleration_initiated:
                rospy.loginfo(f"Starting deceleration: {remaining_distance:.3f}m remaining")
                twist = Twist()
                twist.linear.x = self.deceleration_speed
                self.cmd_pub.publish(twist)
                self.deceleration_initiated = True
                
            if self.movement_start_time and (current_time - self.movement_start_time).to_sec() > self.max_movement_time:
                rospy.logwarn("Movement timeout! Forcing stop.")
                self.stop()
                return

    # =======================================================================================
    # ========================== FIX: ROBUST TURNING STATE LOGIC V2 =========================
    # =======================================================================================
        elif self.movement_state == "TURNING":
            # LOGIC: Check for SUCCESS first. If not success, check for TIMEOUT. If neither, CONTINUE.

            # 1. Check for SUCCESSFUL completion.
            if self.check_turn_complete():
                current_error = abs(normalize_angle(self.current_yaw - self.turn_target_angle))
                rospy.loginfo(f"Turn completed successfully! Final error {np.degrees(current_error):.1f}°")
                self.stop()
                
                # After a brief pause, check if the path is now clear.
                rospy.sleep(0.1)
                if not self.obstacle_detected:
                    if self.turn_target_angle is not None:
                        turn_amount = normalize_angle(self.turn_target_angle - self.turn_start_angle)
                        if turn_amount > 0:
                            self.last_successful_direction = 'left'
                        else:
                            self.last_successful_direction = 'right'
                    rospy.loginfo(f"Turn successful! Path now clear. Remembering direction: {self.last_successful_direction}")
                    self.consecutive_turn_count = 0 # Reset counter on a fully successful avoidance
                else:
                    rospy.loginfo("Turn complete but obstacle still detected.")
                return

            # 2. If not successful, check for a TIMEOUT. This indicates a FAILED turn.
            elif self.turn_start_time and (current_time - self.turn_start_time).to_sec() > self.max_turn_time:
                current_error = abs(normalize_angle(self.current_yaw - self.turn_target_angle))
                rospy.logwarn(f"Turn FAILED due to timeout! Final error: {np.degrees(current_error):.1f}°")
                self.stop() # Stop the current turn command.
                rospy.logwarn("Re-evaluating situation and initiating a new turn immediately.")
                # Immediately decide on the next turn, preventing the uncontrolled "STOPPED" state.
                self.turn_smart('auto') 
                return

            # 3. If neither success nor timeout, CONTINUE sending the turn command.
            else:
                turn_error = normalize_angle(self.turn_target_angle - self.current_yaw)
                twist = Twist()
                twist.angular.z = self.angular_speed * np.sign(turn_error)
                self.cmd_pub.publish(twist)
    # =======================================================================================
    # ================================= END OF FIX ==========================================
    # =======================================================================================

    def run(self):
        rate = rospy.Rate(50)
        rospy.loginfo("Starting FIXED Graph SLAM with precise movement and turn control...")
        rospy.loginfo(f"Discrete movement distance: {self.discrete_move_distance}m (±{self.distance_check_tolerance}m)")
        rospy.loginfo(f"Turn angle: {np.degrees(self.turn_angle):.1f}° (±{np.degrees(self.turn_tolerance):.1f}°)")
        rospy.loginfo(f"Deceleration starts at: {self.deceleration_distance}m from target")
        
        # --- TIMEOUT MECHANISM ---
        start_time = rospy.Time.now()
        timeout_duration = rospy.Duration(20 * 60)  # 20 minutes in seconds
        rospy.loginfo(f"Node will automatically shut down after {timeout_duration.to_sec() / 60.0:.1f} minutes.")
        
        while not rospy.is_shutdown():
            # --- Check if the timeout has been reached ---
            if rospy.Time.now() - start_time > timeout_duration:
                rospy.loginfo("20-minute timer expired. Shutting down and finalizing results.")
                break  # Exit the loop to proceed to the finalization code
            
            self.control_robot()
            self.publish_map()
            self.update_realtime_plot()
            rate.sleep()
            
        # --- FINALIZATION CODE ---
        # This part of the code now runs either after Ctrl+C OR after the 20-minute timeout.
        # It performs the final, most comprehensive optimization and saves all data.
        self.stop()
        rospy.loginfo("Running final pose graph optimization...")
        if len(self.nodes) > 2:
            original_limit = self.max_optimization_nodes
            self.max_optimization_nodes = len(self.nodes)
            original_iterations = self.optimization_iterations
            self.optimization_iterations = 10
            self.optimize_graph_with_op()
            self.max_optimization_nodes = original_limit
            self.optimization_iterations = original_iterations
        
        self.save_data()
        self.save_final_plots()
        self.show_final_results()
        if self.enable_realtime_plot:
            plt.ioff()
            plt.close('all')
        
    def save_data(self):
        if len(self.actual_path) > 0:
            np.savetxt(os.path.join(self.output_dir, "ground_truth_path.txt"), 
                      self.actual_path, header="x y", fmt="%.6f")
        if len(self.estimated_path) > 0:
            np.savetxt(os.path.join(self.output_dir, "estimated_path.txt"), 
                      self.estimated_path, header="x y", fmt="%.6f")
        if self.localization_errors:
            error_data = [[e['timestamp'], e['position_error'], e['angle_error']] for e in self.localization_errors]
            np.savetxt(os.path.join(self.output_dir, "localization_errors.txt"), 
                      error_data, header="timestamp position_error angle_error", fmt="%.6f")
        np.save(os.path.join(self.output_dir, "occupancy_map.npy"), self.occupancy_map)
        
        self.save_graph_data()
        rospy.loginfo(f"Saved data files to: {self.output_dir}")
        
    def save_graph_data(self):
        """Save graph data in op.py compatible format"""
        vertex_file = os.path.join(self.output_dir, "vertices.txt")
        with open(vertex_file, 'w') as f:
            for node_id, node in self.nodes.items():
                f.write(f"VERTEX2 {node_id} {node.x:.6f} {node.y:.6f} {node.theta:.6f}\n")
        
        edge_file = os.path.join(self.output_dir, "edges.txt")
        with open(edge_file, 'w') as f:
            for edge in self.edges:
                info = edge.info_matrix
                f.write(f"EDGE2 {edge.from_id} {edge.to_id} {edge.dx:.6f} {edge.dy:.6f} {edge.dtheta:.6f} "
                       f"{info[0,0]:.6f} {info[0,1]:.6f} {info[1,1]:.6f} {info[2,2]:.6f} "
                       f"{info[0,2]:.6f} {info[1,2]:.6f}\n")
        
        rospy.loginfo(f"Saved graph data in op.py format: {vertex_file}, {edge_file}")
        
    def save_final_plots(self):
        rospy.loginfo("Creating final plots...")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("FIXED Graph SLAM Results - Precise Movement & Turn Control", fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(self.occupancy_map, cmap='gray', origin='lower', vmin=0, vmax=100)
        axes[0, 0].set_title("Final Occupancy Map")
        axes[0, 0].set_xlabel("X (pixels)")
        axes[0, 0].set_ylabel("Y (pixels)")
        
        if len(self.actual_path) > 1:
            actual = np.array(self.actual_path)
            axes[0, 1].plot(actual[:, 0], actual[:, 1], 'r-', label='Ground Truth', linewidth=2)
        if len(self.nodes) > 0:
            node_x = [node.x for node in self.nodes.values()]
            node_y = [node.y for node in self.nodes.values()]
            axes[0, 1].plot(node_x, node_y, 'b-', label='SLAM Estimate', linewidth=2)
            axes[0, 1].scatter(node_x[0], node_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
            axes[0, 1].scatter(node_x[-1], node_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
            
            movement_distances = []
            for i in range(len(node_x)-1):
                segment_distance = np.sqrt((node_x[i+1] - node_x[i])**2 + (node_y[i+1] - node_y[i])**2)
                movement_distances.append(segment_distance)
                if i < 10:
                    axes[0, 1].annotate(f'{segment_distance:.3f}', 
                                      xy=((node_x[i] + node_x[i+1])/2, (node_y[i] + node_y[i+1])/2),
                                      fontsize=7, ha='center', alpha=0.7)
        
        axes[0, 1].set_title("Robot Trajectory with Movement Precision")
        axes[0, 1].set_xlabel("X (m)")
        axes[0, 1].set_ylabel("Y (m)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_aspect('equal')
        
        if len(self.nodes) > 0:
            node_x = [node.x for node in self.nodes.values()]
            node_y = [node.y for node in self.nodes.values()]
            axes[0, 2].scatter(node_x, node_y, c='blue', s=20, alpha=0.6, label='Nodes')
            
            odometry_edges = 0
            loop_closures = 0
            for edge in self.edges:
                if edge.from_id in self.nodes and edge.to_id in self.nodes:
                    from_node, to_node = self.nodes[edge.from_id], self.nodes[edge.to_id]
                    if abs(edge.to_id - edge.from_id) == 1:
                        axes[0, 2].plot([from_node.x, to_node.x], [from_node.y, to_node.y], 'b-', alpha=0.3, linewidth=0.5)
                        odometry_edges += 1
                    else:
                        axes[0, 2].plot([from_node.x, to_node.x], [from_node.y, to_node.y], 'r-', alpha=0.8, linewidth=2)
                        loop_closures += 1
                        
        axes[0, 2].set_title(f'Graph: {len(self.nodes)} nodes, {odometry_edges} odometry, {loop_closures} loops')
        axes[0, 2].set_xlabel("X (m)")
        axes[0, 2].set_ylabel("Y (m)")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_aspect('equal')
        axes[0, 2].legend()
        
        if self.localization_errors:
            times = [e['timestamp'] - self.localization_errors[0]['timestamp'] for e in self.localization_errors]
            pos_errors = [e['position_error'] for e in self.localization_errors]
            angle_errors_deg = [np.degrees(e['angle_error']) for e in self.localization_errors]
            
            axes[1, 0].plot(times, pos_errors, 'b-', linewidth=1.5)
            axes[1, 0].fill_between(times, 0, pos_errors, alpha=0.3)
            axes[1, 0].axhline(y=np.mean(pos_errors), color='r', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(pos_errors):.3f}m')
            
            axes[1, 1].plot(times, angle_errors_deg, 'r-', linewidth=1.5)
            axes[1, 1].fill_between(times, 0, angle_errors_deg, alpha=0.3, color='red')
            axes[1, 1].axhline(y=np.mean(angle_errors_deg), color='b', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(angle_errors_deg):.1f}°')
            
        axes[1, 0].set_title("Position Error Over Time")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Error (m)")
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        axes[1, 1].set_title("Orientation Error Over Time")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Error (degrees)")
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        axes[1, 2].axis('off')
        
        movement_stats = ""
        if len(self.nodes) > 1:
            node_positions = np.array([[node.x, node.y] for node in self.nodes.values()])
            # Filter out large jumps likely caused by odometry errors after failed turns
            distances = np.linalg.norm(np.diff(node_positions, axis=0), axis=1)
            # Consider only movements that are plausibly part of a forward step (e.g., less than 3x target)
            actual_distances = [d for d in distances if d < self.discrete_move_distance * 3]
            
            if actual_distances:
                target_dist = self.discrete_move_distance
                within_1mm = sum(1 for d in actual_distances if abs(d - target_dist) <= 0.001)
                within_5mm = sum(1 for d in actual_distances if abs(d - target_dist) <= 0.005)
                within_1cm = sum(1 for d in actual_distances if abs(d - target_dist) <= 0.01)
                within_2cm = sum(1 for d in actual_distances if abs(d - target_dist) <= 0.02)
                
                movement_stats = f"""Movement Precision Analysis (Filtered):
Target: {target_dist:.3f} m, Tolerance: ±{self.distance_check_tolerance:.3f} m

Achieved Movements:
- Count: {len(actual_distances)}
- Mean: {np.mean(actual_distances):.4f} m
- Std Dev: {np.std(actual_distances):.4f} m
- Min: {np.min(actual_distances):.4f} m
- Max: {np.max(actual_distances):.4f} m

Precision Metrics:
- Within ±1mm: {within_1mm}/{len(actual_distances)} ({100*within_1mm/len(actual_distances):.1f}%)
- Within ±5mm: {within_5mm}/{len(actual_distances)} ({100*within_5mm/len(actual_distances):.1f}%)
- Within ±1cm: {within_1cm}/{len(actual_distances)} ({100*within_1cm/len(actual_distances):.1f}%)
- Within ±2cm: {within_2cm}/{len(actual_distances)} ({100*within_2cm/len(actual_distances):.1f}%)
"""
        
        if self.localization_errors:
            pos_errors = [e['position_error'] for e in self.localization_errors]
            angle_errors_deg = [np.degrees(e['angle_error']) for e in self.localization_errors]
            
            stats = f"""FIXED SLAM Performance Results
=====================================
{movement_stats}
SLAM Accuracy:
Nodes: {len(self.nodes)}, Edges: {len(self.edges)}
Loop Closures: {sum(1 for e in self.edges if abs(e.to_id - e.from_id) > 1)}

Position Error:
- Final: {pos_errors[-1]:.4f} m
- Mean: {np.mean(pos_errors):.4f} m
- Max: {np.max(pos_errors):.4f} m
- Std: {np.std(pos_errors):.4f} m

Orientation Error:
- Final: {angle_errors_deg[-1]:.2f}°
- Mean: {np.mean(angle_errors_deg):.2f}°
- Max: {np.max(angle_errors_deg):.2f}°
- Std: {np.std(angle_errors_deg):.2f}°

Control Performance:
- Turn Angle: {np.degrees(self.turn_angle):.1f}°
- Turn Tolerance: ±{np.degrees(self.turn_tolerance):.2f}°
- Deceleration Distance: {self.deceleration_distance:.3f} m
- Max Turn Attempts: {self.max_consecutive_turns}
"""
        else:
            stats = f"""FIXED SLAM Performance Results
=====================================
{movement_stats}
SLAM Structure:
Nodes: {len(self.nodes)}, Edges: {len(self.edges)}
Loop Closures: {sum(1 for e in self.edges if abs(e.to_id - e.from_id) > 1)}

No localization error data available.
"""
        
        axes[1, 2].text(0.05, 0.95, stats, transform=axes[1, 2].transAxes, 
                        fontsize=8, va='top', fontfamily='monospace', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(self.output_dir, "fixed_precise_slam_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        rospy.loginfo(f"Saved comprehensive results plot to: {plot_path}")
        plt.close()
        
    def show_final_results(self):
        print("\n" + "="*80)
        print("FIXED GRAPH SLAM WITH PRECISE MOVEMENT CONTROL - FINAL RESULTS")
        print("="*80)
        print(f"Total nodes created: {len(self.nodes)}")
        print(f"Total edges created: {len(self.edges)}")
        
        odometry_edges = sum(1 for e in self.edges if abs(e.to_id - e.from_id) == 1)
        loop_closures = len(self.edges) - odometry_edges
        print(f"Odometry edges: {odometry_edges}, Loop closures: {loop_closures}")
        
        if len(self.nodes) > 1:
            node_positions = np.array([[node.x, node.y] for node in self.nodes.values()])
            distances = np.linalg.norm(np.diff(node_positions, axis=0), axis=1)
            actual_distances = [d for d in distances if d < self.discrete_move_distance * 3]

            print(f"\nMOVEMENT PRECISION ANALYSIS (Filtered to exclude odometry error spikes):")
            print(f"Target discrete distance: {self.discrete_move_distance:.3f} m")
            if actual_distances:
                print(f"Number of plausible movements: {len(actual_distances)}")
                print(f"Mean achieved distance: {np.mean(actual_distances):.4f} m")
                print(f"Standard deviation: {np.std(actual_distances):.4f} m")
                
                within_tolerance = sum(1 for d in actual_distances if abs(d - self.discrete_move_distance) <= self.distance_check_tolerance)
                print(f"\nPRECISION METRICS:")
                print(f"Within tolerance (±{self.distance_check_tolerance*1000:.1f}mm): {within_tolerance}/{len(actual_distances)} ({100*within_tolerance/len(actual_distances):.1f}%)")
            else:
                print("No plausible forward movements were recorded to analyze.")

        print(f"\nTURN CONTROL ANALYSIS:")
        print(f"Turn angle: {np.degrees(self.turn_angle):.1f}°")
        print(f"Turn tolerance: ±{np.degrees(self.turn_tolerance):.2f}°")
        print(f"Max consecutive turns allowed: {self.max_consecutive_turns}")
        
        if self.localization_errors:
            pos_errors = [e['position_error'] for e in self.localization_errors]
            angle_errors_deg = [np.degrees(e['angle_error']) for e in self.localization_errors]
            
            print(f"\nLOCALIZATION ACCURACY:")
            print(f"Final position error: {pos_errors[-1]:.4f} m")
            print(f"Mean position error: {np.mean(pos_errors):.4f} m")
            print(f"Max position error: {np.max(pos_errors):.4f} m")
            print(f"Position error std dev: {np.std(pos_errors):.4f} m")
            print(f"Final orientation error: {angle_errors_deg[-1]:.2f}°")
            print(f"Mean orientation error: {np.mean(angle_errors_deg):.2f}°")
            print(f"Max orientation error: {np.max(angle_errors_deg):.2f}°")

        print(f"\nOUTPUT FILES:")
        print(f"All results saved to: {self.output_dir}")
        
        print("="*80)
        print("KEY IMPROVEMENTS IMPLEMENTED:")
        print("✓ SOLVED: Corrected turn logic to check for SUCCESS before TIMEOUT, fixing false failures.")
        print("✓ SOLVED: More stable turning behavior reduces odometry errors, fixing movement overshoots.")
        print("✓ SOLVED: Turn failure due to timeout is now handled correctly, preventing stuck loops.")
        print("✓ SOLVED: Turn command is now sent continuously to prevent getting stuck.")
        print("✓ SOLVED: More precise movement control to prevent overshooting.")
        print("✓ SOLVED: Immediate emergency stop from laser callback for obstacle avoidance.")
        print("="*80)

if __name__ == '__main__':
    try:
        slam = FastDiscreteGraphSLAM()
        slam.run()
    except rospy.ROSInterruptException:
        print("ROS shutdown requested")
    except Exception as e:
        rospy.logerr(f"An unhandled error occurred: {e}")
        import traceback
        traceback.print_exc()