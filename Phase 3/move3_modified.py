#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Range
from gazebo_msgs.msg import ModelStates
import rosgraph_msgs.msg
import random
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate
import matplotlib.pyplot as plt
from scipy.stats import norm
import threading

def parse_world_file(world_file_path):
    try:
        tree = ET.parse(world_file_path)
        return tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        rospy.logerr(f"Error reading or parsing world file: {e}")
        return None

def extract_walls_from_model(root, model_name="vector_world_4"):
    wall_polygons = []
    if root is None: return wall_polygons
    model = root.find(f".//model[@name='{model_name}']")
    if model is None:
        rospy.logwarn(f"Model '{model_name}' not found.")
        return wall_polygons
    for link in model.findall("link"):
        pose_tag, collision = link.find("pose"), link.find("collision")
        if collision is None: continue
        geometry = collision.find("geometry")
        if geometry is None: continue
        box = geometry.find("box")
        if box is None: continue
        size_tag = box.find("size")
        if pose_tag is None or size_tag is None: continue
        x, y, _, _, _, yaw = map(float, pose_tag.text.strip().split())
        length, width, _ = map(float, size_tag.text.strip().split())
        dx, dy = length / 2, width / 2
        rect = Polygon([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]])
        rect = rotate(rect, np.degrees(yaw), origin=(0, 0))
        rect = translate(rect, xoff=x, yoff=y)
        wall_polygons.append(rect)
    return wall_polygons

class ParticleFilter:
    def __init__(self, num_particles, map_polygons, map_bounds):
        self.num_particles = num_particles
        self.map_polygons = map_polygons
        self.map_bounds = map_bounds
        self.particles = self._create_uniform_particles()
        self.weights = [1.0 / self.num_particles] * self.num_particles
        self.motion_noise_x = 0.01
        self.motion_noise_theta = 0.02
        self.sensor_noise_std = 0.05

    def _create_uniform_particles(self):
        particles = []
        min_x, max_x, min_y, max_y = self.map_bounds
        while len(particles) < self.num_particles:
            x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
            if not any(wall.contains(Point(x, y)) for wall in self.map_polygons):
                p = Pose()
                p.position.x, p.position.y = x, y
                yaw = random.uniform(-np.pi, np.pi)
                p.orientation.z, p.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
                particles.append(p)
        return particles

    def predict(self, v, w, dt):
        """
        Propagate each particle using unicycle kinematics *with the correct
        Gazebo-to-world axis mapping* (body x → world +y).
        """
        for p in self.particles:
            # current yaw in world frame
            yaw = 2 * np.arctan2(p.orientation.z, p.orientation.w)

            # noisy control
            v_noisy = v + np.random.normal(0, self.motion_noise_x)
            w_noisy = w + np.random.normal(0, self.motion_noise_theta)

            # --- 90° swap: body-x is +y in world --------------------
            p.position.x += v_noisy * np.sin(yaw) * dt   # cos → sin
            p.position.y += v_noisy * np.cos(yaw) * dt   # sin → cos

            # integrate yaw normally
            yaw = (yaw + w_noisy * dt + np.pi) % (2 * np.pi) - np.pi
            p.orientation.z, p.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)

    def update(self, measured_distance):
        weights = []
        for p in self.particles:
            yaw = 2 * np.arctan2(p.orientation.z, p.orientation.w)
            expected_distance = self._get_expected_range(p.position.x, p.position.y, yaw)
            weight = norm.pdf(measured_distance, loc=expected_distance, scale=self.sensor_noise_std)
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight > 1e-9:
            self.weights = [w / total_weight for w in weights]
        else:
            self.weights = [1.0 / self.num_particles] * self.num_particles

    def _get_expected_range(self, x, y, theta):
        max_range = 10.0
        origin_point = Point(x, y)
        ray = LineString([origin_point, (x + max_range * np.cos(theta), y + max_range * np.sin(theta))])
        min_dist = max_range
        for wall in self.map_polygons:
            intersection = wall.intersection(ray)
            if not intersection.is_empty:
                dist = origin_point.distance(intersection)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def resample(self):
        new_particles = []
        indices = np.arange(self.num_particles)
        resampled_indices = np.random.choice(indices, size=self.num_particles, p=self.weights, replace=True)
        
        for i in resampled_indices:
            p_orig = self.particles[i]
            p_new = Pose()
            p_new.position.x = p_orig.position.x
            p_new.position.y = p_orig.position.y
            p_new.orientation.z = p_orig.orientation.z
            p_new.orientation.w = p_orig.orientation.w
            new_particles.append(p_new)
        self.particles = new_particles

    def estimate_pose(self):
        est_pose = Pose()
        x_mean, y_mean, z_sin_sum, w_cos_sum = 0, 0, 0, 0
        for p, w in zip(self.particles, self.weights):
            x_mean += p.position.x * w
            y_mean += p.position.y * w
            yaw = 2 * np.arctan2(p.orientation.z, p.orientation.w)
            w_cos_sum += np.cos(yaw) * w
            z_sin_sum += np.sin(yaw) * w
        est_pose.position.x, est_pose.position.y = x_mean, y_mean
        avg_yaw = np.arctan2(z_sin_sum, w_cos_sum)
        est_pose.orientation.z, est_pose.orientation.w = np.sin(avg_yaw/2), np.cos(avg_yaw/2)
        return est_pose

    def get_convergence(self):
        if not self.particles: return float('inf')
        x_coords = [p.position.x for p in self.particles]
        y_coords = [p.position.y for p in self.particles]
        return max(np.var(x_coords), np.var(y_coords))

class LocalizationNode:
    def __init__(self):
        rospy.init_node('particle_filter_localizer', anonymous=True)
        self.world_file_path = rospy.get_param('~world_file', "/home/kowsar/catkin_ws/src/anki_description/world/sample1.world")
        self.num_particles = rospy.get_param('~num_particles', 1200)
        self.convergence_threshold = rospy.get_param('~convergence_threshold', 0.00001)
        
        self.actual_pose, self.estimated_pose = None, None
        self.last_twist = Twist()
        self.last_time = rospy.Time.now()
        self.is_localized = False
        self.plot_lock = threading.Lock()
        # --- new: buffers that will store the path history ---
        self.actual_path_xy    = []       # list of (x, y) tuples
        self.estimated_path_xy = []


        root = parse_world_file(self.world_file_path)
        self.wall_polygons = extract_walls_from_model(root)
        if not self.wall_polygons:
            rospy.logerr("No walls extracted. Shutting down.")
            return

        all_x = [x for poly in self.wall_polygons for x in poly.exterior.xy[0]]
        all_y = [y for poly in self.wall_polygons for y in poly.exterior.xy[1]]
        self.map_bounds = (min(all_x), max(all_x), min(all_y), max(all_y))
        self.particle_filter = ParticleFilter(self.num_particles, self.wall_polygons, self.map_bounds)

        self.cmd_pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/vector/laser', Range, self.laser_callback, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)
        rospy.Subscriber('/vector/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        rospy.loginfo("Particle filter localization node started.")

    def cmd_vel_callback(self, msg): self.last_twist = msg
    def model_states_callback(self, msg):
        try: self.actual_pose = msg.pose[msg.name.index('vector')]
        except ValueError: pass

    def laser_callback(self, scan):
        if self.is_localized or self.actual_pose is None: return
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        if dt <= 0: return
        self.last_time = current_time

        self.particle_filter.predict(self.last_twist.linear.x, self.last_twist.angular.z, dt)
        self.particle_filter.update(scan.range)
        self.particle_filter.resample()
        self.estimated_pose = self.particle_filter.estimate_pose()
        convergence = self.particle_filter.get_convergence()
        rospy.loginfo_throttle(1.0, f"Particle convergence (variance): {convergence:.4f}")
        print(self.estimated_pose)
        print(convergence)
        print('----------------------------------------------------------------------')
        if convergence < self.convergence_threshold:
            print(convergence)
            print()
            self.is_localized = True
            self.stop_robot_and_report()
    def setup_plot(self):
        """Initialise the Matplotlib figure & artists."""
        with self.plot_lock:
            # --- global axes styling -------------------------------------------
            self.ax.set_aspect('equal')
            self.ax.set_title("Particle Filter Localization")
            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.grid(True)

            # --- draw static map walls -----------------------------------------
            for poly in self.wall_polygons:
                self.ax.fill(*poly.exterior.xy,
                            fc='lightgray', ec='black', alpha=0.5)

            # --- live particle cloud (orange dots) -----------------------------
            self.particles_plot, = self.ax.plot([], [], 'o',
                                                color='orange', markersize=2,
                                                alpha=0.5, label='Particles')

            # --- live path traces ----------------------------------------------
            # (buffers are filled in visualize(); we just create empty lines here)
            self.actual_path_line,    = self.ax.plot([], [], '-',
                                                    color='red',  lw=1,
                                                    label='Actual Path')
            self.estimated_path_line, = self.ax.plot([], [], '-',
                                                    color='blue', lw=1,
                                                    label='Estimated Path')

            # --- heading arrows (quivers) --------------------------------------
            # start invisible by placing them at NaN and setting alpha=0
            self.actual_quiver = self.ax.quiver(np.nan, np.nan, np.nan, np.nan,
                                                color='red',  alpha=0.0,
                                                scale=1, scale_units='xy',
                                                angles='xy', label='Actual Heading')

            self.estimated_quiver = self.ax.quiver(np.nan, np.nan, np.nan, np.nan,
                                                color='blue', alpha=0.0,
                                                scale=1, scale_units='xy',
                                                angles='xy', label='Estimated Heading')

            # --- axes limits & legend ------------------------------------------
            self.ax.set_xlim(self.map_bounds[0], self.map_bounds[1])
            self.ax.set_ylim(self.map_bounds[2], self.map_bounds[3])
            self.ax.legend(loc='upper right')

    def visualize(self):
        with self.plot_lock:
            # — 1.  update scatter of particles  —
            self.particles_plot.set_data(
                [p.position.x for p in self.particle_filter.particles],
                [p.position.y for p in self.particle_filter.particles])

            # — 2.  extend and draw the actual & estimated paths  —
            def append_and_update_path(path_buf, line, pose):
                if pose:
                    path_buf.append((pose.position.x, pose.position.y))
                    xs, ys = zip(*path_buf)
                    line.set_data(xs, ys)

            append_and_update_path(self.actual_path_xy,    self.actual_path_line,    self.actual_pose)
            append_and_update_path(self.estimated_path_xy, self.estimated_path_line, self.estimated_pose)

            # — 3.  update quivers (heading arrows)  —
            def update_quiver(quiver, pose):
                if pose:
                    arrow_len = 0.25
                    yaw = 2*np.arctan2(pose.orientation.z, pose.orientation.w)
                    quiver.set_offsets([[pose.position.x, pose.position.y]])
                    quiver.set_UVC(arrow_len*np.cos(yaw), arrow_len*np.sin(yaw))
                    quiver.set_alpha(0.8)           # (re‑enable if hidden)
                else:
                    quiver.set_offsets([[np.nan, np.nan]])
                    quiver.set_UVC(np.nan, np.nan)
                    quiver.set_alpha(0.0)           # hide until first pose arrives

            update_quiver(self.actual_quiver,    self.actual_pose)
            update_quiver(self.estimated_quiver, self.estimated_pose)

    def stop_robot_and_report(self):
        """ ## CORRECTED: Fetches orientation from the correct message field. """
        rospy.loginfo("Convergence threshold reached. Robot localized!")
        self.cmd_pub.publish(Twist())
        
        if self.actual_pose and self.estimated_pose:
            # Separate position and orientation for clarity
            est_pos = self.estimated_pose.position
            actual_pos = self.actual_pose.position
            est_orient = self.estimated_pose.orientation
            actual_orient = self.actual_pose.orientation

            # Calculate position error
            pos_error = np.sqrt((actual_pos.x - est_pos.x)**2 + (actual_pos.y - est_pos.y)**2)
            
            # CORRECTLY calculate yaw from the orientation part of the pose
            actual_yaw = 2 * np.arctan2(actual_orient.z, actual_orient.w)
            est_yaw = 2 * np.arctan2(est_orient.z, est_orient.w)
            angle_error = abs((actual_yaw - est_yaw + np.pi) % (2 * np.pi) - np.pi)

            # Print the final report
            print("\n" + "="*45)
            print("========= LOCALIZATION COMPLETE =========")
            print("="*45)
            print("\n--- Final Estimated Position and Angle ---")
            print(f"  > Position (x,y,z): ({est_pos.x:.3f}, {est_pos.y:.3f}, {est_pos.z:.3f})")
            print(f"  > Angle (degrees):  {np.degrees(est_yaw):.2f}°")
            print("\n---   Actual Position and Angle    ---")
            print(f"  > Position (x,y,z): ({actual_pos.x:.3f}, {actual_pos.y:.3f}, {actual_pos.z:.3f})")
            print(f"  > Angle (degrees):  {np.degrees(actual_yaw):.2f}°")
            print("\n---      Final Estimation Error      ---")
            print(f"  > Position Error: {pos_error:.4f} meters")
            print(f"  > Angle Error:    {np.degrees(angle_error):.2f}°")
            print("\n" + "="*45 + "\n")

if __name__ == '__main__':
    try:
        plt.ion()
        node = LocalizationNode()
        if not node.wall_polygons:
             raise rospy.ROSInterruptException("Failed to load map, cannot run node.")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if node.is_localized:
                rospy.loginfo("Localization complete. Preparing final plot.")
                break
            node.visualize()


            with node.plot_lock:
                node.fig.canvas.draw_idle()
            node.fig.canvas.flush_events()
                
            rate.sleep()

    except rospy.ROSInterruptException:
        print("ROS shutdown request received.")
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred: {e}")
    finally:
        # Final visualization update before showing the blocking plot
        if 'node' in locals() and node.is_localized:
            node.visualize()
        print("Displaying final plot. Close the plot window to exit.")
        plt.ioff()
        plt.show()
