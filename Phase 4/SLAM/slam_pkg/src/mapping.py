#!/usr/bin/env python3

# mapping.py

import numpy as np
import os
import yaml
from utils import bresenham, bresenham_clip, transform_point
import math
import rospy


def _logit(p): return np.log(p / (1.0 - p))
def _inv_logit(l): return 1.0 / (1.0 + np.exp(-l))


class OccupancyGridMap:
    """
    Simple log-odds occupancy grid map with inverse sensor model.
    """
    def __init__(self, size_x=5000, size_y=5000, resolution=0.02, origin=None):
        self.size_x = int(size_x)
        self.size_y = int(size_y)
        self.resolution = float(resolution)
        if origin is None:
            # center map at (0,0)
            origin = (-self.size_x * self.resolution / 2.0,
                      -self.size_y * self.resolution / 2.0)
        self.origin = tuple(origin)

        self.log_odds = np.zeros((self.size_y, self.size_x), dtype=np.float32)

        # Log-odds clamps & sensor model
        self.lo_min, self.lo_max = -6.0, 6.0
        self.p_occ, self.p_free = 0.80, 0.25 # Probabilities for updates
        self.l_occ = _logit(self.p_occ)
        self.l_free = _logit(self.p_free)

    # ----- coordinates -----
    def world_to_map(self, x, y):
        ox, oy = self.origin
        ix = int(np.floor((x - ox) / self.resolution))
        iy = int(np.floor((y - oy) / self.resolution))
        return ix, iy

    def map_to_world(self, ix, iy):
        ox, oy = self.origin
        x = ox + (ix + 0.5) * self.resolution
        y = oy + (iy + 0.5) * self.resolution
        return x, y

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.size_x and 0 <= iy < self.size_y

    # ----- update -----
    def update_by_scan(self, robot_pose, ranges, angles, max_range, beam_step=1):

        rospy.loginfo("--- MAP UPDATE CYCLE ---")

        rx, ry, rth = robot_pose
        ix0, iy0 = self.world_to_map(rx, ry)

        # Early exit if robot is off the map
        if not self.in_bounds(ix0, iy0):
            return

        beams_processed = 0

        n = len(ranges)
        for k in range(0, n, beam_step):
            r = ranges[k]
            a = angles[k] # Get the angle for the current beam

            # Handle invalid or max-range readings
            if np.isinf(r) or np.isnan(r) or r > max_range:
                r = max_range

            if k % 100 == 0:
                beams_processed += 1

            ex_robot = r * math.cos(a)
            ey_robot = r * math.sin(a)

            wx, wy = transform_point(ex_robot, ey_robot, (rx, ry, rth))
            ix1, iy1 = self.world_to_map(wx, wy)

            pts = bresenham_clip(ix0, iy0, ix1, iy1, self.size_x, self.size_y)

            if len(pts) == 0:
                continue

            for (cx, cy) in pts[:-1]:
                self.log_odds[cy, cx] = np.clip(self.log_odds[cy, cx] + self.l_free, self.lo_min, self.lo_max)

            lx, ly = pts[-1]

            if r < max_range - 1e-3: # We hit something -> last cell is occupied
                self.log_odds[ly, lx] = np.clip(self.log_odds[ly, lx] + self.l_occ, self.lo_min, self.lo_max)
            else: # Max range reading -> last cell is free
                self.log_odds[ly, lx] = np.clip(self.log_odds[ly, lx] + self.l_free, self.lo_min, self.lo_max)

        rospy.loginfo("Processed %d beams for this scan.", beams_processed)

    def get_prob_map(self):
        return _inv_logit(self.log_odds)

    # ----- export -----
    def to_pgm(self, filename_pgm, free_thresh=0.35, occ_thresh=0.65):
        prob = self.get_prob_map()

        # CORRECTED LOGIC: Use thresholds to create the final image
        # -1 (unknown) in ROS maps becomes gray (205) in PGM
        # 0 (free) becomes white (254)
        # 100 (occupied) becomes black (0)
        img = np.full(prob.shape, 205, dtype=np.uint8) # Default to gray (unknown)
        img[prob < free_thresh] = 254 # free
        img[prob > occ_thresh] = 0 # occupied

        # PGM format requires the image to be flipped vertically
        img_out = np.flipud(img)

        with open(filename_pgm, 'wb') as f:
            header = f"P5\n{self.size_x} {self.size_y}\n255\n"
            f.write(header.encode('ascii'))
            f.write(img_out.tobytes())

        yaml_fn = filename_pgm.replace('.pgm', '.yaml')
        meta = {
            'image': os.path.basename(filename_pgm),
            'resolution': float(self.resolution),
            'origin': [float(self.origin[0]), float(self.origin[1]), 0.0],
            'negate': 0,
            'occupied_thresh': float(occ_thresh),
            'free_thresh': float(free_thresh),
        }

        with open(yaml_fn, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)

        return filename_pgm, yaml_fn


