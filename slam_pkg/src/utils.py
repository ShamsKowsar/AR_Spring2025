#!/usr/bin/env python3

# utils.py

import math
import numpy as np


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q) -> float:
    """
    q: geometry_msgs/Quaternion (has x,y,z,w)
    Returns yaw in radians
    """
    # ROS standard quaternion to yaw
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_point(px: float, py: float, pose_xyz: tuple) -> tuple:
    """
    Transform a point (px,py) in robot frame to world frame given pose (x,y,yaw).
    """
    xr, yr, th = pose_xyz
    c, s = math.cos(th), math.sin(th)
    X = xr + c * px - s * py
    Y = yr + s * px + c * py
    return X, Y


def bresenham(x0, y0, x1, y1):
    """
    Integer Bresenham line. Returns list of (x,y) cells including both ends.
    """
    x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
    points = []
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def bresenham_clip(ix0, iy0, ix1, iy1, max_x, max_y):
    """
    Clip a ray from (ix0,iy0) to (ix1,iy1) to the [0..max_x-1, 0..max_y-1] box,
    then run Bresenham to the clipped endpoint.
    """
    x0, y0 = float(ix0), float(iy0)
    x1, y1 = float(ix1), float(iy1)
    dx, dy = x1 - x0, y1 - y0

    # Liang–Barsky style clipping on the grid box
    p = [-dx, dx, -dy, dy]
    q = [x0 - 0, (max_x - 1) - x0, y0 - 0, (max_y - 1) - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return [] # parallel and outside
            else:
                continue
        t = qi / pi
        if pi < 0:
            if t > u2: return []
            if t > u1: u1 = t
        else:
            if t < u1: return []
            if t < u2: u2 = t

    xx0 = x0 + u1 * dx
    yy0 = y0 + u1 * dy
    xx1 = x0 + u2 * dx
    yy1 = y0 + u2 * dy

    return bresenham(int(round(xx0)), int(round(yy0)),
                     int(round(xx1)), int(round(yy1)))


