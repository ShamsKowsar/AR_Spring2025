#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
import matplotlib.pyplot as plt
import numpy as np

def parse_world_file(world_file_path):
    tree = ET.parse(world_file_path)
    return tree.getroot()

def extract_walls_from_model(root, model_name="vector_world_4"):
    wall_polygons = []

    model = root.find(f".//model[@name='{model_name}']")


    for link in model.findall("link"):
        pose_tag = link.find("pose")
        collision = link.find("collision")
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

        # Read pose and size from XML
        x, y, _, _, _, yaw = map(float, pose_tag.text.strip().split())
        length, width, _ = map(float, size_tag.text.strip().split())

        # Define rectangle centered at origin
        dx, dy = length / 2, width / 2
        rect = Polygon([
            [-dx, -dy],
            [-dx, dy],
            [ dx, dy],
            [ dx, -dy]
        ])

        # Rotate and translate
        rect = rotate(rect, np.degrees(yaw), origin=(0, 0))
        rect = translate(rect, xoff=x, yoff=y)

        print(f"{link.attrib['name']}: x={x:.3f}, y={y:.3f}, yaw={np.degrees(yaw):.1f}°, size=({length}, {width})")
        wall_polygons.append(rect)

    return wall_polygons

def visualize_polygons(polygons):
    fig, ax = plt.subplots()
    for poly in polygons:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='blue', ec='black')
    ax.set_aspect('equal')
    plt.title("Walls Extracted from Custom Maze World")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    world_file = "/home/user/catkin_ws/src/anki_description/world/sample1.world"  
    root = parse_world_file(world_file)
    wall_polygons = extract_walls_from_model(root, model_name="vector_world_4")
    print(f"\nExtracted {len(wall_polygons)} unique wall polygons.\n")
    visualize_polygons(wall_polygons)

