import pdal
import json
import open3d as o3d
import numpy as np
import laspy

def filter_point_cloud(laz_file, output_file):
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": laz_file
            },
            {
                "type": "filters.smrf"
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]",
                "tag": "ground_only"
            },
            {
                "type": "readers.las",
                "filename": laz_file
            },
            {
                "type": "filters.range",
                "limits": "Classification[9:9]",
                "tag": "water_only"
            },
            {
                "type": "filters.merge",
                "inputs": ["ground_only", "water_only"]
            },
            {
                "type": "writers.las",
                "filename": output_file
            }
        ]
    }

    print(json.dumps(pipeline_json, indent=2))
    # Run PDAL pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    print("✅ Filtered point cloud (ground + water) saved.")
    # Load the filtered LAS
    las = laspy.read(output_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Show it
    o3d.visualization.draw_geometries([pcd])

def read_laz_file(file_path):
    las = laspy.read(file_path)
    print(set(las.classification))

def edit_laz_file(input_file, output_file):
    # Open the LAZ file
    laz_file = laspy.read(input_file)

    # Get the Z-axis range directly from the header
    z_min = laz_file.header.z_min
    z_max = laz_file.header.z_max

    print(f"Z-axis minimum: {z_min:.2f}")
    print(f"Z-axis maximum: {z_max:.2f}")

    # Get the Z coordinates and create a mask
    z_coords = laz_file.z
    points_to_keep = z_coords <= 20
    filtered_points = laz_file.points[points_to_keep]

    # Create a new LasData object
    new_laz_file = laspy.LasData(laz_file.header, points=filtered_points)

    # Write the new data to a new LAZ file
    new_laz_file.write(output_file)

    print(f"Original points: {len(laz_file.points)}")
    print(f"Edited points: {len(new_laz_file.points)}")

if __name__ == "__main__":
    input_file = "D:\DataCollection\DroneSurveys\Crookstown\WebODM\odm_georeferencing\odm_georeferenced_model.laz"
    output_file = "../../output/point_cloud/updated_odm_georeferenced_model.laz"
    edit_laz_file(input_file, output_file)

# if __name__ == "__main__":
#     # input_file = "../../input/samples/G_Sw_Anny.laz"
#     input_file = "../../input/samples/Cloghmacow-Road-3-24-2025-georeferenced_model.laz"
#     output_file = "../../input/samples/filtered_ground_water.las"
#     filter_point_cloud(input_file, output_file)

