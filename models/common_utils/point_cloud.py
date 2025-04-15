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

# if __name__ == "__main__":
#     input_file = "../../input/samples/Cloghmacow-Road-3-24-2025-georeferenced_model.laz"
#     read_laz_file(input_file)

if __name__ == "__main__":
    # input_file = "../../input/samples/G_Sw_Anny.laz"
    input_file = "../../input/samples/Cloghmacow-Road-3-24-2025-georeferenced_model.laz"
    output_file = "../../input/samples/filtered_ground_water.las"
    filter_point_cloud(input_file, output_file)

