import pdal
import json
import open3d as o3d
import numpy as np
import laspy


if __name__ == "__main__":
    las_file = "../../input/samples/Task-of-2025-04-02T085015246Z-georeferenced_model.las"
    output_file = "../../input/samples/ground_only.las"

    # PDAL pipeline for ground classification and vegetation removal
    pipeline_json = {
        "pipeline": [
            las_file,
            {
                "type": "filters.smrf",  # Classify ground using Simple Morphological Filter
                "scalar": 1.2,
                "slope": 0.2,
                "threshold": 0.45,
                "window": 16.0
            },
            {
                "type": "filters.range",  # Keep only ground points (classification == 2)
                "limits": "Classification[2:2]"
            },
            {
                "type": "writers.las",
                "filename": output_file
            }
        ]
    }

    # Run PDAL pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    print(f"Filtered LAS saved to: {output_file}")
    # Load the filtered LAS
    las = laspy.read(output_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Show it
    o3d.visualization.draw_geometries([pcd])

