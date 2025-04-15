import pdal
import json
import open3d as o3d
import numpy as np
import laspy


if __name__ == "__main__":
    laz_file = "../../input/samples/G_Sw_Anny.laz"
    output_file = "../../input/samples/filtered_ground_water.las"
    # output_file = "filtered_ground_water.las"

    # las = laspy.read(laz_file)
    # print(set(las.classification))  # See all unique classification values

    # PDAL pipeline for ground classification and vegetation removal

    # pipeline_json = {
    #     "pipeline": [
    #         laz_file,
    #         {
    #             "type": "filters.smrf"
    #         },
    #         {
    #             "type": "filters.range",
    #             "limits": "Classification[2:2]"
    #         },
    #         {
    #             "type": "filters.range",
    #             "limits": "Classification[9:9]"
    #         },
    #         {
    #             "type": "writers.las",
    #             "filename": output_file
    #         }
    #     ]
    # }

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

