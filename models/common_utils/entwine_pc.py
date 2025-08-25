import pdal
import numpy as np
import json # Import the json module

# --- Step 1: Define the PDAL pipeline ---
# The pipeline is defined as a JSON string or Python dictionary.
# We use 'readers.ept' to specify the Entwine dataset.
# You can add filters (e.g., 'filters.crop' or 'filters.range') if you only want a subset.
# 'writers.numpy' is used to get the data directly into a NumPy array.

def read_file(entwine_path):
    # Alternatively, define it as a Python dictionary:
    pipeline_definition = [
        {
            "type": "readers.ept",
            "filename": entwine_path
        }
    ]

    pipeline_json_string = json.dumps(pipeline_definition)

    # --- Step 2: Create and Execute the PDAL Pipeline ---
    try:
        # Create a PDAL pipeline object
        pipeline = pdal.Pipeline(pipeline_json_string)
        # You can also use: pipeline = pdal.Pipeline(pipeline_json)

        # Execute the pipeline. This will read the data.
        # For large datasets, this might take some time or require more memory.
        pipeline.execute()

        # --- Step 3: Access the data ---
        # The 'writers.numpy' stage makes the data available as a NumPy array.
        # The data is a structured NumPy array, where each field corresponds to a dimension.
        point_data = pipeline.arrays[0]

        print(f"Successfully read {len(point_data)} points.")
        print(f"Data fields (dimensions): {point_data.dtype.names}")

        # Access specific dimensions, e.g., X, Y, Z coordinates
        x = point_data['X']
        y = point_data['Y']
        z = point_data['Z']

        print(f"\nFirst 5 X coordinates: {x[:5]}")
        print(f"First 5 Y coordinates: {y[:5]}")
        print(f"First 5 Z coordinates: {z[:5]}")

        # If your EPT has other dimensions (e.g., Intensity, Classification), you can access them similarly:
        if 'Intensity' in point_data.dtype.names:
            intensity = point_data['Intensity']
            print(f"First 5 Intensity values: {intensity[:5]}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the 'entwine_path' is correct and accessible.")
        print("For local files, ensure the full path is correct.")

def read_entwine(entwine_path):
    # --- Step 1: Define the PDAL pipeline as a Python dictionary ---
    # IMPORTANT: Removed the 'writers.numpy' stage
    pipeline_definition = [
        {
            "type": "readers.ept",
            "filename": entwine_path
        }
        # No 'writers.numpy' stage here! PDAL's Python bindings handle this automatically.
    ]

    # --- Step 2: Convert the Python dictionary pipeline to a JSON string ---
    pipeline_json_string = json.dumps(pipeline_definition)

    # --- Step 3: Create and Execute the PDAL Pipeline ---
    try:
        # Create a PDAL pipeline object using the JSON string
        pipeline = pdal.Pipeline(pipeline_json_string)

        # Execute the pipeline. This will read the data.
        # The PDAL Python bindings will automatically convert the data to NumPy arrays.
        pipeline.execute()

        # --- Step 4: Access the data ---
        # The 'pipeline.arrays' attribute will contain a list of NumPy arrays.
        # For a simple reader pipeline like this, the data will be in the first element.
        point_data = pipeline.arrays[0]

        print(f"Successfully read {len(point_data)} points.")
        print(f"Data fields (dimensions): {point_data.dtype.names}")

        # Access specific dimensions, e.g., X, Y, Z coordinates
        # Note: Dimension names are case-sensitive and depend on your data.
        # Common names are 'X', 'Y', 'Z', 'Intensity', 'Classification', 'Red', 'Green', 'Blue', etc.
        x = point_data['X']
        y = point_data['Y']
        z = point_data['Z']

        print(f"\nFirst 5 X coordinates: {x[:5]}")
        print(f"First 5 Y coordinates: {y[:5]}")
        print(f"First 5 Z coordinates: {z[:5]}")

        # Example for checking and accessing other common dimensions:
        if 'Intensity' in point_data.dtype.names:
            intensity = point_data['Intensity']
            print(f"First 5 Intensity values: {intensity[:5]}")
        if 'Classification' in point_data.dtype.names:
            classification = point_data['Classification']
            print(f"First 5 Classification values: {classification[:5]}")
        if 'Red' in point_data.dtype.names and 'Green' in point_data.dtype.names and 'Blue' in point_data.dtype.names:
            colors = point_data[['Red', 'Green', 'Blue']]
            print(f"First 5 RGB colors: {colors[:5]}")

        print(f"Z coordinates: {z}")
        print(f"Z coordinates length: {len(z)}")


    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Ensure 'entwine_path' is correct and accessible (full path to 'ept.json').")
        print(
            "2. Verify your PDAL installation: Run `pdal --version` and `python -c 'import pdal; print(pdal.__version__)'` in your Anaconda Prompt.")
        print("3. Check for any firewall or network issues if the EPT dataset is remote.")
        print("4. For local files, ensure Python/your user has read permissions to the directory.")

if __name__ == "__main__":
    # Define the path to your Entwine Point Cloud dataset.
    # This should be the root URL or local path to your EPT JSON file.
    # Example: "https://assets.entwine.io/nyc/ept/" (for a remote dataset)
    # Example: "C:/path/to/my_entwine_data/ept.json" (for a local dataset)
    # entwine_path = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\input\samples\pc_data\entwine_pointcloud\ept.json"
    # entwine_path = r"C:\Users\AdikariAdikari\PycharmProjects\Segmentation\input\samples\pc_data\entwine_pointcloud\ept.json"
    entwine_path = r"C:\Users\AdikariAdikari\DataCollection\DroneSurvey\Crookstown\WebODM\entwine_pointcloud\ept.json"

    # read_file(entwine_path)
    read_entwine(entwine_path)