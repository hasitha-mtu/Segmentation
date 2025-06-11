import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from model import (Up, Down, FFCBlock, DoubleConv, OutConv, UNetWithLaMaFeaturesTF,
                   psnr_metric, ssim_metric)
from PIL import Image

# --- Main Inference Block ---
if __name__ == '__main__':
    model_path = "C:/Users/AdikariAdikari/PycharmProjects/Segmentation/models/unet_lama/ckpt/unet_lama_best_model"

    # Load the model with custom_objects
    try:
        # Ensure all custom classes are passed in custom_objects
        loaded_model = tf.keras.models.load_model(model_path, custom_objects={
            "UNetWithLaMaFeaturesTF": UNetWithLaMaFeaturesTF,
            "FFCBlock": FFCBlock,
            "DoubleConv": DoubleConv,
            "Down": Down,
            "Up": Up,
            "OutConv": OutConv,
            "psnr_metric": psnr_metric, # If these were used in model.compile
            "ssim_metric": ssim_metric
        })
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        exit() # Exit if model loading fails

    input_image_path = "../../input/samples/segnet_256/images/DJI_20250324092908_0001_V.jpg" # Make sure this path is correct

    # --- Image Loading and Preprocessing ---
    try:
        image_pil = Image.open(input_image_path).convert('RGB')
        image_array = np.array(image_pil)
    except FileNotFoundError:
        print(f"Error: Image file not found at {input_image_path}")
        # Create a dummy image for demonstration if file not found
        # This dummy image IS in 0-255 range, so it *will* need normalization below.
        image_array = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
        print("Using a randomly generated dummy image for demonstration (values 0-255).")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    print(f"Original image shape: {image_array.shape}, dtype: {image_array.dtype}")
    print(f"Original image pixel value example (top-left): {image_array[0,0,:]}")

    print(f"DEBUG (pre-norm): image_array shape: {image_array.shape}")
    print(f"DEBUG (pre-norm): image_array dtype: {image_array.dtype}")
    print(f"DEBUG (pre-norm): image_array min value: {np.min(image_array)}")
    print(f"DEBUG (pre-norm): image_array max value: {np.max(image_array)}")

    # --- Step 2: Normalize the image to 0-1 range (ADJUSTED LOGIC) ---
    if image_array.max() > 1.0 or image_array.dtype == np.uint8:
        print("Detected unnormalized image (max > 1 or uint8 dtype). Normalizing now.")
        normalized_image_array = image_array.astype(np.float32) / 255.0
    else:
        print("Image appears to be already normalized (0.0-1.0 float). Skipping 255 division.")
        normalized_image_array = image_array.astype(np.float32)

    print(f"Normalized image shape: {normalized_image_array.shape}, dtype: {normalized_image_array.dtype}")
    print(f"Normalized image pixel value example (top-left): {normalized_image_array[0,0,:]}")

    # --- Step 3: Add batch dimension ---
    input_for_model = np.expand_dims(normalized_image_array, axis=0)

    print(f"Input for model shape: {input_for_model.shape}, dtype: {input_for_model.dtype}")
    print(f"Input for model min value: {np.min(input_for_model)}")
    print(f"Input for model max value: {np.max(input_for_model)}")

    # --- Primary Test: Predict with a dummy all-grey image (0.5) ---
    print("\n--- Testing with a dummy all-grey image ---")
    dummy_input = np.full((1, 256, 256, 3), 0.5, dtype=np.float32)
    print(f"DEBUG (dummy): shape={dummy_input.shape}, dtype={dummy_input.dtype}, min={np.min(dummy_input)}, max={np.max(dummy_input)}")

    try:
        dummy_predictions = loaded_model.predict(dummy_input)
        print("Prediction with dummy all-grey image successful!")
        print(f"Dummy predictions shape: {dummy_predictions.shape}")
        print(f"Dummy predictions min value: {np.min(dummy_predictions)}")
        print(f"Dummy predictions max value: {np.max(dummy_predictions)}")
    except Exception as e:
        print(f"Error predicting with dummy all-grey image: {e}")
        print("This failure indicates an issue with the model itself or the TensorFlow environment.")
        print("Please investigate TensorFlow version compatibility or the model's training normalization.")
        print("If the dummy image fails, the problem is not with your input image loading/normalization.")
        exit() # Exit if dummy test fails, as the actual image will also fail.

    # --- Actual Prediction with your image ---
    print("\n--- Attempting prediction with your provided image ---")
    try:
        predictions = loaded_model.predict(input_for_model)
        print("Prediction with actual image successful!")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions min value: {np.min(predictions)}")
        print(f"Predictions max value: {np.max(predictions)}")
    except Exception as e:
        print(f"Error predicting with actual image: {e}")
        print("If the dummy image worked, but this failed, there might still be an issue with your specific image's content or subtle numerical properties.")

    # --- Optional: Inspect Model Input Spec (if dummy test passes) ---
    print("\n--- Model Input/Output Specs ---")
    try:
        if hasattr(loaded_model, 'input_spec'):
            print(f"Model Input Spec: {loaded_model.input_spec}")
        if hasattr(loaded_model, 'input_shape'):
            print(f"Model Input Shape: {loaded_model.input_shape}")
        if hasattr(loaded_model, 'output_shape'):
            print(f"Model Output Shape: {loaded_model.output_shape}")
    except Exception as e:
        print(f"Error inspecting model specs: {e}")

