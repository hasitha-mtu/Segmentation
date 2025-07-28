
# Install required libraries: pip install tensorflow spektral scikit-image matplotlib opencv-python

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from spektral.layers import GATConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from skimage.segmentation import slic
from skimage.future import graph
import scipy.sparse as sp

# --- Step 1: Generate a synthetic image with occlusion (for demo purposes) ---
def generate_synthetic_image():
    img = np.ones((128, 128, 3), dtype=np.uint8) * 255
    # Draw some water regions (blue)
    cv2.rectangle(img, (30, 30), (100, 80), (0, 0, 255), -1)
    # Add occlusion (black rectangle)
    cv2.rectangle(img, (50, 50), (80, 100), (0, 0, 0), -1)
    return img

image = generate_synthetic_image()

# Display the synthetic image
plt.imshow(image)
plt.title("Synthetic Image with Occlusion")
plt.axis('off')
plt.show()

# --- Step 2: Superpixel segmentation ---
n_segments = 100  # Adjust for complexity
superpixels = slic(image, n_segments=n_segments, compactness=10)

# Visualize superpixels
plt.imshow(spixels_overlay(image, superpixels))
plt.title("Superpixels Overlay")
plt.axis('off')
plt.show()

def spixels_overlay(image, superpixels):
    overlay = image.copy()
    for label in np.unique(superpixels):
        mask = superpixels == label
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0,255,0), 1)
    return overlay

# --- Step 3: Extract features for each superpixel ---
# Load pretrained CNN
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image, superpixels):
    features = []
    labels = []
    for label in np.unique(superpixels):
        mask = superpixels == label
        region_img = np.zeros_like(image)
        for c in range(3):
            region_img[:, :, c][mask] = image[:, :, c][mask]
        region_resized = tf.image.resize(region_img, (224, 224))
        region_resized = preprocess_input(region_resized)
        input_tensor = tf.expand_dims(region_resized, axis=0)
        feature = base_model.predict(input_tensor)
        features.append(feature.squeeze())
    return np.array(features)

node_features = extract_features(image, superpixels)

# --- Step 4: Build adjacency matrix via RAG ---
rag = graph.rag_mean_color(image, superpixels)
num_nodes = len(np.unique(superpixels))
adjacency = np.zeros((num_nodes, num_nodes))

for edge in rag.edges:
    n1, n2 = edge
    adjacency[n1, n2] = 1
    adjacency[n2, n1] = 1

# --- Step 5: Create labels for superpixels ---
# For demo, define water as superpixels with centroid inside water region
superpixel_labels = []
for label in np.unique(superpixels):
    mask = superpixels == label
    centroid = np.mean(np.argwhere(mask), axis=0)
    y, x = int(centroid[0]), int(centroid[1])
    label_value = 1 if np.array_equal(image[y, x], [0, 0, 255]) else 0
    superpixel_labels.append(label_value)

superpixel_labels = np.array(superpixel_labels)

# --- Step 6: Build graph data (for Spektral) ---
import tensorflow as tf
from spektral.data import Graph

X = node_features
A = adjacency
y = superpixel_labels

# Convert adjacency to sparse tensor
A_sparse = tf.sparse.from_dense(A)

graph_data = Graph(x=X, a=A, y=y)

# --- Step 7: Define GAT model ---
from spektral.layers import GATConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

num_features = X.shape[1]
num_classes = 2

X_in = Input(shape=(num_features,))
A_in = Input((None,), sparse=True)

x = GATConv(8, attn_heads=4, activation='relu')([X_in, A_in])
x = GATConv(num_classes, attn_heads=1, activation='softmax')([x, A_in])

model = Model(inputs=[X_in, A_in], outputs=x)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Step 8: Prepare data for training ---
X_input = np.expand_dims(X, axis=0)
A_input = tf.sparse.expand_dims(A_sparse, axis=0)
y_input = np.expand_dims(y, axis=0)

# --- Step 9: Train the model ---
model.fit([X_input, A_input], y_input, epochs=50, batch_size=1, verbose=1)

# --- Step 10: Inference ---
preds = model.predict([X_input, A_input])
pred_labels = np.argmax(preds.squeeze(), axis=-1)

# Map superpixel predictions back to image
mask_pred = np.zeros(superpixels.shape, dtype=np.uint8)
for label_idx, superpixel_id in enumerate(np.unique(superpixels)):
    mask_pred[superpixels == superpixel_id] = 255 if pred_labels[label_idx] == 1 else 0

# Visualize the segmentation
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(mask_pred, cmap='gray')
plt.title("Predicted Water Mask")
plt.axis('off')

# Overlay on original
overlay = image.copy()
overlay[mask_pred==255] = [0,255,0]
plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Overlay with Detected Water")
plt.axis('off')

plt.show()

# How to Use:
# Replace the synthetic image generation with your actual images and masks.
# Adjust n_segments based on your image complexity.
# Use your dataset for training; this example is for demonstration.
# Final notes:
# This script provides a complete pipeline with dummy data for demonstration.
# For real datasets, load your images/masks accordingly.
# For improved performance, consider batching, data augmentation, and hyperparameter tuning.
