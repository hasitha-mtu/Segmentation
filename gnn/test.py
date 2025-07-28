# Step 1: Data Preparation

# a) Load your dataset
# Assuming you have a folder with RGB images and corresponding pixel-wise annotations (masks).

import os
import cv2
import numpy as np

# Paths
images_dir = 'path/to/images/'
masks_dir = 'path/to/masks/'

# List image files
image_files = sorted(os.listdir(images_dir))
mask_files = sorted(os.listdir(masks_dir))

# b) Load a sample image and mask
def load_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

# Example
img_path = os.path.join(images_dir, image_files[0])
mask_path = os.path.join(masks_dir, mask_files[0])
image, mask = load_image_mask(img_path, mask_path)

# Step 2: Superpixel Segmentation

from skimage.segmentation import slic

def generate_superpixels(image, n_segments=200):
    superpixels = slic(image, n_segments=n_segments, compactness=10)
    return superpixels

superpixels = generate_superpixels(image)

# Step 3: Extract Features per Superpixel

# a) Load pretrained CNN (MobileNetV2)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# b) Extract features for each superpixel
def extract_superpixel_features(image, superpixels):
    features = []
    labels = []
    for label in np.unique(superpixels):
        mask = superpixels == label
        # Create a blank image
        region_img = np.zeros_like(image)
        for c in range(3):  # RGB channels
            region_img[:, :, c][mask] = image[:, :, c][mask]
        # Resize to model input size
        region_resized = tf.image.resize(region_img, (224, 224))
        region_resized = preprocess_input(region_resized)
        input_tensor = tf.expand_dims(region_resized, axis=0)
        feature = base_model.predict(input_tensor)
        features.append(feature.squeeze())
    return np.array(features)

node_features = extract_superpixel_features(image, superpixels)

# c) Create labels for superpixels

# Assuming the mask is binary (water=1, non-water=0):
superpixel_labels = []
for label in np.unique(superpixels):
    mask = superpixels == label
    # Majority voting within superpixel
    label_mask = mask & (mask > 0)
    label_value = np.bincount(mask[mask].flatten()).argmax()
    superpixel_labels.append(label_value)
superpixel_labels = np.array(superpixel_labels)

# Step 4: Build Graph Connectivity

# a) Determine adjacency of superpixels
import networkx as nx
from skimage.future import graph

# Using RAG (Region Adjacency Graph)
rag = graph.rag_mean_color(image, superpixels)

# Build adjacency matrix
num_nodes = len(np.unique(superpixels))
adjacency = np.zeros((num_nodes, num_nodes))

for edge in rag.edges:
    n1 = edge[0]
    n2 = edge[1]
    adjacency[n1, n2] = 1
    adjacency[n2, n1] = 1

# Step 5: Prepare Data for Spektral
import tensorflow as tf
from spektral.data import Dataset, Graph

# Create a Graph object
graph = Graph(x=node_features, a=adjacency, y=superpixel_labels)
dataset = [graph]

# Step 6: Define GNN Model
from spektral.layers import GATConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

num_node_features = node_features.shape[1]
num_classes = 2  # water / non-water

X_in = Input(shape=(num_node_features,))
A_in = Input((None,), sparse=True)

# GAT layer
x = GATConv(64, attn_heads=4, activation='relu')([X_in, A_in])
x = GATConv(num_classes, attn_heads=1, activation='softmax')([x, A_in])

model = Model(inputs=[X_in, A_in], outputs=x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model

# Prepare inputs
X = node_features
A = tf.sparse.from_dense(adjacency)
y = superpixel_labels

# Expand dims for batch dimension
X = np.expand_dims(X, axis=0)
A = tf.sparse.expand_dims(A, axis=0)
y = np.expand_dims(y, axis=0)

# Train
model.fit([X, A], y, epochs=50, batch_size=1)

# Step 8: Inference and Mapping Back to Pixels

# Predict superpixel labels
preds = model.predict([X, A])
pred_labels = np.argmax(preds.squeeze(), axis=-1)

# Map superpixel labels back to pixel mask
pixel_mask = np.zeros(superpixels.shape, dtype=np.uint8)
for superpixel_id, label in enumerate(pred_labels):
    pixel_mask[superpixels == superpixel_id] = label * 255  # 0 or 255

# Display or save the result
import matplotlib.pyplot as plt
plt.imshow(pixel_mask, cmap='gray')
plt.show()

# Summary of the Pipeline
# Data Loading: Load images and masks.
# Superpixel Segmentation: Divide images into superpixels.
# Feature Extraction: Use a pretrained CNN to extract features per superpixel.
# Graph Construction: Build a graph where nodes are superpixels, edges encode adjacency.
# GNN Model: Define a GAT-based GNN with Spektral.
# Training: Train the model to classify superpixels.
# Inference: Predict labels and reconstruct pixel-wise segmentation.
# Additional Tips
# Batch Processing: For multiple images, create a custom Dataset class to handle batching.
# Data Augmentation: Apply transformations to images/masks to improve robustness.
# Occlusion Handling: Incorporate synthetic occlusions during training for robustness.
# Hyperparameter Tuning: Adjust number of superpixels, GNN layers, hidden units, learning rate, etc.
