#!/usr/bin/env python
# coding: utf-8

# # Palette-Based Image Decomposition to Color Layers with Alpha Channel
# 
# This notebook implements the algorithm described in "Image Decomposition using Geometric Region Colour Unmixing" by Mairéad Grogan and Aljosa Smolic. The algorithm extracts a palette from an image and then decomposes the image into separate layers, each corresponding to a palette color with an alpha channel.

# ## Import Required Libraries

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from scipy.spatial import ConvexHull, Delaunay
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from tqdm.auto import tqdm
    
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ## Define Utility Functions

def load_image(filename, downsample=None):
    """
    Load an image and optionally downsample it.
    
    Args:
        filename: Path to the image file
        downsample: Factor to downsample the image by (if None, no downsampling)
    
    Returns:
        image: RGB image as a numpy array with values in [0, 1]
    """
    image = io.imread(filename)
    
    # Convert to float in [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Ensure image has 3 channels (RGB)
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] > 3:  # RGBA or other
        image = image[:, :, :3]
    
    # Downsample if requested
    if downsample is not None:
        h, w = image.shape[:2]
        new_h, new_w = h // downsample, w // downsample
        image = resize(image, (new_h, new_w), anti_aliasing=True)
    
    return image

def visualize_palette(palette, figsize=(10, 2)):
    """
    Visualize a color palette.
    
    Args:
        palette: Array of RGB colors
        figsize: Figure size
    """
    n_colors = palette.shape[0]
    fig, ax = plt.subplots(1, n_colors, figsize=figsize)
    
    for i in range(n_colors):
        ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=palette[i]))
        ax[i].set_xlim([0, 1])
        ax[i].set_ylim([0, 1])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f"Color {i+1}")
    
    plt.tight_layout()
    plt.show()

def visualize_layers(layers, figsize=(15, 10)):
    """
    Visualize image layers (RGB + alpha).
    
    Args:
        layers: List of RGBA images
        figsize: Figure size
    """
    n_layers = len(layers)
    fig, axes = plt.subplots(2, n_layers, figsize=figsize)
    
    for i in range(n_layers):
        # Show RGB part
        axes[0, i].imshow(layers[i][:, :, :3])
        axes[0, i].set_title(f"Layer {i+1} RGB")
        axes[0, i].axis('off')
        
        # Show alpha channel
        axes[1, i].imshow(layers[i][:, :, 3], cmap='gray')
        axes[1, i].set_title(f"Layer {i+1} Alpha")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_3d_rgb(points, palette=None, subsample=1000):
    """
    Plot points in 3D RGB space.
    
    Args:
        points: RGB points to plot
        palette: Optional palette points to highlight
        subsample: Number of points to subsample for visualization
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample points for visualization
    if len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points_sub = points[indices]
    else:
        points_sub = points
    
    # Plot the points
    ax.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
               c=points_sub, s=5, alpha=0.5)
    
    # Plot the palette points if provided
    if palette is not None:
        ax.scatter(palette[:, 0], palette[:, 1], palette[:, 2], 
                   c=palette, s=100, edgecolor='black')
    
    # Set the labels and limits
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    
    plt.tight_layout()
    plt.show()

# ## Palette Extraction

def extract_palette(image, n_colors=5):
    """
    Extract a color palette using the method described in "Image Decomposition using Geometric Region Colour Unmixing".
    This method is based on Aksoy et al. but refined for the paper's layer decomposition approach.
    
    The method divides the RGB space into bins, and iteratively adds colors to the palette
    by finding the most representative colors in the image that are not well-represented
    by existing palette colors.
    
    Args:
        image: RGB image
        n_colors: Number of colors to extract
    
    Returns:
        palette: Array of RGB colors
    """
    # Downsample the image for faster processing
    h, w = image.shape[:2]
    
    # Reshape to list of pixels
    pixels = image.reshape(-1, 3)
    
    # Create a histogram of colors by dividing RGB space into bins
    n_bins = 10  # 10x10x10 bins as mentioned in the paper
    hist_edges = np.linspace(0, 1, n_bins+1)
    
    # Create 3D histogram
    hist, edges = np.histogramdd(pixels, bins=[hist_edges, hist_edges, hist_edges])
    
    # Get centers of histogram bins
    bin_centers = []
    for i in range(3):
        centers = (edges[i][:-1] + edges[i][1:]) / 2
        bin_centers.append(centers)
    
    # Initialize palette with the most frequent color
    palette = []
    votes = np.copy(hist)
    
    for _ in range(n_colors):
        # Find bin with highest number of votes
        max_bin_idx = np.unravel_index(np.argmax(votes), hist.shape)
        
        # Get the RGB value for this bin center
        r = bin_centers[0][max_bin_idx[0]]
        g = bin_centers[1][max_bin_idx[1]]
        b = bin_centers[2][max_bin_idx[2]]
        
        # Add to palette
        palette.append([r, g, b])
        
        # Update votes to avoid picking similar colors in the future
        # Each pixel votes based on how far it is from existing palette colors
        bin_indices = np.mgrid[0:n_bins, 0:n_bins, 0:n_bins].reshape(3, -1).T
        for bin_idx in bin_indices:
            r_bin, g_bin, b_bin = bin_idx
            bin_color = np.array([bin_centers[0][r_bin], bin_centers[1][g_bin], bin_centers[2][b_bin]])
            
            # Check if the bin is well-represented by the current palette
            min_dist = float('inf')
            for color in palette:
                dist = np.sum((bin_color - color)**2)
                min_dist = min(min_dist, dist)
            
            # Vote is reduced based on how well the bin is represented
            votes[r_bin, g_bin, b_bin] *= min_dist
    
    # Refine palette colors by finding the actual most representative pixel in each region
    refined_palette = []
    for palette_color in palette:
        # Find pixels closest to this palette color
        dists = np.sum((pixels - palette_color)**2, axis=1)
        closest_indices = np.argsort(dists)[:10]  # Consider top 100 closest pixels
        
        # Take the average of these pixels for a refined color
        refined_color = np.mean(pixels[closest_indices], axis=0)
        refined_palette.append(refined_color)
    
    # Ensure palette colors are well distributed in RGB space for better layer decomposition
    # Handle colors near RGB cube boundaries (as mentioned in the paper)
    for i, color in enumerate(refined_palette):
        # Check if color is very close to RGB boundary (values > 0.98 or < 0.02)
        for c in range(3):
            if color[c] > 0.98:
                refined_palette[i][c] = 0.98
            elif color[c] < 0.02:
                refined_palette[i][c] = 0.02
    
    return np.array(refined_palette)


# More advanced spatial-aware palette extraction that handles small but visually important regions
def extract_palette_gpu(image, n_colors=5):
    """
    GPU-accelerated version of palette extraction using the method described 
    in "Image Decomposition using Geometric Region Colour Unmixing".
    
    The method divides the RGB space into bins and iteratively adds colors to the palette
    by finding the most representative colors in the image that are not well-represented
    by existing palette colors.
    
    Args:
        image: RGB image
        n_colors: Number of colors to extract
    
    Returns:
        palette: Array of RGB colors
    """
    
    # Ensure we're using the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Reshape to list of pixels
    pixels = image.reshape(-1, 3)
    
    # Create a histogram of colors by dividing RGB space into bins
    n_bins = 10  # 10x10x10 bins as mentioned in the paper
    hist_edges = np.linspace(0, 1, n_bins+1)
    
    # Create 3D histogram (this step is more efficient on CPU)
    hist, edges = np.histogramdd(pixels, bins=[hist_edges, hist_edges, hist_edges])
    
    # Get centers of histogram bins
    bin_centers = []
    for i in range(3):
        centers = (edges[i][:-1] + edges[i][1:]) / 2
        bin_centers.append(centers)
    
    # Initialize palette with the most frequent color
    palette = []
    votes = np.copy(hist)
    
    # Transfer bin centers to GPU for faster distance calculations
    bin_indices = np.mgrid[0:n_bins, 0:n_bins, 0:n_bins].reshape(3, -1).T
    bin_colors = np.zeros((len(bin_indices), 3))
    
    for i, (r_bin, g_bin, b_bin) in enumerate(bin_indices):
        bin_colors[i] = [bin_centers[0][r_bin], bin_centers[1][g_bin], bin_centers[2][b_bin]]
    
    bin_colors_tensor = torch.tensor(bin_colors, dtype=torch.float32, device=device)
    
    # Iteratively extract colors
    for _ in range(n_colors):
        # Find bin with highest number of votes (CPU operation)
        max_bin_idx = np.unravel_index(np.argmax(votes), hist.shape)
        
        # Get the RGB value for this bin center
        r = bin_centers[0][max_bin_idx[0]]
        g = bin_centers[1][max_bin_idx[1]]
        b = bin_centers[2][max_bin_idx[2]]
        
        # Add to palette
        palette.append([r, g, b])
        
        # Update votes to avoid picking similar colors in the future
        # Move palette to GPU for faster distance calculations
        palette_tensor = torch.tensor(palette, dtype=torch.float32, device=device)
        
        # Calculate distances between all bin colors and all palette colors (GPU operation)
        # Shape: [n_bins^3, n_palette_colors]
        distances = torch.cdist(bin_colors_tensor, palette_tensor, p=2)
        
        # Find minimum distance for each bin color to any palette color
        min_distances, _ = torch.min(distances, dim=1)
        
        # Transfer min_distances back to CPU for vote updating
        min_distances_np = min_distances.cpu().numpy()
        
        # Update votes (CPU operation since votes array is on CPU)
        for i, (r_bin, g_bin, b_bin) in enumerate(bin_indices):
            votes[r_bin, g_bin, b_bin] *= min_distances_np[i]
    
    # Refine palette colors by finding the actual most representative pixel in each region
    # Move pixels to GPU for faster distance calculation
    pixels_tensor = torch.tensor(pixels, dtype=torch.float32, device=device)
    refined_palette = []
    
    for palette_color in palette:
        palette_color_tensor = torch.tensor(palette_color, dtype=torch.float32, device=device)
        
        # Calculate distances from all pixels to this palette color (GPU operation)
        # Using broadcasting to calculate distances efficiently
        dists = torch.sum((pixels_tensor - palette_color_tensor)**2, dim=1)
        
        # Find closest pixels
        _, indices = torch.topk(dists, k=10, largest=False)
        closest_indices = indices.cpu().numpy()
        
        # Take the average of these pixels for a refined color
        # Use GPU for the mean calculation
        refined_color = torch.mean(pixels_tensor[indices], dim=0).cpu().numpy()
        refined_palette.append(refined_color)
    
    # Ensure palette colors are well distributed in RGB space for better layer decomposition
    # Handle colors near RGB cube boundaries (as mentioned in the paper)
    for i, color in enumerate(refined_palette):
        # Check if color is very close to RGB boundary (values > 0.98 or < 0.02)
        for c in range(3):
            if color[c] > 0.98:
                refined_palette[i][c] = 0.98
            elif color[c] < 0.02:
                refined_palette[i][c] = 0.02
    
    return np.array(refined_palette)

def extract_palette_adaptive_gpu(image, max_palette_size=15):
    """
    Fully adaptive GPU-accelerated palette extraction algorithm.
    
    This algorithm:
    1. Automatically analyzes the image to determine its complexity
    2. Uses clustering and elbow method to determine optimal number of colors
    3. Adaptively refines the palette based on image characteristics
    4. Works well for any image from binary to complex photographs
    
    Args:
        image: RGB image in range [0, 1]
        max_palette_size: Hard upper limit on palette size (rarely reached)
    
    Returns:
        palette: Array of RGB colors
    """
    
    # Ensure we're using the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Reshape to list of pixels
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)
    total_pixels = len(pixels)
    
    # Move pixels to GPU for faster processing
    pixels_tensor = torch.tensor(pixels, dtype=torch.float32, device=device)
    
    # ----- STEP 1: Analyze image complexity -----
    
    print("Analyzing image complexity...")
    start_time = time.time()
    
    # Sample pixels for faster analysis (if image is large)
    max_samples = min(100000, total_pixels)
    if total_pixels > max_samples:
        indices = np.random.choice(total_pixels, max_samples, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
    
    # Compute color histogram
    hist_bins = 20
    hist = np.histogramdd(pixels_sample, bins=[hist_bins, hist_bins, hist_bins], 
                          range=[[0, 1], [0, 1], [0, 1]])[0]
    
    # Count non-empty bins
    non_empty_bins = np.sum(hist > 0)
    bin_entropy = -np.sum((hist / np.sum(hist)) * np.log2(hist / np.sum(hist) + 1e-10))
    
    print(f"Image analysis: {non_empty_bins} non-empty color bins, entropy: {bin_entropy:.2f}")
    
    # ----- STEP 2: Determine optimal palette size using elbow method -----
    
    print("Determining optimal palette size...")
    
    # Start with a binary check
    if non_empty_bins <= 4 and bin_entropy < 2.0:
        # Very simple image - just use K-means with k=non_empty_bins
        optimal_k = max(2, min(4, non_empty_bins))
        print(f"Simple image detected with {non_empty_bins} main colors. Using k={optimal_k}")
    else:
        # For complex images, use adaptive elbow method
        
        # Prepare for elbow method with K-means
        max_k_to_try = min(10, non_empty_bins, max_palette_size)
        inertias = []
        
        # Run K-means with different k values (on CPU, more efficient for small datasets)
        k_range = range(1, max_k_to_try + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
            kmeans.fit(pixels_sample)
            inertias.append(kmeans.inertia_)
        
        # Calculate acceleration (second derivative) to find elbow point
        deltas = np.diff(inertias)
        accelerations = np.diff(deltas)
        
        # Find the elbow point (where acceleration is highest)
        if len(accelerations) > 0:
            elbow_idx = np.argmax(np.abs(accelerations)) + 2  # +2 because of two diff operations
            optimal_k = k_range[elbow_idx]
        else:
            # Fallback if we don't have enough points
            optimal_k = 3
        
        print(f"Complex image detected. Optimal palette size determined: k={optimal_k}")
    
    # ----- STEP 3: Extract initial palette using K-means -----
    
    # Run K-means with the determined optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=1)
    kmeans.fit(pixels_sample)
    
    # Get the cluster centers as the initial palette
    initial_palette = kmeans.cluster_centers_
    
    # ----- STEP 4: Create a histogram-based color space binning for refinement -----
    
    # Create a histogram of colors by dividing RGB space into bins
    n_bins = 10  # 10x10x10 bins
    hist_edges = np.linspace(0, 1, n_bins+1)
    
    # Create 3D histogram (CPU operation)
    hist, edges = np.histogramdd(pixels, bins=[hist_edges, hist_edges, hist_edges])
    
    # Get centers of histogram bins
    bin_centers = []
    for i in range(3):
        centers = (edges[i][:-1] + edges[i][1:]) / 2
        bin_centers.append(centers)
    
    # Initialize palette with the initial K-means colors
    palette = initial_palette.tolist()
    votes = np.copy(hist)
    
    # Transfer bin indices and colors to GPU for faster distance calculations
    bin_indices = np.mgrid[0:n_bins, 0:n_bins, 0:n_bins].reshape(3, -1).T
    bin_colors = np.zeros((len(bin_indices), 3))
    
    # Prepare bin colors
    for i, (r_bin, g_bin, b_bin) in enumerate(bin_indices):
        bin_colors[i] = [bin_centers[0][r_bin], bin_centers[1][g_bin], bin_centers[2][b_bin]]
    
    bin_colors_tensor = torch.tensor(bin_colors, dtype=torch.float32, device=device)
    
    # ----- STEP 5: Refine palette by adaptively adding more colors if needed -----
    
    # Calculate how well current palette represents the image
    palette_tensor = torch.tensor(palette, dtype=torch.float32, device=device)
    
    # Use batching for large images to avoid GPU memory issues
    batch_size = min(200000, total_pixels)
    n_batches = (total_pixels + batch_size - 1) // batch_size
    
    all_min_distances = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_pixels)
        batch_pixels = pixels_tensor[start_idx:end_idx]
        
        # Calculate distances from each pixel to each palette color
        distances = torch.cdist(batch_pixels, palette_tensor, p=2)
        
        # Find minimum distance for each pixel
        min_distances, _ = torch.min(distances, dim=1)
        all_min_distances.append(min_distances)
    
    # Combine all min distances
    all_min_distances = torch.cat(all_min_distances)
    
    # Calculate statistics of distances
    mean_distance = all_min_distances.mean().item()
    percentile_75 = torch.quantile(all_min_distances, 0.75).item()
    
    # Adaptive distance threshold based on image statistics
    distance_threshold = min(0.1, max(0.02, mean_distance))
    
    print(f"Current palette size: {len(palette)}")
    print(f"Mean distance: {mean_distance:.4f}, 75th percentile: {percentile_75:.4f}")
    print(f"Adaptive distance threshold: {distance_threshold:.4f}")
    
    # Determine if we need more colors
    well_represented = torch.sum((all_min_distances < distance_threshold).float()).item()
    representation_percentage = well_represented / total_pixels
    
    print(f"Initial representation: {representation_percentage:.2%} of pixels well-represented")
    
    # Only continue adding colors if representation is poor and complexity is high
    if representation_percentage < 0.9 and bin_entropy > 3.0:
        print("Adding more colors to improve representation...")
        
        # Determine how many more colors to add
        additional_colors = min(max_palette_size - len(palette), 
                               int(np.ceil((1 - representation_percentage) * 10)))
        
        # Update votes based on current palette
        # Calculate distances between all bin colors and palette colors
        distances = torch.cdist(bin_colors_tensor, palette_tensor, p=2)
        
        # Find minimum distance for each bin color to any palette color
        min_distances, _ = torch.min(distances, dim=1)
        
        # Transfer min_distances back to CPU for vote updating
        min_distances_np = min_distances.cpu().numpy()
        
        # Update votes
        for i, (r_bin, g_bin, b_bin) in enumerate(bin_indices):
            votes[r_bin, g_bin, b_bin] *= min_distances_np[i]
        
        # Add more colors iteratively
        for _ in range(additional_colors):
            # Find bin with highest number of votes
            max_bin_idx = np.unravel_index(np.argmax(votes), hist.shape)
            
            # Get the RGB value for this bin center
            r = bin_centers[0][max_bin_idx[0]]
            g = bin_centers[1][max_bin_idx[1]]
            b = bin_centers[2][max_bin_idx[2]]
            
            # Add to palette
            palette.append([r, g, b])
            
            # Update palette tensor
            palette_tensor = torch.tensor(palette, dtype=torch.float32, device=device)
            
            # Update votes
            distances = torch.cdist(bin_colors_tensor, palette_tensor[-1].unsqueeze(0), p=2)
            distances_np = distances.squeeze(1).cpu().numpy()
            
            for i, (r_bin, g_bin, b_bin) in enumerate(bin_indices):
                votes[r_bin, g_bin, b_bin] *= distances_np[i]
            
            # Print progress
            print(f"Added color {len(palette)}: [{r:.2f}, {g:.2f}, {b:.2f}]")
    
    # ----- STEP 6: Refine palette colors by finding the actual most representative pixel in each region -----
    
    print("Refining palette colors...")
    refined_palette = []
    
    for palette_color in palette:
        palette_color_tensor = torch.tensor(palette_color, dtype=torch.float32, device=device)
        
        # Calculate distances from all pixels to this palette color (GPU operation)
        batch_size = min(200000, total_pixels)
        n_batches = (total_pixels + batch_size - 1) // batch_size
        
        all_dists = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_pixels)
            batch_pixels = pixels_tensor[start_idx:end_idx]
            
            dists = torch.sum((batch_pixels - palette_color_tensor)**2, dim=1)
            all_dists.append(dists)
        
        all_dists = torch.cat(all_dists)
        
        # Find closest pixels
        k = min(100, total_pixels)
        _, indices = torch.topk(all_dists, k=k, largest=False)
        
        # Take the average of these pixels for a refined color
        refined_color = torch.mean(pixels_tensor[indices], dim=0).cpu().numpy()
        refined_palette.append(refined_color)
    
    # ----- STEP 7: Handle RGB cube boundaries -----
    
    # Ensure palette colors are well distributed in RGB space
    for i, color in enumerate(refined_palette):
        # Check if color is very close to RGB boundary
        for c in range(3):
            if color[c] > 0.98:
                refined_palette[i][c] = 0.98
            elif color[c] < 0.02:
                refined_palette[i][c] = 0.02
    
    print(f"Final palette size: {len(refined_palette)} colors")
    print(f"Palette extraction completed in {time.time() - start_time:.2f} seconds")
    
    return np.array(refined_palette)

# ## Geometric Region Color Unmixing

def find_convex_hull(points):
    """
    Find the convex hull of a set of points.
    
    Args:
        points: Array of points
    
    Returns:
        hull_vertices: Indices of the convex hull vertices
        hull_simplices: Simplices (triangles) that make up the convex hull
    """
    hull = ConvexHull(points)
    return hull.vertices, hull.simplices

def in_hull(point, hull_vertices, hull_points):
    """
    Test if a point is inside a convex hull.
    
    Args:
        point: Point to test
        hull_vertices: Indices of the hull vertices
        hull_points: Points that make up the hull
    
    Returns:
        True if the point is inside the hull, False otherwise
    """
    hull_points_subset = hull_points[hull_vertices]
    try:
        hull = Delaunay(hull_points_subset)
        return hull.find_simplex(point) >= 0
    except:
        # Fallback if Delaunay fails
        return False

# Example of a more fully GPU-accelerated version (requires future optimization)
# Here's a simplified approach that approximates the geometric method
# but is much more GPU-friendly
def decompose_image_fast_gpu(image, palette, show_progress=True):
    """
    Fast GPU implementation that approximates the geometric region color unmixing algorithm.
    
    This version sacrifices some accuracy for much faster computation by using
    a simplified distance-based approach that runs entirely on the GPU.
    
    Args:
        image: RGB image
        palette: Color palette
        show_progress: Whether to show a progress bar
    
    Returns:
        layers: List of RGBA images
    """

    h, w = image.shape[:2]
    n_colors = palette.shape[0]
    n_pixels = h * w
    
    # Ensure we're working with the right device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Convert image and palette to PyTorch tensors
    image_tensor = torch.tensor(image.reshape(-1, 3), dtype=torch.float32, device=device)
    palette_tensor = torch.tensor(palette, dtype=torch.float32, device=device)
    
    # Initialize output layers
    layers = torch.zeros((n_colors, n_pixels, 4), dtype=torch.float32, device=device)
    for i in range(n_colors):
        layers[i, :, :3] = palette_tensor[i].unsqueeze(0).expand(n_pixels, -1)
    
    # Process image in large batches for better GPU utilization
    batch_size = 100000  # Much larger batch size for better GPU utilization
    n_batches = (n_pixels + batch_size - 1) // batch_size
    
    # Create a progress bar if requested
    batch_iterator = tqdm(range(n_batches), desc="Decomposing image") if show_progress else range(n_batches)
    
    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_pixels)
        batch_pixels = image_tensor[start_idx:end_idx]
        
        # Compute the distance of each pixel to each palette color
        # This is a fully vectorized operation that runs entirely on the GPU
        # Shape: [batch_size, n_colors]
        distances = torch.cdist(batch_pixels, palette_tensor, p=2)
        
        # Convert distances to weights using inverse distance weighting
        # Adding small epsilon to avoid division by zero
        weights = 1.0 / (distances + 1e-6)
        
        # Apply power parameter to emphasize closer colors (similar to barycentric coordinates)
        power = 2.0  # Higher values make the effect more pronounced
        weights = weights ** power
        
        # Normalize weights to sum to 1 for each pixel
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        normalized_weights = weights / weights_sum
        
        # Update the alpha channels for all layers
        for i in range(n_colors):
            layers[i, start_idx:end_idx, 3] = normalized_weights[:, i]
    
    # Reshape layers to image format
    output_layers = []
    for i in range(n_colors):
        layer = layers[i].reshape(h, w, 4).cpu().numpy()
        output_layers.append(layer)
    
    return output_layers


def decompose_image_geometric_regions(image, palette):
    """
    Decompose an image into layers using the geometric region color unmixing approach
    as described in "Image Decomposition using Geometric Region Colour Unmixing".
    
    The algorithm:
    1. Splits the RGB space into different regions based on the palette colors
    2. For each pixel, determines which region it belongs to
    3. Applies the appropriate geometric unmixing method depending on the region
    
    Args:
        image: RGB image
        palette: Color palette
    
    Returns:
        layers: List of RGBA images
    """
    h, w = image.shape[:2]
    n_colors = palette.shape[0]
    
    # Convert image and palette to PyTorch tensors for faster processing
    image_tensor = torch.tensor(image.reshape(-1, 3), dtype=torch.float32, device=device)
    palette_tensor = torch.tensor(palette, dtype=torch.float32, device=device)
    
    # Initialize output layers
    layers = torch.zeros((n_colors, h*w, 4), dtype=torch.float32, device=device)
    for i in range(n_colors):
        layers[i, :, :3] = palette_tensor[i].unsqueeze(0).expand(h*w, -1)
    
    # Process pixels in batches for memory efficiency
    batch_size = 10000
    n_batches = (h*w + batch_size - 1) // batch_size
    
    # Find convex hull of palette colors for region determination
    palette_np = palette_tensor.cpu().numpy()
    hull = ConvexHull(palette_np)
    
    # Create Delaunay triangulation for inside-hull points
    try:
        tri = Delaunay(palette_np)
    except:
        # Fallback if Delaunay fails (e.g., for coplanar points)
        # Add small random noise to palette colors to avoid coplanarity
        palette_np_noisy = palette_np + np.random.normal(0, 0.001, palette_np.shape)
        tri = Delaunay(palette_np_noisy)

    batch_iterator = tqdm(range(n_batches), desc="Decomposing image")

    
    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, h*w)
        batch_pixels = image_tensor[start_idx:end_idx].cpu().numpy()
        
        # Calculate alphas for each pixel in the batch
        batch_alphas = np.zeros((end_idx - start_idx, n_colors))
        
        for i in range(len(batch_pixels)):
            pixel = batch_pixels[i]
            
            # Step 1: Determine which region the pixel belongs to
            simplex_index = tri.find_simplex(pixel)
            
            if simplex_index >= 0:
                # Inside convex hull: use barycentric coordinates for unmixing
                simplex_vertices = tri.simplices[simplex_index]
                simplex_points = palette_np[simplex_vertices]
                
                # Calculate barycentric coordinates
                # This is more accurate than the distance-based approximation
                b = tri.transform[simplex_index, :3].dot(pixel - tri.transform[simplex_index, 3])
                barycentric = np.zeros(len(simplex_vertices))
                barycentric[:-1] = b
                barycentric[-1] = 1 - np.sum(b)
                
                # Assign weights based on barycentric coordinates
                for j, vertex_idx in enumerate(simplex_vertices):
                    batch_alphas[i, vertex_idx] = max(0, barycentric[j])  # Ensure non-negative
            else:
                # Outside convex hull: project to nearest face or edge
                # Find the closest facet of the convex hull
                min_dist = float('inf')
                closest_facet = None
                
                for facet in hull.simplices:
                    facet_points = palette_np[facet]
                    # Create a plane from the facet
                    v1 = facet_points[1] - facet_points[0]
                    v2 = facet_points[2] - facet_points[0] if len(facet_points) > 2 else np.cross(v1, np.array([1, 0, 0]))
                    normal = np.cross(v1, v2)
                    normal = normal / np.linalg.norm(normal)
                    
                    # Calculate distance to plane
                    d = np.abs(np.dot(normal, pixel - facet_points[0]))
                    if d < min_dist:
                        min_dist = d
                        closest_facet = facet
                
                if closest_facet is not None:
                    # Project pixel to the closest facet
                    facet_points = palette_np[closest_facet]
                    
                    # Special handling for RGB cube boundaries
                    # Check if any palette color is close to RGB cube boundary
                    boundary_case = False
                    for color in facet_points:
                        if np.any(color > 0.98) or np.any(color < 0.02):
                            boundary_case = True
                            break
                    
                    if boundary_case:
                        # Special handling for RGB cube boundaries as mentioned in the paper
                        # For simplicity, we'll use a weighted average based on distances
                        dists = np.sum((facet_points - pixel)**2, axis=1)
                        weights = 1.0 / (dists + 1e-10)
                        weights = weights / np.sum(weights)
                        
                        for j, idx in enumerate(closest_facet):
                            batch_alphas[i, idx] = weights[j]
                    else:
                        # Calculate barycentric coordinates for the projection
                        # Simplified for implementation - using weighted distances
                        dists = np.sum((facet_points - pixel)**2, axis=1)
                        weights = 1.0 / (dists + 1e-10)
                        weights = weights / np.sum(weights)
                        
                        for j, idx in enumerate(closest_facet):
                            batch_alphas[i, idx] = weights[j]
                else:
                    # Fallback: use nearest palette color
                    dists = np.sum((palette_np - pixel)**2, axis=1)
                    nearest_idx = np.argmin(dists)
                    batch_alphas[i, nearest_idx] = 1.0
        
        # Normalize alphas to ensure they sum to 1
        row_sums = np.sum(batch_alphas, axis=1, keepdims=True)
        batch_alphas = np.divide(batch_alphas, row_sums, out=np.zeros_like(batch_alphas), where=row_sums!=0)
        
        # Update the alpha channels for all layers
        for i in range(n_colors):
            layers[i, start_idx:end_idx, 3] = torch.tensor(batch_alphas[:, i], device=device)
    
    # Reshape layers to image format
    output_layers = []
    for i in range(n_colors):
        layer = layers[i].reshape(h, w, 4).cpu().numpy()
        output_layers.append(layer)
    
    return output_layers

def apply_alpha_smoothing(layers, sigma=1.0):
    """
    Apply Gaussian smoothing to the alpha channels of all layers.
    
    Args:
        layers: List of RGBA images
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        smoothed_layers: List of RGBA images with smoothed alpha channels
    """
    smoothed_layers = []
    
    for layer in layers:
        smoothed_layer = layer.copy()
        # Apply Gaussian smoothing only to the alpha channel
        smoothed_layer[:, :, 3] = ndimage.gaussian_filter(layer[:, :, 3], sigma=sigma)
        smoothed_layers.append(smoothed_layer)
    
    return smoothed_layers

def refine_layer_colors(image, layers):
    """
    Refine the colors of each layer based on the algorithm in the paper.
    
    Args:
        image: Original RGB image
        layers: List of RGBA layers
    
    Returns:
        refined_layers: List of RGBA layers with refined colors
    """
    h, w = image.shape[:2]
    n_layers = len(layers)
    
    # Reshape image and layers
    image_flat = image.reshape(-1, 3)
    alphas = np.array([layer[:, :, 3].reshape(-1) for layer in layers]).T  # shape: (h*w, n_layers)
    
    # Convert to PyTorch tensors
    image_tensor = torch.tensor(image_flat, dtype=torch.float32, device=device)
    alphas_tensor = torch.tensor(alphas, dtype=torch.float32, device=device)
    
    # Get current layer colors
    layer_colors = torch.tensor(np.array([layer[0, 0, :3] for layer in layers]), 
                               dtype=torch.float32, device=device)
    
    # Iterative refinement (Algorithm 1 in the paper)
    for iter in range(5):  # Number of iterations
        for i in range(n_layers):
            # Skip layers with very small contribution
            if torch.sum(alphas_tensor[:, i]) < 1e-6:
                continue
            
            # Calculate the residual color
            other_contributions = torch.zeros_like(image_tensor)
            for j in range(n_layers):
                if j != i:
                    other_contributions += alphas_tensor[:, j].unsqueeze(1) * layer_colors[j].unsqueeze(0)
            
            residual = image_tensor - other_contributions
            
            # Compute new layer color
            weights = alphas_tensor[:, i]
            masked_residual = residual * weights.unsqueeze(1)
            new_color = torch.sum(masked_residual, dim=0) / (torch.sum(weights) + 1e-10)
            
            # Constrain the new color to be not too different from the original
            max_change = 0.2
            color_diff = new_color - layer_colors[i]
            if torch.norm(color_diff) > max_change:
                color_diff = color_diff * max_change / torch.norm(color_diff)
            
            layer_colors[i] = layer_colors[i] + color_diff
            
            # Ensure color values are in valid range
            layer_colors[i] = torch.clamp(layer_colors[i], 0.0, 1.0)
    
    # Update the layers with refined colors
    refined_layers = []
    for i in range(n_layers):
        refined_layer = layers[i].copy()
        refined_layer[:, :, :3] = layer_colors[i].cpu().numpy()
        refined_layers.append(refined_layer)
    
    return refined_layers

def save_layers_as_png(layers, output_dir):
    """
    Save each layer as a PNG file with alpha channel.
    
    Args:
        layers: List of RGBA images
        output_dir: Directory to save the PNG files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, layer in enumerate(layers):
        # Convert to 8-bit
        layer_8bit = (layer * 255).astype(np.uint8)
        
        # Save as PNG with alpha channel
        image = Image.fromarray(layer_8bit, mode='RGBA')
        image.save(os.path.join(output_dir, f'layer_{i+1}.png'))

def save_palette(palette, output_path):
    """
    Save a color palette as a PNG file.
    
    Args:
        palette: Array of RGB colors
        output_path: Path to save the PNG file
    """
    n_colors = palette.shape[0]
    palette_image = np.zeros((50, n_colors * 50, 3))
    
    for i in range(n_colors):
        palette_image[:, i*50:(i+1)*50, :] = palette[i]
    
    # Convert to 8-bit
    palette_image = (palette_image * 255).astype(np.uint8)
    
    # Save as PNG
    image = Image.fromarray(palette_image)
    image.save(output_path)

# ## Main Function

def main(image_path, n_colors=5, output_dir='output'):
    """
    Main function to decompose an image into layers using the
    "Image Decomposition using Geometric Region Colour Unmixing" algorithm.
    
    Args:
        image_path: Path to the input image
        n_colors: Number of colors in the palette
        output_dir: Directory to save the output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    
    # Extract the palette
    print(f"Extracting palette with {n_colors} colors...")
    start_time = time.time()
    palette = extract_palette_gpu(image, n_colors=n_colors)
    # palette = extract_palette_adaptive_gpu(image)

    print(f"Palette extraction took {time.time() - start_time:.2f} seconds")
    
    # Visualize the palette
    visualize_palette(palette)
    
    # Save the palette
    save_palette(palette, os.path.join(output_dir, 'palette.png'))
    
    # Plot RGB space
    pixels = image.reshape(-1, 3)
    plot_3d_rgb(pixels, palette, subsample=5000)
    
    # Decompose the image into layers using geometric region color unmixing
    print("Decomposing image into layers...")
    start_time = time.time()
    # layers = decompose_image_geometric_regions(image, palette)
    layers = decompose_image_fast_gpu(image, palette)

    
    print(f"Decomposition took {time.time() - start_time:.2f} seconds")
    
    # Smooth the alpha channels
    print("Smoothing alpha channels...")
    smoothed_layers = apply_alpha_smoothing(layers, sigma=1.0)
    
    # Refine layer colors
    print("Refining layer colors...")
    refined_layers = refine_layer_colors(image, smoothed_layers)
    
    # Visualize the layers
    # visualize_layers(layers)
    
    # Save the layers
    print(f"Saving layers to {output_dir}...")
    save_layers_as_png(layers, output_dir)
    
    print("Done!")
    
    return palette, refined_layers

# ## Usage Example

# Replace with your image path
image_path = "testing.png"

# Run the main function
palette, layers = main(image_path, n_colors=5, output_dir='output')