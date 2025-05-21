import numpy as np
import torch
from PIL import Image
import os
import argparse
import re
from torch.nn import functional as F
import matplotlib.pyplot as plt

def load_png_layers(folder_path):
    """
    Load all PNG images from a folder as PyTorch tensors with their alpha channels.
    Returns them sorted by layer number (layer_0, layer_1, etc.)
    """
    layers = []
    filenames = []
    
    # Get all PNG files in the directory
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # Extract layer numbers and sort by them
    layer_pattern = re.compile(r'layer_(\d+)\.png', re.IGNORECASE)
    sorted_files = []
    
    for filename in all_files:
        match = layer_pattern.match(filename)
        if match:
            layer_num = int(match.group(1))
            sorted_files.append((layer_num, filename))
    
    # Sort by layer number
    sorted_files.sort()
    filenames = [f[1] for f in sorted_files]
    
    # Load each layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for filename in filenames:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        
        # Convert to RGBA
        if img.mode == 'P':
            if 'transparency' in img.info:
                img = img.convert('RGBA')
            else:
                img = img.convert('RGBA')
        elif img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        # Convert PIL image to tensor and move to GPU
        img_tensor = torch.from_numpy(np.array(img)).to(device)
        layers.append(img_tensor)
        print(f"Loaded layer: {filename}")
    
    return layers, filenames, device

def generate_void_cluster_pattern(height, width, device):
    """
    Generate a void-and-cluster dithering pattern.
    Returns a normalized pattern where values range from 0 to 1.
    """
    print("Generating void-and-cluster pattern...")
    
    # Initialize with random pattern at 50% density
    binary = torch.rand(height, width, device=device)
    binary = (binary < 0.5).float()
    
    # Parameters for void and cluster
    sigma = 1.5  # Filter standard deviation
    
    # Create Gaussian kernel for filtering
    kernel_size = 9
    x = torch.linspace(-sigma * 3, sigma * 3, kernel_size, device=device)
    kernel_1d = torch.exp(-x.pow(2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Reshape for separable convolution
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
    
    # Perform initial relaxation phase
    for i in range(20):
        if i % 5 == 0:
            print(f"Initial relaxation: {i+1}/20", end="\r")
            
        # Apply Gaussian filter
        padded = F.pad(binary.unsqueeze(0).unsqueeze(0), 
                       (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                       mode='reflect')
        
        blurred = F.conv2d(padded, kernel_x, padding=0)
        blurred = F.conv2d(blurred, kernel_y, padding=0)
        blurred = blurred.squeeze()
        
        # Find tightest cluster and largest void
        white_pixels = binary > 0.5
        black_pixels = ~white_pixels
        
        if torch.sum(white_pixels) > 0:
            # Find maximum cluster (among white pixels)
            masked_cluster = blurred.clone()
            masked_cluster[~white_pixels] = -float('inf')
            max_idx = torch.argmax(masked_cluster.view(-1))
            max_y, max_x = max_idx // width, max_idx % width
            
            # Find minimum void (among black pixels)
            masked_void = blurred.clone()
            masked_void[~black_pixels] = float('inf')
            min_idx = torch.argmin(masked_void.view(-1))
            min_y, min_x = min_idx // width, min_idx % width
            
            # Swap these pixels to improve the pattern
            binary[max_y, max_x] = 0
            binary[min_y, min_x] = 1
    
    # Generate the complete ordering/ranking of all pixels
    print("\nGenerating pixel ranking...")
    total_pixels = height * width
    rank_order = torch.zeros(total_pixels, dtype=torch.int32, device=device)
    
    # Create a binary pattern for ranking
    rank_binary = binary.clone()
    white_count = torch.sum(rank_binary).item()
    
    # First, rank all white pixels by cluster tightness
    for rank in range(int(white_count) - 1, -1, -1):
        if rank % 1000 == 0:
            print(f"Ranking white pixels: {white_count-rank}/{white_count}", end="\r")
            
        # Apply Gaussian filter
        padded = F.pad(rank_binary.unsqueeze(0).unsqueeze(0), 
                       (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                       mode='reflect')
        
        blurred = F.conv2d(padded, kernel_x, padding=0)
        blurred = F.conv2d(blurred, kernel_y, padding=0)
        blurred = blurred.squeeze()
        
        # Find max only among white pixels
        masked_measure = blurred.clone()
        masked_measure[rank_binary < 0.5] = -float('inf')
        max_idx = torch.argmax(masked_measure.view(-1))
        
        # Set the rank and remove this pixel
        rank_order[max_idx] = rank
        rank_binary.view(-1)[max_idx] = 0
    
    print("\nRanking black pixels...")
    
    # Then, rank all black pixels by void size
    for rank in range(int(white_count), total_pixels):
        if rank % 1000 == 0:
            print(f"Ranking black pixels: {rank-white_count}/{total_pixels-white_count}", end="\r")
            
        # Apply Gaussian filter
        padded = F.pad(rank_binary.unsqueeze(0).unsqueeze(0), 
                       (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                       mode='reflect')
        
        blurred = F.conv2d(padded, kernel_x, padding=0)
        blurred = F.conv2d(blurred, kernel_y, padding=0)
        blurred = blurred.squeeze()
        
        # Find min only among black pixels
        masked_measure = blurred.clone()
        masked_measure[rank_binary > 0.5] = float('inf')
        min_idx = torch.argmin(masked_measure.view(-1))
        
        # Set the rank and add this pixel
        rank_order[min_idx] = rank
        rank_binary.view(-1)[min_idx] = 1
    
    # Reshape and normalize ranks to range [0, 1]
    normalized_ranks = rank_order.float() / (total_pixels - 1)
    normalized_ranks = normalized_ranks.view(height, width)
    
    print("\nVoid-and-cluster pattern generation complete.")
    return normalized_ranks

def create_color_mixing_dither(layers, output_path, device, alpha_min=0.05):
    """
    Create a dithered image that mixes colors from overlapping layers based on their alpha values.
    
    Parameters:
    - layers: List of layer tensors in RGBA format
    - output_path: Path to save the output image
    - device: Torch device
    - alpha_min: Minimum alpha value to consider (0.0-1.0)
    """
    if not layers:
        raise ValueError("No layers provided")
    
    # Get image dimensions from the first layer
    height, width = layers[0].shape[:2]
    
    # Generate the void-and-cluster pattern
    pattern = generate_void_cluster_pattern(height, width, device)
    
    # Extract alpha and RGB information from each layer
    alpha_channels = []
    rgb_channels = []
    
    for layer in layers:
        rgb = layer[:, :, :3].float()  # Convert to float for better precision
        alpha = layer[:, :, 3].float() / 255.0  # Normalize to [0,1]
        
        # Apply threshold to filter very low alpha values
        alpha = torch.where(alpha >= alpha_min, alpha, torch.zeros_like(alpha))
        
        alpha_channels.append(alpha)
        rgb_channels.append(rgb)
    
    # Create output image
    output = torch.zeros((height, width, 4), dtype=torch.uint8, device=device)
    
    # For each pixel location, determine which layers contribute and how much
    print("Computing layer contributions...")
    
    # Stack alpha channels for easier processing
    stacked_alpha = torch.stack(alpha_channels)  # Shape: [num_layers, height, width]
    
    # At each pixel, calculate total alpha and relative contributions
    total_alpha = torch.sum(stacked_alpha, dim=0)
    total_alpha = torch.clamp(total_alpha, min=1e-5)  # Avoid division by zero
    
    # Calculate the normalized contribution of each layer at each pixel
    layer_weights = []
    for alpha in alpha_channels:
        weight = alpha / total_alpha
        layer_weights.append(weight)
    
    # Create per-pixel allocation based on pattern value and layer weights
    print("Creating color mixing pattern...")
    pixel_assignments = torch.zeros((height, width), dtype=torch.int64, device=device) - 1
    
    # Determine how much of the pattern range each layer gets at each pixel
    cumulative_thresholds = torch.zeros((len(layers) + 1, height, width), device=device)
    for i in range(1, len(layers) + 1):
        cumulative_thresholds[i] = cumulative_thresholds[i-1] + layer_weights[i-1]
    
    # Assign each pixel to a layer based on pattern value and thresholds
    for i in range(len(layers)):
        lower = cumulative_thresholds[i]
        upper = cumulative_thresholds[i+1]
        mask = (pattern >= lower) & (pattern < upper)
        pixel_assignments = torch.where(mask, torch.tensor(i, device=device), pixel_assignments)
    
    # Apply the pixel assignments to generate the final image
    print("Generating final dithered image with color mixing...")
    has_content = torch.sum(stacked_alpha, dim=0) > 0
    
    for i in range(len(layers)):
        layer_mask = (pixel_assignments == i)
        effective_mask = layer_mask & has_content
        
        if torch.any(effective_mask):
            for c in range(3):  # RGB channels
                output[:, :, c] = torch.where(effective_mask, rgb_channels[i][:, :, c].byte(), output[:, :, c])
            
            # Set alpha to fully opaque where we placed pixels
            output[:, :, 3] = torch.where(effective_mask, 
                                         torch.tensor(255, dtype=torch.uint8, device=device),
                                         output[:, :, 3])
    
    # Create visualizations to verify the mixing
    print("Creating visualization of the mixing process...")
    
    # Visualize layer contributions
    for i in range(min(3, len(layers))):
        weight_vis_path = os.path.splitext(output_path)[0] + f'_layer{i}_weight.png'
        weight_vis = (layer_weights[i] * 255).byte().cpu().numpy()
        Image.fromarray(weight_vis).save(weight_vis_path)
        print(f"Saved layer {i} weight visualization to {weight_vis_path}")
    
    # Visualize pixel assignments
    assignment_vis_path = os.path.splitext(output_path)[0] + '_assignments.png'
    # Create a colorful visualization using different colors for each layer
    assignment_vis = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)
    
    for i in range(len(layers)):
        layer_mask = (pixel_assignments == i)
        if i % 3 == 0:  # Red-ish
            assignment_vis[layer_mask, 0] = 180 + i * 20
            assignment_vis[layer_mask, 1] = 50 + i * 10
            assignment_vis[layer_mask, 2] = 50 + i * 10
        elif i % 3 == 1:  # Green-ish
            assignment_vis[layer_mask, 0] = 50 + i * 10
            assignment_vis[layer_mask, 1] = 180 + i * 20
            assignment_vis[layer_mask, 2] = 50 + i * 10
        else:  # Blue-ish
            assignment_vis[layer_mask, 0] = 50 + i * 10
            assignment_vis[layer_mask, 1] = 50 + i * 10
            assignment_vis[layer_mask, 2] = 180 + i * 20
    
    assignment_vis_cpu = assignment_vis.cpu().numpy()
    Image.fromarray(assignment_vis_cpu).save(assignment_vis_path)
    print(f"Saved pixel assignment visualization to {assignment_vis_path}")
    
    # Move the result back to CPU for saving
    output_cpu = output.cpu().numpy()
    
    # Save the output image
    print(f"Saving color-mixed dithered image to {output_path}...")
    Image.fromarray(output_cpu).save(output_path)
    print("Processing complete!")
    return output

def create_advanced_mixing_dither(layers, output_path, device, alpha_min=0.05, num_patterns=4):
    """
    Create a sophisticated dithered image that mixes colors from overlapping layers
    using multiple dithering patterns for a more natural look.
    
    Parameters:
    - layers: List of layer tensors in RGBA format
    - output_path: Path to save the output image
    - device: Torch device
    - alpha_min: Minimum alpha value to consider (0.0-1.0)
    - num_patterns: Number of different dithering patterns to use
    """
    if not layers:
        raise ValueError("No layers provided")
    
    # Get image dimensions from the first layer
    height, width = layers[0].shape[:2]
    
    # Generate dithering patterns
    print(f"Generating {num_patterns} distinct dithering patterns...")
    patterns = []
    for i in range(num_patterns):
        pattern = generate_void_cluster_pattern(height, width, device)
        patterns.append(pattern)
    
    # Create output image
    output = torch.zeros((height, width, 4), dtype=torch.uint8, device=device)
    
    # Extract alpha and RGB information from each layer
    alpha_channels = []
    rgb_channels = []
    
    for layer in layers:
        rgb = layer[:, :, :3].float()
        alpha = layer[:, :, 3].float() / 255.0  # Normalize to [0,1]
        
        # Apply threshold to filter very low alpha values
        alpha = torch.where(alpha >= alpha_min, alpha, torch.zeros_like(alpha))
        
        alpha_channels.append(alpha)
        rgb_channels.append(rgb)
    
    # Stack alpha channels for easier processing
    stacked_alpha = torch.stack(alpha_channels)  # Shape: [num_layers, height, width]
    
    # Identify content pixels (any layer has significant alpha)
    content_mask = torch.max(stacked_alpha, dim=0)[0] > 0
    
    # Calculate the total alpha at each pixel
    total_alpha = torch.sum(stacked_alpha, dim=0)
    
    # Calculate normalized weights for each layer
    # Where total_alpha is 0, we'll handle separately
    layer_weights = []
    for alpha in alpha_channels:
        # For pixels with content, calculate weight as alpha / total_alpha
        # For pixels without content, weight is 0
        weight = torch.zeros_like(alpha)
        weight = torch.where(content_mask, alpha / total_alpha.clamp(min=1e-5), weight)
        layer_weights.append(weight)
    
    # Initialize pixel assignments to an invalid value (-1)
    pixel_assignments = torch.full((height, width), -1, dtype=torch.int64, device=device)
    
    print("Creating dithering pattern with proportional color mixing...")
    
    # Apply each pattern to a portion of the image
    for pattern_idx, pattern in enumerate(patterns):
        # Use the first pattern to partition the image into regions
        selector_pattern = patterns[0]
        
        # Define the region for this pattern
        lower_bound = pattern_idx / num_patterns
        upper_bound = (pattern_idx + 1) / num_patterns
        
        # Only consider content pixels in this region
        region_mask = (selector_pattern >= lower_bound) & (selector_pattern < upper_bound) & content_mask
        
        if not torch.any(region_mask):
            continue
        
        # For each layer, calculate its allocation in the pattern
        cumulative_thresholds = torch.zeros((len(layers) + 1, height, width), device=device)
        
        for i in range(1, len(layers) + 1):
            # Add this layer's weight to the cumulative threshold
            cumulative_thresholds[i] = cumulative_thresholds[i-1] + layer_weights[i-1]
        
        # Ensure the final threshold is exactly 1.0 for all content pixels
        # This prevents gaps due to floating point errors
        cumulative_thresholds[-1] = torch.where(
            content_mask,
            torch.ones_like(cumulative_thresholds[-1]),
            cumulative_thresholds[-1]
        )
        
        # For each layer, assign pixels based on thresholds
        for i in range(len(layers)):
            lower_threshold = cumulative_thresholds[i]
            upper_threshold = cumulative_thresholds[i+1]
            
            # A pixel belongs to this layer if:
            # 1. It's in this pattern's region
            # 2. The pattern value falls between this layer's thresholds
            # 3. This layer has non-zero alpha at this pixel
            layer_mask = (
                region_mask & 
                (pattern >= lower_threshold) & 
                (pattern < upper_threshold) & 
                (alpha_channels[i] > 0)
            )
            
            # Assign matching pixels to this layer
            pixel_assignments = torch.where(layer_mask, torch.tensor(i, device=device), pixel_assignments)
    
    # Handle any unassigned content pixels
    unassigned_mask = (pixel_assignments == -1) & content_mask
    if torch.any(unassigned_mask):
        print(f"Fixing {torch.sum(unassigned_mask).item()} unassigned content pixels...")
        
        # Find the layer with highest alpha for each unassigned pixel
        max_alpha_values, max_alpha_indices = torch.max(stacked_alpha, dim=0)
        
        # Assign to the layer with highest alpha
        pixel_assignments = torch.where(unassigned_mask & (max_alpha_values > 0), 
                                       max_alpha_indices, 
                                       pixel_assignments)
    
    # Final check: there should be no unassigned content pixels
    still_unassigned = (pixel_assignments == -1) & content_mask
    if torch.any(still_unassigned):
        print(f"CRITICAL: Still have {torch.sum(still_unassigned).item()} unassigned content pixels")
        
        # For any remaining unassigned pixels, just assign to layer 0 as fallback
        pixel_assignments = torch.where(still_unassigned, 
                                       torch.tensor(0, device=device), 
                                       pixel_assignments)
    
    # Apply the pixel assignments to generate the final image
    print("Generating final dithered image with color mixing...")
    
    # For each layer, add its colors to the output based on pixel assignments
    for i in range(len(layers)):
        layer_mask = (pixel_assignments == i)
        
        if torch.any(layer_mask):
            for c in range(3):  # RGB channels
                output[:, :, c] = torch.where(layer_mask, rgb_channels[i][:, :, c].byte(), output[:, :, c])
            
            # Set alpha to fully opaque where we placed pixels
            output[:, :, 3] = torch.where(layer_mask, 
                                         torch.tensor(255, dtype=torch.uint8, device=device),
                                         output[:, :, 3])
    
    # Final verification: ensure all content pixels are assigned and visible
    missing_mask = (output[:, :, 3] == 0) & content_mask
    if torch.any(missing_mask):
        print(f"ERROR: Found {torch.sum(missing_mask).item()} content pixels still not covered in output")
        # Make them visible with a bright color for debugging
        for y, x in zip(*torch.where(missing_mask)):
            output[y, x, 0] = 255  # Red
            output[y, x, 1] = 0
            output[y, x, 2] = 0
            output[y, x, 3] = 255
    else:
        print("SUCCESS: All content pixels covered successfully.")
    
    # Count pixels assigned to each layer for verification
    layer_counts = []
    for i in range(len(layers)):
        count = torch.sum(pixel_assignments == i).item()
        layer_counts.append(count)
        print(f"Layer {i}: {count} pixels")
    
    # Move the result back to CPU for saving
    output_cpu = output.cpu().numpy()
    
    # Save the output image
    print(f"Saving color-mixed dithered image to {output_path}...")
    Image.fromarray(output_cpu).save(output_path)
    
    # Create a visualization of the pixel assignments
    assignment_vis_path = os.path.splitext(output_path)[0] + '_assignments.png'
    assignment_vis = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)
    
    # Define colors for each layer
    colors = [
        [255, 50, 50],    # Red for layer 0
        [50, 255, 50],    # Green for layer 1
        [50, 50, 255],    # Blue for layer 2
        [255, 255, 50],   # Yellow for layer 3
        [50, 255, 255],   # Cyan for layer 4
        [255, 50, 255],   # Magenta for layer 5
        [255, 128, 50],   # Orange for layer 6
        [128, 50, 255],   # Purple for layer 7
    ]
    
    # Assign colors to each layer - FIX THE INDEXING HERE
    for i in range(min(len(layers), len(colors))):
        layer_mask = (pixel_assignments == i)
        if torch.any(layer_mask):
            # Get the color for this layer
            r, g, b = colors[i % len(colors)]
            
            # Apply the color to the visualization
            mask_indices = torch.nonzero(layer_mask, as_tuple=True)
            assignment_vis[mask_indices[0], mask_indices[1], 0] = r
            assignment_vis[mask_indices[0], mask_indices[1], 1] = g
            assignment_vis[mask_indices[0], mask_indices[1], 2] = b
    
    # Any unassigned content pixels (should be none) are bright pink
    unassigned_content = (pixel_assignments == -1) & content_mask
    if torch.any(unassigned_content):
        mask_indices = torch.nonzero(unassigned_content, as_tuple=True)
        assignment_vis[mask_indices[0], mask_indices[1], 0] = 255  # Pink
        assignment_vis[mask_indices[0], mask_indices[1], 1] = 0
        assignment_vis[mask_indices[0], mask_indices[1], 2] = 255
    
    assignment_vis_cpu = assignment_vis.cpu().numpy()
    Image.fromarray(assignment_vis_cpu).save(assignment_vis_path)
    print(f"Saved pixel assignment visualization to {assignment_vis_path}")
    
    print("Processing complete!")
    return output


def main():
    parser = argparse.ArgumentParser(description='Create a dithered image with color mixing from PNG layers using GPU acceleration.')
    parser.add_argument('folder', help='Folder containing PNG layers (named layer_0.png, layer_1.png, etc.)')
    parser.add_argument('--output', default='dithered_output.png', help='Output image path')
    parser.add_argument('--alpha_min', type=float, default=0.1, 
                       help='Minimum alpha value to consider (0.0-1.0)')
    parser.add_argument('--patterns', type=int, default=4,
                       help='Number of patterns for advanced mode')
    args = parser.parse_args()
    
    # Load all PNG layers from the folder
    layers, filenames, device = load_png_layers(args.folder)
    print(f"Loaded {len(layers)} layers: {', '.join(filenames)}")
   
    output = create_advanced_mixing_dither(
        layers, 
        args.output, 
        device, 
        alpha_min=args.alpha_min,
        num_patterns=args.patterns
    )

    print("All processing complete!")

if __name__ == "__main__":
    main()