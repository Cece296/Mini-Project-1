import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from scipy.spatial.distance import mahalanobis

def estimate_covariance(samples):
    """Estimates the covariance matrix from a set of sample pixels."""
    cov_matrix = np.cov(samples, rowvar=False)
    return np.linalg.inv(cov_matrix)  # Compute the inverse covariance matrix

def calculate_mahalanobis_distance(image, mean_rgb, cov_inv):
    """Calculates the Mahalanobis distance for each pixel in the image."""
    diff = image.astype(np.float32) - mean_rgb
    distances = np.sqrt(np.einsum('ijk,kl,ijl->ij', diff, cov_inv, diff))
    return distances

def segment_pumpkins(image, mean_rgb, cov_inv, threshold=7.0):
    """Segments pumpkins using Mahalanobis distance for color classification."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color space
    distances = calculate_mahalanobis_distance(image, mean_rgb, cov_inv)
    mask = (distances < threshold).astype(np.uint8) * 255  # Convert to binary mask
    
    # Apply morphological closing to fill gaps
    kernel = np.ones((5,5), np.uint8)  # Adjust kernel size if needed
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask_filled

def split_large_box(x, y, w, h, max_width, max_height):
    """Splits a large bounding box into smaller boxes based on width and height limits."""
    boxes = []
    for i in range(0, w, max_width):
        for j in range(0, h, max_height):
            x1 = x + i
            y1 = y + j
            x2 = min(x + i + max_width, x + w)
            y2 = min(y + j + max_height, y + h)
            boxes.append([x1, y1, x2, y2])
    return boxes

def count_pumpkins(mask, max_width=25, max_height=25, min_width=3, min_height=3):
    """Counts pumpkins using connected component analysis and returns bounding boxes restricted by width and height."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        if w < min_width or h < min_height:  # Skip small objects
            continue
        if w > max_width or h > max_height:
            boxes.extend(split_large_box(x, y, w, h, max_width, max_height))  # Split oversized objects
        else:
            boxes.append([x, y, x + w, y + h])
    return boxes

def process_orthomosaic(image_path, mean_rgb, cov_inv, tile_size=512, overlap=128):
    """Processes an orthomosaic by dividing it into tiles and counting pumpkins, displaying final results and saving the image."""
    pumpkin_count = 0
    
    with rasterio.open(image_path) as dataset:
        width, height = dataset.width, dataset.height
        full_image = dataset.read([1, 2, 3]).transpose(1, 2, 0)  # Read entire image
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)  # Ensure correct color space
        mask_image = np.zeros((height, width), dtype=np.uint8)
        output_image = full_image.copy()
        all_boxes = []
        
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                window = Window(x, y, tile_size, tile_size)
                image = dataset.read([1, 2, 3], window=window).transpose(1, 2, 0)  # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                mask = segment_pumpkins(image, mean_rgb, cov_inv)
                
                mask_image[y:y+tile_size, x:x+tile_size] = np.maximum(mask_image[y:y+tile_size, x:x+tile_size], mask[:tile_size, :tile_size])
                
                boxes = count_pumpkins(mask)
                for box in boxes:
                    x1, y1, x2, y2 = box
                    all_boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])  # Adjust coordinates to full image
        
        pumpkin_count = len(all_boxes)
        
        # Draw bounding boxes and label the pumpkin count on the image
        for box in all_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(output_image, f"Total Pumpkins: {pumpkin_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
    print(f"Total estimated pumpkin count: {pumpkin_count}")
    
    # Save the final image with bounding boxes
    cv2.imwrite("output_pumpkin_count.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    
    # Display final bounding box image
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_image, cmap='gray')
    plt.title("Final Masked Image")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(output_image)
    plt.title("Final Image with Bounding Boxes")
    plt.axis("off")
    plt.show()
    
    return pumpkin_count

mean_rgb = np.array([232, 167, 86])  # Adjust based on exercise3_1_1.py
sample_pixels = np.random.multivariate_normal(mean_rgb, np.diag([50, 50, 50]), size=1000)  # Simulated pumpkin pixels
cov_inv = estimate_covariance(sample_pixels)
process_orthomosaic("Gyldensteensvej-9-19-2017-orthophoto (1).tif", mean_rgb, cov_inv)
