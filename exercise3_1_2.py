import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the test image
image = cv2.imread("EB-02-660_0595_0067.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Manually set the mean pumpkin color (from Exercise 3.1.1)
mean_rgb = np.array([232.23529412, 167.66666667, 86.7745098])  # Replace with actual mean values

# Reshape image for Mahalanobis distance computation
reshaped_image = image.reshape(-1, 3).astype(np.float32)

# Compute the covariance matrix and inverse covariance matrix
cov_matrix = np.cov(reshaped_image.T)  # Transpose for correct shape
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Compute Mahalanobis distance for each pixel
mahalanobis_distances = np.array([
    np.sqrt((pixel - mean_rgb).T @ inv_cov_matrix @ (pixel - mean_rgb))
    for pixel in reshaped_image
])

# Reshape distance map back to the original image shape
mahalanobis_distances = mahalanobis_distances.reshape(image.shape[:2])

# Apply threshold to create a binary mask
threshold = 6  # Adjust threshold as needed
mask_mahalanobis = (mahalanobis_distances < threshold).astype(np.uint8) * 255  # Convert to binary mask

# Apply morphological closing to fill small gaps
kernel = np.ones((7, 7), np.uint8)
mask_filled = cv2.morphologyEx(mask_mahalanobis, cv2.MORPH_CLOSE, kernel)

# Save the mask image
cv2.imwrite("pumpkin_mask_mahalanobis.png", mask_filled)
print("Mask image saved as 'pumpkin_mask_mahalanobis.png'")

### Counting Pumpkins and Drawing Bounding Boxes ###
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)

# Create an output image to draw bounding boxes
output_image = image.copy()

pumpkin_count = 0
for i in range(1, num_labels):  # Skip background (label 0)
    x, y, w, h, area = stats[i]
    
    # Apply minimum area filter to remove small noise regions
    if area > 100:
        pumpkin_count += 1
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(output_image, f"Pumpkin {pumpkin_count}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"Number of pumpkins detected: {pumpkin_count}")

# Display the results
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(mask_filled, cmap="gray")
ax[1].set_title("Mahalanobis Distance Mask")
ax[1].axis("off")

ax[2].imshow(output_image)
ax[2].set_title(f"Detected Pumpkins: {pumpkin_count}")
ax[2].axis("off")

plt.show()

# Save the final image with bounding boxes
cv2.imwrite("pumpkins_detected_mahalanobis.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print("Annotated image saved as 'pumpkins_detected_mahalanobis.png'.")
