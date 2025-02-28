import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the test image
image = cv2.imread("/home/talha/Downloads/Images for first miniproject/EB-02-660_0595_0068.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Manually set the mean pumpkin color (from Exercise 3.1.1)
mean_rgb = np.array([232.23529412,167.66666667,86.7745098 ])  # Example values, replace with actual mean values
mean_lab = np.array([187.33333333,143.71568627,178.01960784])  # Example values, replace with actual mean values

# Convert image to CieLAB color space
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

### 1. Thresholding using `cv2.inRange()` in RGB ###
lower_rgb = mean_rgb - np.array([40, 20, 40])  # Adjust threshold range
upper_rgb = mean_rgb + np.array([40, 15, 40])
mask_rgb = cv2.inRange(image, lower_rgb, upper_rgb)

### 2. Thresholding using `cv2.inRange()` in CieLAB ###
lower_lab = mean_lab - np.array([27, 42, 25])  # Adjust threshold range
upper_lab = mean_lab + np.array([27, 42, 25])
mask_lab = cv2.inRange(image_lab, lower_lab, upper_lab)

### 3. Euclidean Distance in RGB Space ###
distance = np.linalg.norm(image - mean_rgb, axis=2)
threshold = 60  # Adjust threshold
mask_distance = (distance < threshold).astype(np.uint8) * 255  # Convert to binary mask

# Apply morphological closing to fill gaps
kernel = np.ones((7,7), np.uint8)  # Adjust kernel size if needed
mask_filled = cv2.morphologyEx(mask_distance, cv2.MORPH_CLOSE, kernel)
# Save the mask image
cv2.imwrite("pumpkin_mask1.png", mask_filled)
print("Mask image saved as 'pumpkin_mask.png'")
# Display the results
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(mask_rgb, cmap="gray")
ax[1].set_title("RGB inRange Mask")
ax[1].axis("off")

ax[2].imshow(mask_lab, cmap="gray")
ax[2].set_title("CieLAB inRange Mask")
ax[2].axis("off")

ax[3].imshow(mask_filled, cmap="gray")
ax[3].set_title("RGB Distance Mask")
ax[3].axis("off")

plt.show()

