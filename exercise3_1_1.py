import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables for storing selected points
selected_pixels = []

# Mouse callback function to capture selected points
def select_pixels(event, x, y, flags, param):
    global selected_pixels
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixels.append(image[y, x])  # Store the color at the clicked point
        cv2.circle(display_image, (x, y), 3, (0, 255, 0), -1)  # Mark selection
        cv2.imshow("Select Pumpkin Pixels", display_image)

# Load the test image
image = cv2.imread("/home/talha/Desktop/LSDP_miniproject/Images for first miniprojectEB-02-660_0595_0067.JPG")  # Change filename as needed
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
display_image = image.copy()

cv2.imshow("Select Pumpkin Pixels", display_image)
cv2.setMouseCallback("Select Pumpkin Pixels", select_pixels)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite("annotated_pumpkin.jpg", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
print("Annotated image saved as 'annotated_pumpkin.jpg'.")

# Convert selected pixels to NumPy array
selected_pixels = np.array(selected_pixels, dtype=np.uint8)

# Compute mean and standard deviation in RGB
mean_rgb = np.mean(selected_pixels, axis=0)
std_rgb = np.std(selected_pixels, axis=0)

# Convert selected pixels to CieLAB
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
# Convert selected RGB pixels to CieLAB
selected_pixels_lab = cv2.cvtColor(selected_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)

# Compute mean and standard deviation in CieLAB
mean_lab = np.mean(selected_pixels_lab, axis=0)
std_lab = np.std(selected_pixels_lab, axis=0)

# Print results
print("Mean RGB:", mean_rgb)
print("Std RGB:", std_rgb)
print("Mean CieLAB:", mean_lab)
print("Std CieLAB:", std_lab)

# Visualize color distributions
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# RGB histogram
ax[0].hist(selected_pixels[:, 0], bins=25, color='r', alpha=0.6, label="Red")
ax[0].hist(selected_pixels[:, 1], bins=25, color='g', alpha=0.6, label="Green")
ax[0].hist(selected_pixels[:, 2], bins=25, color='b', alpha=0.6, label="Blue")
ax[0].set_title("RGB Color Distribution")
ax[0].legend()

# CieLAB histogram
ax[1].hist(selected_pixels_lab[:, 1], bins=25, color='orange', alpha=0.6, label="A (Green-Red)")
ax[1].hist(selected_pixels_lab[:, 2], bins=25, color='blue', alpha=0.6, label="B (Blue-Yellow)")
ax[1].set_title("CieLAB Color Distribution")
ax[1].legend()

plt.show()

