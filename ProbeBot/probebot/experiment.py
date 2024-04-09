import os
import cv2
import numpy as np

print(f"directory : {os.getcwd()}")
# Load the TIFF image
image = cv2.imread('probebot/images/3630_006.tif', cv2.IMREAD_GRAYSCALE)

# Define the region of interest (ROI) to exclude the bottom info bar
roi = image[:-50, :]

# Extracted scale information (200 nm spans from x-coordinates 608 to 688)
scale_width_pixels = 688 - 608 + 1  # Width of the scale in pixels
scale_width_nm = 200  # Width of the scale in nanometers

# Calculate nanometers per pixel
nm_per_pixel = scale_width_nm / scale_width_pixels

# Preprocessing
# Apply Gaussian blur for noise reduction
image_blurred = cv2.GaussianBlur(roi, (5, 5), 0)

# Edge Detection
edges = cv2.Canny(image_blurred, 30, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area (probe tip)
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour with a polygon using the Douglas-Peucker algorithm
epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # Adjust epsilon as needed
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Sort the vertices based on their y-coordinates
sorted_vertices = sorted(approx[:, 0], key=lambda x: x[1])
pt1 = sorted_vertices[0]

for vertex in sorted_vertices:
    print(vertex[0] - pt1[0])
    if(vertex[0] - pt1[0] > 5):
        pt2 = vertex
        break
    
print(sorted_vertices)

# Draw the lines connecting the selected vertices on the image
line_image = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convert to color image for visualization
cv2.line(line_image, tuple(pt1), tuple(pt2), (0, 255, 0), thickness=2)  # Draw the line in green with thickness 2
# cv2.line(line_image, tuple(pt1), tuple(pt3), (255, 0, 0), thickness=2)
# cv2.line(line_image, tuple(pt1), tuple(pt4), (0,0,255), thickness=2)

# Display the image with the drawn lines
cv2.imshow('Image with Lines (Probe Tip)', line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate the distance between the two points (diameter of the tip)
diameter = np.linalg.norm(pt1 - pt2)

# Convert the diameter to nanometers using the scale
diameter_nm = diameter * nm_per_pixel
print("Probe Tip Diameter (in nanometers):", diameter_nm)
