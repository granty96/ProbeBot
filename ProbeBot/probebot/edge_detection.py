import cv2
import numpy as np
from skimage import measure
from sklearn.ensemble import RandomForestClassifier
import pytesseract
from pytesseract import Output
import re
from PIL import Image

def extract_scale_from_info_bar(image) -> float:
    # Extract the text from the bottom info bar
    bottom_bar_text = image[-50:, 320:430].copy()  # Extract bottom 50 pixels
 
    _, thresholded = cv2.threshold(bottom_bar_text, 200, 255, cv2.THRESH_BINARY)
    
    # cv2.imshow("thresholded",thresholded)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    
    scale_text = pytesseract.image_to_string(thresholded, config='--psm 6').strip()
    
    scale_text = float(re.sub(r'\D', '', scale_text))
    print(scale_text)
    
    return scale_text


image = cv2.imread('probebot/images/3630_005.tif', cv2.IMREAD_COLOR)


if(image is None):
    print("null image")
    
roi = image[:-60, 0:607]




if roi is None:
    print("Error: Failed to load the ROI.")
else:
    scale = extract_scale_from_info_bar(image)
    if scale and scale == 300000.0:
        
        nm_per_pixel = 200.0/81.0;
        image_blurred_noHSV = cv2.GaussianBlur(roi, (5, 5), 0)
        image_blurred = cv2.cvtColor(image_blurred_noHSV, cv2.COLOR_BGR2HSV)
        
        lower = np.array([0,0,0], dtype="uint8")
        upper = np.array([180,255,40], dtype="uint8")
        
        mask = cv2.inRange(image_blurred, lower, upper)
        mask = 255-mask
       
        image_blurred = mask        
       # Edge Detection
        edges = cv2.Canny(image_blurred, 30, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with the largest area (probe tip)
        largest_contour = max(contours, key=cv2.contourArea)

        print(largest_contour)
        # Approximate the contour with a polygon using the Douglas-Peucker algorithm
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # Adjust epsilon as needed
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Sort the vertices based on their y-coordinates
        sorted_vertices = sorted(approx[:, 0], key=lambda x: x[1])
        pt1 = sorted_vertices[0]

        
        #select the next vertex in the line
        for vertex in sorted_vertices:
            print(abs(vertex[0] - pt1[0]))
            if(abs(vertex[0] - pt1[0]) > 10):
                pt2 = vertex
                break
            

        # Draw the lines connecting the selected vertices on the image
          # Convert to color image for visualization
        cv2.line(roi, tuple(pt1), tuple(pt2), (0, 255, 0), thickness=2)  # Draw the line in green with thickness 2
        # cv2.line(line_image, tuple(pt1), tuple(pt3), (255, 0, 0), thickness=2)
        # cv2.line(line_image, tuple(pt1), tuple(pt4), (0,0,255), thickness=2)

        # Display the image with the drawn lines
        cv2.imshow('Image with Lines (Probe Tip)', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.circle(result_image, (int(x), int(y)), int(radius), (0,255,0), 2)
        
        # radius_nm = radius*nm_per_pixel

        # print(str(radius_nm) + " nm")

        # cv2.imshow('Result', result_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("No scale found")

