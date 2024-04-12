import cv2
import numpy as np
from skimage import measure
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestClassifier
import pytesseract
from pytesseract import Output
from scipy.optimize import brentq
from matplotlib import *
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
    
    return scale_text


image = cv2.imread('probebot/ProbeImages/4932_007.tif', cv2.IMREAD_COLOR)


if(image is None):
    print("null image")
    
roi = image[:-60, 0:607]
roicopy = roi.copy()


def solve_for_y(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc).real

if roi is None:
    print("Error: Failed to load the ROI.")
else:
    scale = extract_scale_from_info_bar(image)
        
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
    

    
    
    contour_points = [tuple(point[0]) for contour in contours for point in contour]
    
    
    
    x = [point[0] for point in contour_points]
    y = [point[1] for point in contour_points]
    
    
    fit =  np.polyfit(x,y,15)
    approximationFunction = np.poly1d(fit)
    
    height, width, channels = roi.shape
    
    for i in range(width):
        cv2.circle(roi, (i, int(approximationFunction(i))), 2, (255,255,0), -1 )
    
    
    rootsX = np.array([root for root in approximationFunction.deriv().r.real if root > 0 ])

    rootsY = np.array([root for root in approximationFunction(rootsX)])
    
    minY = min(rootsY)
    # xy = np.c_[rootsX, rootsY].
    roots =list(zip(rootsX, rootsY))
    
    criticalPoint = (0,minY)
    
    for root in roots:
        if root[1] == minY:
            criticalPoint = (int(root[0]), int(root[1]))
            break
    
    # for root in roots:
    #     cv2.circle(roi, (int(root[0]), int(root[1])), 5, (0,0,255), -1)
           
    # for i in range(width):
    #     cv2.circle(roi, (i, 294), 2, (255,255,255), -1)
    
    
    print(f"cp: {criticalPoint}")
    
    
    cv2.circle(roi, criticalPoint, 5, (0,0,255), -1)
    
    left_partition = [point for point in contour_points if point[0] < criticalPoint[0]]
    right_partition = [point for point in contour_points if point[0] > criticalPoint[0]]
    
    
   
    # Loop over each contour
    for point in left_partition:
        
        
        color = (0,255,0)
        
        cv2.circle(roi, point, 3, color, -1)
            
    # Loop over each contour
    for point in right_partition:
        color = (255,0,0)
        cv2.circle(roi, point, 4, color, -1)
    

    left_partition_x = [point[0] for point in left_partition]
    left_partition_y = [point[1] for point in left_partition]
    
    left_partition_fit = np.polyfit(left_partition_x, left_partition_y, 1)
    left_partition_fit_function = np.poly1d(left_partition_fit)
    
    right_partition_x = [point[0] for point in right_partition]
    right_partition_y = [point[1] for point in right_partition]
    
    right_partition_fit = np.polyfit(right_partition_x, right_partition_y, 1)
    right_partition_fit_function = np.poly1d(right_partition_fit)
    
    for i in range(width):
        cv2.circle(roi, (i, int(left_partition_fit_function(i))), 2, (0,0,255), -1)
    
    for i in range(width):
        cv2.circle(roi, (i, int(right_partition_fit_function(i))), 2, (0,0,255), -1)
    
    highestPoint = min(contour_points, key=lambda point: point[1])
    
    cv2.circle(roi, highestPoint, 5, (0,0,255), -1)
    
    print()
    
    left_roots = solve_for_y(left_partition_fit, highestPoint[1])
    left_roots_distance_from_cp = {}
    
    print(f"left roots: {left_roots}")
    
    for root in left_roots:
        left_roots_distance_from_cp[root] = (np.abs(criticalPoint[0] - root))
    
    left_measurement_point_x = int(min(left_roots_distance_from_cp, key=left_roots_distance_from_cp.get))
    
    print(left_measurement_point_x)
    
    right_roots = solve_for_y(right_partition_fit, highestPoint[1])
    right_roots_distance_from_cp = {}
    
    for root in right_roots:
        right_roots_distance_from_cp[root] = np.abs(criticalPoint[0] - root)
    
    right_measurement_point_x = int(min(right_roots_distance_from_cp, key=right_roots_distance_from_cp.get))
    
    left_measurement_point = (left_measurement_point_x, highestPoint[1])
    right_measurement_point = (right_measurement_point_x, highestPoint[1])
    
    cv2.line(roicopy, left_measurement_point, right_measurement_point, (0,255,255), 2)
    cv2.line(roi, left_measurement_point, right_measurement_point, (0,255,255), 2)
    
    diameter = abs(right_measurement_point_x - left_measurement_point_x)
    
    print(f"diameter : {diameter}")

    tip_radius = diameter/2
    
    print(f"radius: {tip_radius}")
    
    result =""
    
    if (tip_radius > 0 and tip_radius <= 10):
        result = "Sub 10"
    elif (tip_radius > 10 and tip_radius <= 20):
        result = "Sub 20"
    elif (tip_radius > 20 and tip_radius <= 40):
        result = "Sub 40"
    else:
        result = str(int(tip_radius))
   
   
    roiFinal = np.concatenate((roi, roicopy), axis=1)
    
    cv2.imshow(result, roiFinal)
    # cv2.imshow(result, roi)
       
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

