import cv2
import numpy as np
import pytesseract
from matplotlib import *
import re
import warnings

batchNumber = "4955"

full_fit_degree = 8
left_fit_degree = 2
right_fit_degree = 2


def extract_magnification(image) -> float:
    # Extract the text from the bottom info bar
    bottom_bar_text = image[-50:, 320:430].copy()  # Extract bottom 50 pixels
    _, thresholded = cv2.threshold(bottom_bar_text, 200, 255, cv2.THRESH_BINARY)
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    )
    magnification = pytesseract.image_to_string(thresholded, config="--psm 6").strip()
    magnification = float(re.sub(r"\D", "", magnification))
    return magnification


def solve_for_y(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc).real


def measureTip(imagePath, orientation):

    warnings.simplefilter("ignore", np.RankWarning)

    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

    scale = extract_magnification(image)

    if image is None:
        raise ("Image not found")

    # Regularize orientation, Crop out scale bars and info bar
    if orientation != "up":
        image = cv2.rotate(image, cv2.ROTATE_180)
        roi = image[60:, 298:]
    else:
        roi = image[:-60, 0:607]

    roiCopy = roi.copy()

    if scale < 29999:
        raise Exception(
            "Invalid scale, please only use images with 300,000x magnification."
        )

    image_blurred = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)

    # Mask image
    maskLowerBound = np.array([0, 0, 0], dtype="uint8")
    maskUpperBound = np.array([180, 255, 40], dtype="uint8")
    mask = cv2.inRange(image_blurred, maskLowerBound, maskUpperBound)
    mask = 255 - mask

    # Edge Detection on mask
    edges = cv2.Canny(mask, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_points = [tuple(point[0]) for contour in contours for point in contour]
    x = [point[0] for point in contour_points]
    y = [point[1] for point in contour_points]

    # fit contour to a n-th degree polynomial
    fit = np.polyfit(x, y, full_fit_degree)
    approximationFunction = np.poly1d(fit)
    width = roi.shape[1]

    # draw approximation function
    for i in range(width):
        try:
            cv2.circle(roi, (i, int(approximationFunction(i))), 2, (255, 255, 0), -1)
        except:
            print("Error displaying approximation function")
            print(str(int(approximationFunction(i))))

    # compute the highest or lowest critical point of the probe using the fit (x of probe tip)
    rootsX = np.array(
        [root for root in approximationFunction.deriv().r.real if root > 0]
    )
    rootsY = np.array([root for root in approximationFunction(rootsX)])

    criticalPointY = min(rootsY)

    roots = list(zip(rootsX, rootsY))

    criticalPoint = (0, criticalPointY)

    for root in roots:
        if root[1] == criticalPointY:
            criticalPoint = (int(root[0]), int(root[1]))
            break

    cv2.circle(roi, criticalPoint, 5, (0, 0, 255), -1)

    # Partition contour about critical point x.
    left_partition = [point for point in contour_points if point[0] < criticalPoint[0]]
    right_partition = [point for point in contour_points if point[0] > criticalPoint[0]]

    # Draw each partition

    for point in left_partition:
        color = (0, 255, 0)
        cv2.circle(roi, point, 3, color, -1)

    for point in right_partition:
        color = (255, 0, 0)
        cv2.circle(roi, point, 4, color, -1)

    # Fit left partition
    left_partition_x = [point[0] for point in left_partition]
    left_partition_y = [point[1] for point in left_partition]

    left_partition_fit = np.polyfit(left_partition_x, left_partition_y, left_fit_degree)
    left_partition_fit_function = np.poly1d(left_partition_fit)

    # Fit right partition
    right_partition_x = [point[0] for point in right_partition]
    right_partition_y = [point[1] for point in right_partition]

    right_partition_fit = np.polyfit(
        right_partition_x, right_partition_y, right_fit_degree
    )
    right_partition_fit_function = np.poly1d(right_partition_fit)

    # Draw left and right partition fit functions
    for i in range(width):
        cv2.circle(roi, (i, int(left_partition_fit_function(i))), 2, (0, 0, 255), -1)

    for i in range(width):
        cv2.circle(roi, (i, int(right_partition_fit_function(i))), 2, (0, 0, 255), -1)

    # Compute the y value of the tip
    highestPoint = min(contour_points, key=lambda point: point[1])

    # draw the tip
    cv2.circle(roi, highestPoint, 5, (0, 0, 255), -1)

    # Compute the root of the partition fit functions that are closest to the tip.
    left_roots = solve_for_y(left_partition_fit, highestPoint[1])
    left_roots_distance_from_cp = {}

    for root in left_roots:
        left_roots_distance_from_cp[root] = np.abs(criticalPoint[0] - root)

    left_measurement_point_x = int(
        min(left_roots_distance_from_cp, key=left_roots_distance_from_cp.get)
    )

    right_roots = solve_for_y(right_partition_fit, highestPoint[1])
    right_roots_distance_from_cp = {}

    for root in right_roots:
        right_roots_distance_from_cp[root] = np.abs(criticalPoint[0] - root)

    right_measurement_point_x = int(
        min(right_roots_distance_from_cp, key=right_roots_distance_from_cp.get)
    )

    left_measurement_point = (left_measurement_point_x, highestPoint[1])
    right_measurement_point = (right_measurement_point_x, highestPoint[1])

    # draw tip measurement

    cv2.line(roiCopy, left_measurement_point, right_measurement_point, (0, 255, 255), 2)
    cv2.line(roi, left_measurement_point, right_measurement_point, (0, 255, 255), 2)

    diameter = abs(right_measurement_point_x - left_measurement_point_x)

    # compute radius & diameter
    print(f"diameter: {diameter}")
    tip_radius = diameter / 2
    print(f"radius: {tip_radius}")

    cv2.putText(
        roiCopy,
        f"{int(tip_radius)} nm",
        (left_measurement_point[0], left_measurement_point[1] - 10),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    result = ""

    if tip_radius > 0 and tip_radius <= 10:
        result = "Sub 10"
    elif tip_radius > 10 and tip_radius <= 20:
        result = "Sub 20"
    elif tip_radius > 20 and tip_radius <= 40:
        result = "Sub 40"
    else:
        result = str(int(tip_radius))

    roiFinal = np.concatenate((roi, roiCopy), axis=1)

    # return image and measurement
    return (roiFinal, result)


image, result = measureTip(
    "C:/Users/Grant/Desktop/ProbeBot/ProbeBot/probebot/ProbeImages/12Feb24/4962_007.tif",
    "down",
)

cv2.imshow(result, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
