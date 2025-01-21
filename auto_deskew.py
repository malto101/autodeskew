from types import coroutine
import cv2
import numpy as np


def fill_notches(mask):
    """
    Detects and fills notches (U-shaped gaps) in the green mask.
    Args:
        mask (numpy.ndarray): Binary mask of the green screen.
    Returns:
        filled_mask (numpy.ndarray): Binary mask with notches filled.
    """

    # Find contours of the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw filled contours
    filled_mask = np.zeros_like(mask)

    for contour in contours:
        # Approximate the contour to simplify it
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Create a convex hull to cover notches (fills U-shaped gaps)
        hull = cv2.convexHull(approx)

        # Draw the filled hull on the mask
        cv2.drawContours(filled_mask, [hull], -1, 255, thickness=cv2.FILLED)

    filled_mask = cv2.bitwise_or(filled_mask, mask)

    return filled_mask

def highlight_green_as_blue(image):

    """
    Highlights all green regions in the image by changing them to blue.
    Args:
        image (numpy.ndarray): The input image.
    Returns:
        result_image (numpy.ndarray): The image with green regions changed to blue.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the green color range in HSV (you can adjust these values if needed)
    lower_green = np.array([40, 100, 100])  # Lower bound for bright green
    upper_green = np.array([80, 255, 255])  # Adjust based on your image
    
    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean the mask (removes small noise)
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the size of the kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill notches in the mask
    mask = fill_notches(mask)
    # Convert the mask to blue
    image[mask != 0] = [255, 0, 0]  # Change green areas to blue (BGR: [255, 0, 0])
    
    return image


def detect_edges(image): 
    """
    Detects edges in the blue regions of the image using the Canny edge detector.
    Args:
        image (numpy.ndarray): The input image with highlighted blue regions.
    Returns:
        edges (numpy.ndarray): The edges of the blue regions.
    """
    # Create a mask to isolate blue regions (BGR: [255, 0, 0])
    blue_mask = cv2.inRange(image, np.array([255, 0, 0]), np.array([255, 0, 0]))
    
    # Convert to grayscale, but only on the blue regions
    blue_regions = cv2.bitwise_and(image, image, mask=blue_mask)
    gray = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (1, 1), 1.5)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def draw_lines(image, edges):
    """
    Smooths the edges and draws lines based on the detected edges in the image.
    Highlights the top 4 longest lines in red.
    Args:
        image (numpy.ndarray): The original image.
        edges (numpy.ndarray): The edge-detected image.
    Returns:
        line_image (numpy.ndarray): The image with smoothed lines and highlighted longest lines.
    """
    # Smooth the edges using morphological operations and Gaussian blur
    kernel = np.ones((5, 5), np.uint8)
    smoothed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    smoothed_edges = cv2.GaussianBlur(smoothed_edges, (5, 5), 1)

    # Threshold to ensure binary edges (0 or 255)
    _, smoothed_edges = cv2.threshold(smoothed_edges, 50, 255, cv2.THRESH_BINARY)

    # Find contours from the smoothed edge image
    contours, _ = cv2.findContours(smoothed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw lines
    line_image = image.copy()

    # Store all line segments with their lengths
    line_segments = []

    for contour in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, epsilon=2, closed=True)
        
        # Draw lines connecting the points of the approximated contour
        for i in range(len(approx)):
            pt1 = tuple(approx[i][0])  # Start point of the line
            pt2 = tuple(approx[(i + 1) % len(approx)][0])  # End point of the line
            
            # Calculate the length of the line segment
            length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            line_segments.append((pt1, pt2, length))
            
            # Draw the line in green
            cv2.line(line_image, pt1, pt2, (0, 255, 0), 2)  # Green lines

    # Sort line segments by length in descending order
    line_segments.sort(key=lambda x: x[2], reverse=True)

    # Highlight the top 4 longest lines in red
    for segment in line_segments[:4]:  # Get the top 4 longest lines
        pt1, pt2, _ = segment
        cv2.line(line_image, pt1, pt2, (0, 0, 255), 3)  # Red lines (thicker)

    return line_image, line_segments


def find_line_intersection(line1, line2):
    """
    Finds the intersection point of two lines (extended indefinitely).
    Args:
        line1, line2: Tuples ((x1, y1), (x2, y2)) representing two lines.
    Returns:
        (x, y): Intersection point as a tuple, or None if lines are parallel.
    """
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]

    # Calculate the determinant
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:  # Lines are parallel or coincident
        return None

    # Calculate the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
          (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
          (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return int(px), int(py)


def fit_intersections_to_lines(image, line_segments):
    """
    Finds and marks the intersection points of the four longest lines.
    Args:
        image (numpy.ndarray): The original image.
        line_segments (list): List of line segments [(pt1, pt2, length), ...].
    Returns:
        image_with_points (numpy.ndarray): The image with intersection points drawn.
        intersections (list): The intersection points of the lines.
    """
    # Ensure we have at least 4 lines
    if len(line_segments) < 4:
        print("Error: Not enough line segments to calculate intersections.")
        return image, []

    # Get the top 4 longest lines
    longest_lines = line_segments[:4]

    # Find all pairwise intersections of these lines
    intersections = []
    for i in range(len(longest_lines)):
        for j in range(i + 1, len(longest_lines)):
            pt1, pt2, _ = longest_lines[i]
            pt3, pt4, _ = longest_lines[j]
            intersection = find_line_intersection((pt1, pt2), (pt3, pt4))
            if intersection:
                intersections.append(intersection)

    # Draw the intersections on a copy of the image
    image_with_points = image.copy()
    for point in intersections:
        cv2.circle(image_with_points, point, radius=5,
                   color=(0, 0, 255), thickness=-1)

    # Connect all the points with line
    for i in range(len(intersections)):
        for j in range(i + 1, len(intersections)):
            cv2.line(image_with_points, intersections[i],
                     intersections[j], (0, 0, 255), 2)

    return image_with_points, intersections

def main(image_path):
    """
    Main function to load the image, highlight green regions as blue, detect edges, and save the results.
    Args:
        image_path (str): Path to the input image.
        output_path_with_corners (str): Path to save the image with blue corners.
        output_path_with_edges (str): Path to save the image with detected edges.
        output_path_with_lines (str): Path to save the image with drawn lines.
        output_path_with_rectangle (str): Path to save the image with the fitted rectangle.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Call the function to highlight green areas as blue
    result_image = highlight_green_as_blue(image)

    # Detect edges in the blue areas
    edges = detect_edges(result_image)
    # Draw lines from the detected edges
    line_image, line_segments = draw_lines(result_image, edges)

    # Save the intermediate outputs
    cv2.imshow("output_path_with_corners", result_image)
    cv2.imshow("output_path_with_edges", edges)
    cv2.imshow("output_path_with_lines", line_image)

    # Fit a rectangle to the 4 longest lines and save the result
    image_with_points, intersections = fit_intersections_to_lines(
        line_image, line_segments)
    cv2.imshow("output_path_with_rectangle", image_with_points)
    cv2.waitKey(0)
    print(f"Corner points of the rectangle: {intersections}")

if __name__ == "__main__":
    input_image_path = "assets/image.png"
    main(
        input_image_path,
    )
