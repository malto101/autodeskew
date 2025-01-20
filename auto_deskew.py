from types import coroutine
import cv2
import numpy as np

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


def fit_rectangle_to_lines(image, line_segments):
    """
    Fits a rectangle to the four longest lines in the image.
    Args:
        image (numpy.ndarray): The original image.
        line_segments (list): List of line segments [(pt1, pt2, length), ...].
    Returns:
        image_with_rectangle (numpy.ndarray): The image with the fitted rectangle drawn.
        corners (numpy.ndarray): The corner points of the fitted rectangle as an array of shape (4, 2).
    """
    # Ensure we have at least 4 lines
    if len(line_segments) < 4:
        print("Error: Not enough line segments to fit a rectangle.")
        return image

    # Get the top 4 longest lines
    longest_lines = line_segments[:4]

    # Collect all the endpoints of the 4 longest lines
    points = []
    for pt1, pt2, _ in longest_lines:
        points.extend([pt1, pt2])

    # Convert to NumPy array for OpenCV functions
    points = np.array(points, dtype=np.float32)

    # Fit a minimum-area bounding rectangle to the points
    # rect contains (center, (width, height), angle)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)       # Get the 4 corner points of the rectangle
    box = np.int64(box)              # Convert to integer coordinates
    corners = np.int64(box)          # Convert to integer coordinates

  # Reshape box to match the expected format for cv2.drawContours
    contours = box.reshape((-1, 1, 2))  # Shape it into (n, 1, 2)

    # Draw the rectangle on a copy of the image
    image_with_rectangle = image.copy()
    cv2.drawContours(image_with_rectangle, [contours], -1, (0, 0, 255), 3)  # Draw red rectangle

    return image_with_rectangle, corners


def main(image_path, output_path_with_corners, output_path_with_edges, output_path_with_lines, output_path_with_rectangle):
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
    cv2.imwrite(output_path_with_corners, result_image)
    cv2.imwrite(output_path_with_edges, edges)
    cv2.imwrite(output_path_with_lines, line_image)

    # Fit a rectangle to the 4 longest lines and save the result
    image_with_rectangle, main_corners = fit_rectangle_to_lines(result_image, line_segments)

    cv2.imwrite(output_path_with_rectangle, image_with_rectangle)
    
    print(f"Output image with rectangle saved to: {
          output_path_with_rectangle}")
    print(f"Corner points of the fitted rectangle:\n {main_corners}")

if __name__ == "__main__":
    input_image_path = "image.jpeg"
    output_image_path_with_corners = "output_image_with_corners.jpg"
    output_image_path_with_edges = "output_image_with_edges.jpg"
    output_image_path_with_lines = "output_image_with_lines.jpg"
    output_image_path_with_rectangle = "output_image_with_rectangle.jpg"

    main(
        input_image_path,
        output_image_path_with_corners,
        output_image_path_with_edges,
        output_image_path_with_lines,
        output_image_path_with_rectangle,
    )
