import argparse
import cv2
import numpy as np


def fill_notches(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)

    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx)
        cv2.drawContours(filled_mask, [hull], -1, 255, thickness=cv2.FILLED)

    filled_mask = cv2.bitwise_or(filled_mask, mask)
    return filled_mask


def highlight_green_as_blue(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = fill_notches(mask)
    image[mask != 0] = [255, 0, 0]

    return image


def detect_edges(image):
    blue_mask = cv2.inRange(image, np.array(
        [255, 0, 0]), np.array([255, 0, 0]))
    blue_regions = cv2.bitwise_and(image, image, mask=blue_mask)
    gray = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (1, 1), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    return edges


def draw_lines(image, edges):
    kernel = np.ones((5, 5), np.uint8)
    smoothed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    smoothed_edges = cv2.GaussianBlur(smoothed_edges, (5, 5), 1)
    _, smoothed_edges = cv2.threshold(
        smoothed_edges, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        smoothed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_image = image.copy()
    line_segments = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon=2, closed=True)

        for i in range(len(approx)):
            pt1 = tuple(approx[i][0])
            pt2 = tuple(approx[(i + 1) % len(approx)][0])

            length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            line_segments.append((pt1, pt2, length))

            cv2.line(line_image, pt1, pt2, (0, 255, 0), 2)

    line_segments.sort(key=lambda x: x[2], reverse=True)

    for segment in line_segments[:4]:
        pt1, pt2, _ = segment
        cv2.line(line_image, pt1, pt2, (0, 0, 255), 3)

    return line_image, line_segments


def find_line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
          (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
          (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return int(px), int(py)


def fit_intersections_to_lines(image, line_segments):
    if len(line_segments) < 4:
        return image, []

    longest_lines = line_segments[:4]
    intersections = []

    for i in range(len(longest_lines)):
        for j in range(i + 1, len(longest_lines)):
            pt1, pt2, _ = longest_lines[i]
            pt3, pt4, _ = longest_lines[j]
            intersection = find_line_intersection((pt1, pt2), (pt3, pt4))
            if intersection:
                intersections.append(intersection)

    image_with_points = image.copy()
    for point in intersections:
        cv2.circle(image_with_points, point, radius=5,
                   color=(0, 0, 255), thickness=-1)

    for i in range(len(intersections)):
        for j in range(i + 1, len(intersections)):
            cv2.line(image_with_points,
                     intersections[i], intersections[j], (0, 0, 255), 2)

    return image_with_points, intersections[1:-1]


def detect_green_screen_corners(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")

    result_image = highlight_green_as_blue(image)
    edges = detect_edges(result_image)
    line_image, line_segments = draw_lines(result_image, edges)
    final_image, corner_points = fit_intersections_to_lines(
        line_image, line_segments)

    return final_image, corner_points


def main():
    parser = argparse.ArgumentParser(
        description="Detect green screen corners from an image.")
    parser.add_argument("--input", required=True,
                        help="Path to the input image.")
    args = parser.parse_args()

    try:
        result_image, corner_coordinates = detect_green_screen_corners(
            args.input)
        print("Detected corner coordinates:", corner_coordinates)

        cv2.imshow("Corner Detection Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
