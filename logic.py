import cv2

def analyze_navigation_zones(img, detections):
    """
    Analyzes YOLO detections to provide navigation advice.
    Returns: (suggestion_text, text_color)
    """
    h, w, _ = img.shape
    center_left_bound = int(w * 0.33)
    center_right_bound = int(w * 0.66)

    suggestion = "PATH CLEAR"
    color = (0, 255, 0)

    if not detections:
        return suggestion, color

    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        box_w = x2 - x1
        box_h = y2 - y1
        box_center_x = x1 + (box_w // 2)

        # LOGIC: Hazard if object is > 25% height of screen
        if box_h > h * 0.25:
            if center_left_bound < box_center_x < center_right_bound:
                return "STOP! OBSTACLE", (0, 0, 255) # Red
            elif box_center_x < center_left_bound:
                suggestion = "Keep Right >>"
                color = (0, 255, 255) # Yellow
            elif box_center_x > center_right_bound:
                suggestion = "<< Keep Left"
                color = (0, 255, 255) # Yellow

    return suggestion, color
