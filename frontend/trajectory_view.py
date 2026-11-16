
import cv2


def draw_trajectory(
    frame,
    positions,
    bbox=None,
    predicted_landing=None
):
    """
    Draw object tracking information on a frame.

    positions: list of (x, y) coordinates
    bbox: (x, y, width, height)
    predicted_landing: (x, y)
    """

    # Draw green circles at tracked positions
    for x, y in positions:
        cv2.circle(
            frame,
            (int(x), int(y)),
            5,
            (0, 255, 0),
            -1
        )

    # Connect positions with green lines
    for i in range(1, len(positions)):
        start = tuple(map(int, positions[i - 1]))
        end = tuple(map(int, positions[i]))

        cv2.line(
            frame,
            start,
            end,
            (0, 255, 0),
            2
        )

    # Draw bounding box around current object
    if bbox is not None:
        x, y, w, h = map(int, bbox)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Tracked Object [Demo]",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Draw predicted landing marker
    if predicted_landing is not None:
        px, py = map(int, predicted_landing)

        cv2.drawMarker(
            frame,
            (px, py),
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            20,
            3
        )

        cv2.putText(
            frame,
            "Predicted Landing [Demo]",
            (max(0, px - 90), max(20, py - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    return frame
