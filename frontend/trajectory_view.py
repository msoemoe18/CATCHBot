
import cv2


def draw_trajectory(
    frame,
    positions,
    bbox=None,
    predicted_landing=None,
    predicted_path=None
):
    # Green observed tracking points
    for x, y in positions:
        cv2.circle(
            frame, (int(x), int(y)),
            4, (0, 255, 0), -1
        )

    # Green observed tracking line
    for i in range(1, len(positions)):
        cv2.line(
            frame,
            tuple(map(int, positions[i - 1])),
            tuple(map(int, positions[i])),
            (0, 255, 0), 2
        )

    # Green bounding box
    if bbox is not None:
        x, y, w, h = map(int, bbox)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            "Tracked Object",
            (max(5, x), max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

    # Red predicted future path
    if predicted_path is not None:
        points = [
            (int(x), int(y))
            for x, y in predicted_path
        ]

        for i in range(1, len(points)):
            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (0, 0, 255), 2
            )

        cv2.putText(
            frame,
            "Predicted Path [Experimental]",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 255), 2
        )

    # Red predicted landing marker
    if predicted_landing is not None:
        px, py = map(int, predicted_landing)

        cv2.drawMarker(
            frame,
            (px, py),
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            20, 3
        )

    return frame
