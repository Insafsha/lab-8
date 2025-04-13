# для отслеживания, я использовал грань кубика рубика(конктретно синего цвета, поэтому в коде используется маска для синего цвета)
# для кубика синего цвета код работает идеально(не было возможности напечатать метку)
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 70])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 1000:
            x, y, w, h = cv2.boundingRect(c)
            center_x = x + w // 2
            center_y = y + h // 2

            if center_x < 100 and center_y < 100:
                box_color = (255, 0, 0)
            elif center_x > (width - 100) and center_y > (height - 100):
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)

            padding = 10
            x_new = x + padding
            y_new = y + padding
            w_new = max(1, w - 2 * padding)
            h_new = max(1, h - 2 * padding)

            cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), box_color, 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            print(f"Центр метки: ({center_x}, {center_y})")

    cv2.imshow("Tag Tracking", frame)

    if cv2.getWindowProperty("Tag Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

