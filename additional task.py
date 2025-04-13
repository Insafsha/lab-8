import cv2
import numpy as np

# Загружаем изображение мухи
fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)

if fly is None:
    print("Изображение мухи не найдено. Убедитесь, что 'fly64.png' в папке.")
    exit()

fly_h, fly_w = fly.shape[:2]

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

            # === Наложение мухи ===
            top_left_x = center_x - fly_w // 2
            top_left_y = center_y - fly_h // 2

            # Проверка выхода за границы
            if 0 <= top_left_x <= width - fly_w and 0 <= top_left_y <= height - fly_h:
                roi = frame[top_left_y:top_left_y + fly_h, top_left_x:top_left_x + fly_w]

                # Если муха с альфа-каналом (прозрачностью)
                if fly.shape[2] == 4:
                    fly_rgb = fly[:, :, :3]
                    alpha_mask = fly[:, :, 3] / 255.0
                    for c in range(3):
                        roi[:, :, c] = (1 - alpha_mask) * roi[:, :, c] + alpha_mask * fly_rgb[:, :, c]
                else:
                    roi[:] = fly

    cv2.imshow("Tag Tracking", frame)

    if cv2.getWindowProperty("Tag Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
