import cv2
import numpy as np

img_path = 'images/variant-5.jpg'
img = cv2.imread(img_path)

img_float = img / 255.0

#Генерация шума 
mean = 0
st = 0.1
noise = np.random.normal(mean, st, img.shape)
noisy_img = img_float + noise
noisy_img = np.clip(noisy_img, 0, 1)

noisy_img_u8 = (noisy_img * 255).astype(np.uint8)

cv2.imshow("Original", img)
cv2.imshow("With noise", noisy_img_u8)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Сохранение
save_path = 'images/variant-5-noisy.jpg'
cv2.imwrite(save_path, noisy_img_u8)

