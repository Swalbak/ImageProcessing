import cv2
import numpy as np

src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
gaussian_blar = cv2.GaussianBlur(src, (5, 5), 1)
res = src - gaussian_blar
norm_res = ((res - res.min()) / (res.max() - res.min()) * 255).astype(np.uint8)

cv2.imshow("res", norm_res)
cv2.waitKey()
cv2.destroyAllWindows()