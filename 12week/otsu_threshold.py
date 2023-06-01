import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_threshold(img, th=120):
    ######################################################
    # TODO                                               #
    # 실습시간에 배포된 코드 사용                             #
    ######################################################
    dst = np.zeros(img.shape, img.dtype)
    dst[img < th] = 0
    dst[img >= th] = 255
    return dst


def my_otsu_threshold(img):
    hist, bins = np.histogram(img.ravel(),256,[0,256])
    p = hist / np.sum(hist) + 1e-7

    ######################################################
    # TODO                                               #
    # Otsu 방법을 통해 threshold 구한 후 이진화 수행          #
    # cv2의 threshold 와 같은 값이 나와야 함                 #
    ######################################################

    q = np.zeros((256))
    m = np.zeros((256))
    m2 = np.zeros((256))
    var = np.zeros((256))

    q[0] = p[0]
    m[0] = 0
    m2[0] = sum([i * p[i] for i in range(1, 256)]) / (1 - q[0])
    var[0] = q[0] * (1 - q[0]) * m2[0] ** 2

    for k in range(1, 256):
        q[k] = q[k-1] + p[k]
        m[k] = ((m[k-1] * q[k-1]) + p[k] * k) / q[k]
        m2[k] = ((m2[k-1] * (1 - q[k-1])) - p[k] * k) / (1 - q[k])
        
        var[k] = q[k] * (1 - q[k]) * (m[k] - m2[k])**2

    th = np.argmax(var)
    dst = apply_threshold(img, th)

    return th, dst

if __name__ == '__main__':
    img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

    th_cv2, dst_cv2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    th_my, dst_my = my_otsu_threshold(img)

    print('Threshold from cv2: {}'.format(th_cv2))
    print('Threshold from my: {}'.format(th_my))

    cv2.imshow('original image', img)
    cv2.imshow('cv2 threshold', dst_cv2)
    cv2.imshow('my threshold', dst_my)

    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


