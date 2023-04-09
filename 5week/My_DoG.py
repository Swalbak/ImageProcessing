import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_filtering import my_filtering


def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-(fsize//2):(fsize//2)+1, -(fsize//2):(fsize//2)+1]

    DoG_x = (-x / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma**2))
    DoG_y = (-y / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma**2))

    return DoG_x, DoG_y

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=100)
    x = ((x - x.min()) / (x - x.min()).max() * 255).astype(np.uint8)
    y = ((y - y.min()) / (y - y.min()).max() * 255).astype(np.uint8)

    dst_x = np.abs(my_filtering(src, DoG_x))
    dst_y = np.abs(my_filtering(src, DoG_y))

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = np.sqrt((dst_x ** 2 + dst_y ** 2))

    """
   
    """

    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x ', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

