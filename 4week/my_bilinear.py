import cv2
import numpy as np

"""
해당 부분에 여러분 정보 입력해주세요.
한밭대학교 컴퓨터공학과  20191780 육정훈
"""

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            y = row / scale
            x = col / scale

            m = int(y)
            n = int(x)

            t = y - m
            s = x - n

            """
            픽셀 위치가 이미지를 넘어서는 경우를 막기위해서 조건문을 사용
            각 조건문을 생각하여 코드를 완성하기
            Hint: 4가지에 대한 경우를 생각해야함
            1. m+1, n+1 모두 이미지를 넘어서는 경우
            2. m+1이 이미지를 넘어서는 경우 
            3. n+1이 이미지를 넘어서는 경우
            4. 그외
            """
            value = None

            if m+1 >= h and n+1 >= w:
                value = src[h-1, w-1]
            elif m+1 >= h:
                value = (1-s)*src[h-1, n] + s*src[h-1, n+1]
            elif n+1 >= w:
                value = (1-t)*src[m, w-1] + t*src[m+1, w-1]
            else:
                value = s*t*src[m+1, n+1] + t*(1-s)*src[m+1, n] + s*(1-t)*src[m, n+1] + (1-s)*(1-t)*src[m, n]

            dst[row, col] = value

    return dst

def my_nearest(src, scale):

    (h, w) = src.shape
    
    # 0.5를 더한 값에서 버림 진행 -> 반올림과 같음
    # 이미지 확장 시 원본 픽셀을 넘어서는 픽셀값이 생길 수 있음
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)


    dst = np.zeros((h_dst, w_dst))

    for row in range(h_dst):
        for col in range(w_dst):
            y = row / scale
            x = col / scale

            # m, n: 기존 이미지의 픽셀 좌표(반올림)
            m = int(y + 0.5)
            n = int(x + 0.5)
            
            if m >= h:
                m = h - 1
            
            if n >= w:
                n = w - 1
            
            dst[row, col] = src[m, n]

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('[20191780 Yuk Jung Hun]original', src)

    scale = 3
    #이미지 크기 ??x??로 변경
    my_dst_mini = my_bilinear(src, 1/scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, scale)
    my_dst = my_dst.astype(np.uint8)

    # 출력 윈도우에 학번과 이름을 써주시기 바립니다.
    cv2.imshow('[20191780 Yuk Jung Hun]original', src)
    cv2.imshow('[20191780 Yuk Jung Hun]my bilinear mini', my_dst_mini)
    cv2.imshow('[20191780 Yuk Jung Hun]my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()