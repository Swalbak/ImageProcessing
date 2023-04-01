import numpy as np
import cv2

def my_imsi(src, scale):
    
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
            m = int(y)
            n = int(x)
            
            if m >= h:
                m = h - 1
            elif n >= w:
                n = w - 1
            
            dst[row, col] = src[m, n]

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 3
    #이미지 크기 ??x??로 변경
    my_dst_mini = my_imsi(src, 1/scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_imsi(my_dst_mini, scale)
    my_dst = my_dst.astype(np.uint8)

    # 출력 윈도우에 학번과 이름을 써주시기 바립니다.
    cv2.imshow('[20191780 육정훈]original', src)
    cv2.imshow('[20191780 육정훈]my bilinear mini', my_dst_mini)
    cv2.imshow('[20191780 육정훈]my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()