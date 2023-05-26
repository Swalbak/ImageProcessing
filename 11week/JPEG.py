import numpy as np
import cv2
import time

def Quantization_Luminance(scale_factor):
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance * scale_factor

def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################

    h, w = src.shape
    
    if h%n!=0:
        h_pad = n - h%n
    else:
        h_pad = 0
    
    if w%n != 0:
        w_pad = n - w%n
    else:
        w_pad = 0
    
    dst = np.zeros((h+h_pad, w+w_pad))
    dst[:h, :w] = src

    blocks = []

    for row in range(0, h + h_pad, n):
        for col in range(0, w + w_pad, n):
            block = dst[row : row + n, col:col+n]
            blocks.append(block)

    return np.array(blocks)

def C(w, n = 8):
    if w == 0:
        return (1/n) ** 0.5
    else:
        return (2/n) ** 0.5


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    # 4중 for문으로 구현 시 감점 예정          #
    ######################################
    v = np.arange(n)
    u = np.arange(n)
    X, U = np.meshgrid(v, u)

    w = np.sqrt(2 / n) * np.cos(np.pi * (2*X + 1) * U / (2 * n))
    w[0, :] /= np.sqrt(2)
    dct = np.dot(w, np.dot(block, w.T))

    return np.round(dct)

def my_zigzag_scanning(block, mode='encoding', block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_scanning 완성             #
    ######################################
    row = 0
    col = 0
    index = 0
    half = True
    zigzag = []
    block_recover = np.zeros((block_size, block_size))
    encoding = True
    a = 1

    if mode == 'decoding':
        encoding = False

    while True:
       
        if encoding:
            zigzag.append(block[row][col])
        else:
            if index >= len(block) or block[index] == 'EOB':
                break
            block_recover[row][col] = block[index]
            index += 1

        if row == block_size-1 and col == block_size-1:
            break

        if half: 
            if a == 0:
                if col == 0:
                    row += 1
                    a = 1
                else: 
                    row += 1
                    col -= 1
            else:
                if row == 0:
                    col += 1
                    a = 0
                else: 
                    row -= 1
                    col += 1
        else: 
            if a == 0:
                if row == block_size - 1:
                    col += 1
                    a = 1
                else: 
                    row += 1
                    col -= 1
            else:
                if col == block_size - 1:
                    row += 1
                    a = 0
                else:  
                    row -= 1
                    col += 1
        if row == block_size - 1:
            half = False
    if encoding:
        for index, val in reversed(list(enumerate(zigzag))):
            if val == 0:
                del zigzag[index]
            else:
                zigzag.append('EOB')
                break

        return zigzag
    else: 
        return block_recover

def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    x = np.arange(n)
    y = np.arange(n)
    X, U = np.meshgrid(x, y)

    w = np.sqrt(2 / n) * np.cos(np.pi * (2*X + 1) * U / (2 * n))
    w[0, :] /= np.sqrt(2)

    # Compute inverse DCT
    idct = np.dot(w.T, np.dot(block, w))

    return np.round(idct)

def block2img(blocks, src_shape, n = 8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    (h, w) = src_shape

    if h % n != 0:
        h_pad = n - h%n
    else:
        h_pad = 0

    if w % n != 0:
        w_pad = n - w%n
    else:
        w_pad = 0
    dst = np.zeros((h+h_pad, w+w_pad), dtype=np.uint8)
    index = 0

    for row in range(int(h/8)):
        for col in range(int(w/8)):
            dst[8*row:8*(row+1), 8*col:8*(col+1)] = blocks[index]
            index += 1

    return dst

def Encoding(src, n=8,scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)
    print("block = \n",src[150:158,89:97])


    #subtract 128
    blocks -= 128
    b = np.double(src[150:158,89:97])-128
    print("b = \n",b)

    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    # print DCT results
    bd = DCT(b,n=8)
    print("bd = \n",bd)


    #Quantization + thresholding
    Q = Quantization_Luminance(scale_factor)
    QnT = np.round(blocks_dct / Q)
    #print Quantization results
    bq = bd  / Q
    print("bq = \n",bq)

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))

    return zz, src.shape, bq

def Decoding(zigzag, src_shape,bq, n=8,scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)


    # Denormalizing
    Q = Quantization_Luminance(scale_factor=scale_factor)
    blocks = blocks * Q
    # print results Block * Q
    bq2 = bq * Q
    print("bq2 = \n",bq2)

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    #print IDCT results
    bd2 = DCT_inv(bq2,n=8)
    print("bd2 = \n",bd2)

    # add 128
    blocks_idct += 128

    # print block value
    b2 = np.round(bd2 + 128)
    print("b2 = \n",b2)

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst, b2

def main():
    scale_factor = 0
    start = time.time()
    src = cv2.imread('caribou.tif', cv2.IMREAD_GRAYSCALE)

    comp, src_shape,bq = Encoding(src, n=8, scale_factor=scale_factor)
    comp = np.array(comp, dtype=object) 
    np.save('comp.npy', comp)
    np.save('src_shape.npy', src_shape)
    # print(comp)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')
    recover_img, b2 = Decoding(comp, src_shape, bq,n=8,scale_factor=scale_factor)
    print("scale_factor : ",scale_factor,"differences between original and reconstructed = \n",src[150:158,89:97]-b2)
    # print(recover_img)
    total_time = time.time() - start
    #
    print('time : ', total_time)
    if total_time > 12:
        print('감점 예정입니다.')
    print(recover_img.shape)
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
