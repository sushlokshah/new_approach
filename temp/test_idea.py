import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def downscale(img,laplacian, window_size):
    img2 = np.zeros((img.shape[0]//2,img.shape[1]//2))
    window = cv.getGaussianKernel(window_size, 0.3*((window_size-1)*0.5 - 1) + 0.8)
    window = np.exp(window) / np.sum(np.exp(window))
    window = 1
    img_padded = np.zeros((img.shape[0]+window_size//2 -1,img.shape[1]+window_size//2-1))
    img_padded[0:img.shape[0],0:img.shape[1]] = img
    
    laplacian_padded = np.zeros((img.shape[0]+window_size//2 -1,img.shape[1]+window_size//2 -1))
    laplacian_padded[0:img.shape[0],0:img.shape[1]] = abs(laplacian)
    for i in range(img.shape[0]//2):
        for j in range(img.shape[1]//2):
            img2[i][j] = np.sum(img_padded[2*i:2*i+window_size//2,2*j:2*j+window_size//2]*(np.exp(window*laplacian_padded[2*i:2*i+window_size//2,2*j:2*j+window_size//2])) / np.sum(np.exp(window*laplacian_padded[2*i:2*i+window_size//2,2*j:2*j+window_size//2])))
    
    return img2

#scale1
img = cv.imread('temp/00000.png', cv.IMREAD_GRAYSCALE)
laplacian = cv.Laplacian(img,cv.CV_64F)

#scale2
img2 = cv.resize(img, (img.shape[1]//2,img.shape[0]//2), interpolation = cv.INTER_AREA)
laplacian2 = cv.Laplacian(img2,cv.CV_64F)
new_img2 = downscale(img,laplacian,3)
new_laplacian2 = cv.Laplacian(new_img2,cv.CV_64F)


#scale3
img3 = cv.resize(img2, (img2.shape[1]//2,img2.shape[0]//2), interpolation = cv.INTER_AREA)
laplacian3 = cv.Laplacian(img3,cv.CV_64F)
new_img3 = downscale(new_img2,new_laplacian2,3)
new_laplacian3 = cv.Laplacian(new_img3,cv.CV_64F)

#scale4
img4 = cv.resize(img3, (img3.shape[1]//2,img3.shape[0]//2), interpolation = cv.INTER_AREA)
laplacian4 = cv.Laplacian(img4,cv.CV_64F)
new_img4 = downscale(new_img3,new_laplacian3,3)
new_laplacian4 = cv.Laplacian(new_img4,cv.CV_64F)

#scale5
img5 = cv.resize(img4, (img4.shape[1]//2,img4.shape[0]//2), interpolation = cv.INTER_AREA)
laplacian5 = cv.Laplacian(img5,cv.CV_64F)
new_img5 = downscale(new_img4,new_laplacian4,3)
new_laplacian5 = cv.Laplacian(new_img5,cv.CV_64F)

#scale6
img6 = cv.resize(img5, (img5.shape[1]//2,img5.shape[0]//2), interpolation = cv.INTER_AREA)
laplacian6 = cv.Laplacian(img6,cv.CV_64F)
new_img6 = downscale(new_img5,new_laplacian5,3)
new_laplacian6 = cv.Laplacian(new_img6,cv.CV_64F)




print(img.shape,laplacian.shape, laplacian.max(), laplacian.min())
fig, ax = plt.subplots(4,4)
ax[0][0].imshow(img[40:190,600:900], cmap='gray')
ax[0][1].imshow(abs(laplacian)[40:190,600:900], cmap='gray')
ax[0][2].imshow(img[40:190,600:900], cmap='gray')
ax[0][3].imshow(abs(laplacian)[40:190,600:900], cmap='gray')

ax[1][0].imshow(img2[20:95,300:450], cmap='gray')
ax[1][1].imshow(abs(laplacian2)[20:95,300:450], cmap='gray')
ax[1][2].imshow((0.8*new_img2 + 0.2*img2)[20:95,300:450] , cmap='gray')
ax[1][3].imshow(abs(new_laplacian2)[20:95,300:450], cmap='gray')

ax[2][0].imshow(img3[10:47,150:225], cmap='gray')
ax[2][1].imshow(abs(laplacian3)[10:47,150:225], cmap='gray')
ax[2][2].imshow((0.8*new_img3 + 0.2*img3)[10:47,150:225], cmap='gray')
ax[2][3].imshow(abs(new_laplacian3)[10:47,150:225], cmap='gray')

ax[3][0].imshow(img4[5:23,75:112], cmap='gray')
ax[3][1].imshow(abs(laplacian4)[5:23,75:112], cmap='gray')
ax[3][2].imshow((0.8*new_img4 + 0.2*img4)[5:23,75:112], cmap='gray')
ax[3][3].imshow(abs(new_laplacian4)[5:23,75:112], cmap='gray')

#ax[4][0].imshow(img5, cmap='gray')
#ax[4][1].imshow(abs(laplacian5), cmap='gray')
#ax[4][2].imshow(new_img5, cmap='gray')
#ax[4][3].imshow(abs(new_laplacian5), cmap='gray')

#ax[5][0].imshow(img6, cmap='gray')
#ax[5][1].imshow(abs(laplacian6), cmap='gray')
#ax[5][2].imshow(new_img6, cmap='gray')
#ax[5][3].imshow(abs(new_laplacian6), cmap='gray')

#ax[0][1].imshow(abs(laplacian), cmap='gray')
plt.show()

    