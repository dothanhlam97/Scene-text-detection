
# coding: utf-8

# In[71]:


import cv2
import numpy as np

from matplotlib import pyplot as plt

img = cv2.imread('image2.jpg', 0)
# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
edges = cv2.Canny(sobelx8u,0,200)
plt.subplot(2,2,4),plt.imshow(edges,cmap = 'gray')
plt.title('Saaaa'), plt.xticks([]), plt.yticks([])
plt.show()


# In[72]:



ret,thresh1 = cv2.threshold(sobelx8u,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(sobelx8u,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(sobelx8u,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(sobelx8u,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(sobelx8u,127,255,cv2.THRESH_TOZERO_INV)
thresh6 = cv2.Canny(thresh5,0,200)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV', 'Canny']
images = [sobelx8u, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]
for i in xrange(7):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()



# In[73]:


plt.subplot(1,1,1),plt.imshow(sobel_8u, 'gray')
plt.show()
ret,thresh1 = cv2.threshold(sobelx8u,127,255,cv2.THRESH_BINARY)
plt.subplot(1,1,1), plt.imshow(thresh1, 'gray')
plt.show()


# In[89]:


def convertScale(src_array, alpha, beta):
    dst_array = (src_array * alpha + beta).astype(np.uint8)
    return dst_array
#  Detector parameters
blockSize = 2
apertureSize = 3
ksize = 0.04

#   Detecting corners
# cv.CornerHarris(image, harris_dst, blockSize, aperture_size=3, k=0.04)
# cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
dst_harris = cv2.cornerHarris(thresh1, blockSize, 3, 0.04)
dst_harris = cv2.normalize(dst_harris,None,0,255,cv2.NORM_MINMAX)
dst_harris = convertScale(dst_harris, alpha=0.5, beta=6)
#   /// Normalizing
#   normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
#   convertScaleAbs( dst_norm, dst_norm_scaled );

plt.subplot(1,1,1), plt.imshow(dst_harris, 'gray')
plt.show()
ret,thresh_harris = cv2.threshold(dst_harris,127,255,cv2.THRESH_BINARY_INV)
for i in xrange (dst_harris.shape[1]):
    for j in xrange(dst_harris.shape[0]):
#         print dst_harris[j][i]
        if (dst_harris[j][i] > 50):
#             cv2.circle(thresh1, (i, j), 10, (100,100,100))        
            thresh_harris[j][i] = 255
        else:
            thresh_harris[j][i] = 0
# ret,thresh1 = cv2.threshold(dst_harris,127,255,cv2.THRESH_BINARY)

plt.subplot(1,1,1), plt.imshow(thresh_harris, 'gray')
plt.show()


# In[98]:


img2 = img.copy()
for i in xrange (dst_harris.shape[1]):
    for j in xrange(dst_harris.shape[0]):
#         print dst_harris[j][i]
        if (dst_harris[j][i] < 10):
            cv2.circle(img2, (i, j), 2, (0,0,100))        
#             thresh_harris[j][i] = 255
#         else:
#             thresh_harris[j][i] = 0

plt.subplot(1,1,1), plt.imshow(img2, cmap = None)
plt.show()

