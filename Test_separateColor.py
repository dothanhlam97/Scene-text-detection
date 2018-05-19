
# coding: utf-8

# In[92]:


import cv2
import numpy as np

from matplotlib import pyplot as plt
img = cv2.imread('image5.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_ = img.copy()
plt.subplot(1,1,1), plt.imshow(gray, 'gray')
plt.show()


# In[93]:


import math 

def findEdge(img):
    # houghlines
    edges = cv2.Canny(img,10,200)
    sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    ret,edges = cv2.threshold(sobelx8u,127,255,cv2.THRESH_BINARY)
    plt.subplot(1,1,1), plt.imshow(edges, 'gray')
    plt.show()

    def inbound(edges, x, y):
        if (x >= 0 and y >= 0 and x < edges.shape[0] and y < edges.shape[1]):
            return true
        else:
            return false

    # prepare - blur 
    newedges = edges.copy()
    # print edges.shape
    for i in range(5, edges.shape[0]-5):
        for j in range(5, edges.shape[1]-5):
    #         print 'a'
            if (edges[i][j] > 0 or edges[i][j-1] > 0 or edges[i][j+1] > 0):
                newedges[i][j] = 255
            if (edges[i][j] > 0 or edges[i-1][j] > 0 or edges[i+1][j] > 0):
                newedges[i][j] = 255
            if (edges[i][j] > 0 or edges[i-1][j-1] > 0 or edges[i+1][j+1] > 0):
                newedges[i][j] = 255
            if (edges[i][j] > 0 or edges[i-1][j+1] > 0 or edges[i+1][j-1] > 0):
                newedges[i][j] = 255



    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 50
    max_line_gap = 10
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    if hasattr(lines, "__len__") == False:
#     if (lines == None):
        return
    img2 = img_.copy()
    print len(lines)
    for i in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img2,(x1,y1),(x2,y2),(0,min(255, 10 * i),0),1)

    plt.subplot(1,1,1), plt.imshow(img2)
    plt.show()


# In[94]:


import cv2
from matplotlib import pyplot as plt

contours = []

def captch_ex(img, img2gray):
#     img = cv2.imread(file_name)

#     img_final = cv2.imread(file_name)
    img_final = img.copy()
#     img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img2gray = img.copy()
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    # for cv2.x.x

    newImg, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    # for cv3.x.x comment above line and uncomment line below

    #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    return contours
    



# In[96]:


dst = gray.copy()
for k in range(0, 14):
    print k
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if (gray[i][j] < (k + 1) * 20 and gray[i][j] >= k * 20):
                dst[i][j] = 255
            else:
                dst[i][j] = 0
#     findEdge(dst)
    
    contours += captch_ex(img, dst)
    
    
for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 30 or h < 30:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    
plt.subplot(1,1,1), plt.imshow(img)
plt.show()

