
# coding: utf-8

# In[83]:


import cv2
import numpy as np

from matplotlib import pyplot as plt
img = cv2.imread('image3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_ = img.copy()
plt.subplot(1,1,1), plt.imshow(gray, 'gray')
plt.show()


# In[84]:


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


# In[85]:


dst = dst.copy()
for k in range(10, 14):
    print k
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if (gray[i][j] < (k + 1) * 20 and gray[i][j] >= k * 20):
                dst[i][j] = 255
            else:
                dst[i][j] = 0
    findEdge(dst)

