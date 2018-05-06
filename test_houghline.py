
# coding: utf-8

# In[15]:


import cv2
import numpy as np

from matplotlib import pyplot as plt



# In[62]:


# houghlines
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,10,150)


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
            
            

minLineLength = 50
maxLineGap = 10
lines = cv2.HoughLinesP(newedges,1,np.pi/180,100,minLineLength,maxLineGap)
img2 = img.copy()
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)

plt.subplot(1,1,1), plt.imshow(edges)
plt.show()
plt.subplot(1,1,1), plt.imshow(newedges)
plt.show()
plt.subplot(1,1,1), plt.imshow(img2)
plt.show()


# In[110]:


rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 400
max_line_gap = 30
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
img2 = img.copy()
for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
    
plt.subplot(1,1,1), plt.imshow(img2)
plt.show()

    

