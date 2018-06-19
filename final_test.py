import cv2
import numpy as np
import math 
import os

def findEdge(img):
    # houghlines
    edges = cv2.Canny(img,10,200)
    sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    ret,edges = cv2.threshold(sobelx8u,127,255,cv2.THRESH_BINARY)
    plt.subplot(1,1,1), plt.imshow(edges, 'gray')
    plt.show()

    def inbound(edges, x, y):
        if (x >= 0 and 
            y >= 0 and 
            x < edges.shape[0] and 
            y < edges.shape[1]):
            return true
        else:
            return false

    # prepare - blur 
    newedges = edges.copy()
    for i in range(5, edges.shape[0]-5):
        for j in range(5, edges.shape[1]-5):
            if (edges[i][j] > 0 or 
                edges[i][j-1] > 0 or 
                edges[i][j+1] > 0):
                newedges[i][j] = 255
            if (edges[i][j] > 0 or 
                edges[i-1][j] > 0 or 
                edges[i+1][j] > 0):
                newedges[i][j] = 255
            if (edges[i][j] > 0 or 
                edges[i-1][j-1] > 0 or 
                edges[i+1][j+1] > 0):
                newedges[i][j] = 255
            if (edges[i][j] > 0 or 
                edges[i-1][j+1] > 0 or 
                edges[i+1][j-1] > 0):
                newedges[i][j] = 255
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 50
    max_line_gap = 10
    lines = cv2.HoughLinesP(edges, rho, theta, 
                            threshold, np.array([]), 
                            min_line_length, max_line_gap)
    if hasattr(lines, "__len__") == False:
        return
    img2 = img_.copy()
    print len(lines)
    for i in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img2,(x1,y1),(x2,y2),(0,min(255, 10 * i),0),1)

def captch_ex(img, img2gray): # get components 
    img_final = img.copy()
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation
    newImg, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                                            cv2.CHAIN_APPROX_NONE)  # get contours

    return contours


EPS = 0
RECT_TIMES = 2

class TreeLink:
    val = -1 
    link = []
    def __init__(self, val):
        self.val = val
        self.link = [] 
    def add(self, link):
        self.link.append(link)


def check_relation(rec1, rec2):
    if (rec1[0] >= rec2[0] and rec1[0] + rec1[2] <= rec2[0] + rec2[2] and 
        rec1[1] >= rec2[1] and rec1[3] + rec1[1] <= rec2[3] + rec2[1] and 
        rec2[2] * 1.0 / rec2[3] > RECT_TIMES):
        return 0; 
        # rect 2 is sequence 
    if (rec1[0] >= EPS + rec2[0] and 
        rec1[0] + rec1[2] + EPS <= rec2[0] + rec2[2] and 
        rec1[1] >= rec2[1] + EPS and 
        rec1[3] + rec1[1] + EPS <= rec2[3] + rec2[1]):
        return 1; 
        # rect1 in rect2 
    if (rec2[0] >= rec1[0] and rec2[2] + rec2[0] <= rec1[2] + rec1[0] and 
        rec2[1] >= rec1[1] and rec2[3] + rec2[1] <= rec1[3] + rec1[1] and 
        rec1[2] * 1.0 / rec1[3] > RECT_TIMES):
        return 0;
        # rect1 is sequence 
    if (rec2[0] >= rec1[0] + EPS and 
        rec2[2] + rec2[0] + EPS <= rec1[2] + rec1[0] and 
        rec2[1] >= rec1[1] + EPS and 
        rec2[3] + rec2[1] + EPS <= rec1[3] + rec1[1]):
        return -1;
        # rect2 in rect1
    return 0;
    # not related  

def remove_redundant(contours): 
    listTree = []
    n = len(contours)
    for i in range(n):
        listTree.append(TreeLink(i))
    for i in range(n):
        for j in range(i):
            relation = check_relation(contours[i], contours[j])
            if (relation == 1):
                listTree[j].add(i)
            elif (relation == -1):
                listTree[i].add(j)
    new_list = [] 
    for i in range(n):
        if (len(listTree[i].link) == 0 and contours[i][2] >= contours[i][3] * 2.0 / 3 ):
            new_list.append(contours[i])
    return new_list
    
def HarrisCorner(img):
    dst = cv2.cornerHarris(img,2,3,0.04)
    img[dst>0.01*dst.max()]=[0,0,255]


# erc1 = cv2.text.loadClassifierNM1('./trained_classifierNM1.xml')
# er1 = cv2.text.createERFilterNM1(erc1,50,0.00015,0.5,0.8,True,0.3)

# erc2 = cv2.text.loadClassifierNM2('./trained_classifierNM2.xml')
# er2 = cv2.text.createERFilterNM2(erc2,0.8) 

def solve(file_name):
    img = cv2.imread('./Dataset/' + file_name)
    img_ = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = gray.copy()
    contours = []
    index = 0
    for k in range(0, 14):
        print k
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if (gray[i][j] < (k + 1) * 20 and gray[i][j] >= k * 20):
                    dst[i][j] = 255
                else:
                    dst[i][j] = 0
        list_contour = captch_ex(img, dst)
        for contour in list_contour:
            [x, y, w, h] = cv2.boundingRect(contour)
            if (w < gray.shape[1] / 20 or h < gray.shape[0] / 30):
                continue 
            cropped = dst[y :y +  h , x : x + w]
            harris = cv2.cornerHarris(cropped,2,3,0.04)
            count_255 = 0 
            for i in range(h):
                for j in range(w):
                    if (cropped[i][j] > 0):
                        count_255 += 1
            # count_boundary = 0
            # for i in range(h / 10):
            #     for j in range(w):
            #         if (cropped[i][j] > 0):
            #             count_boundary += 1
            #         if (cropped[cropped.shape[0] - i - 1][j] > 0):
            #             count_boundary += 1
            # for j in range(w / 10):
            #     for i in range(h):
            #         if (cropped[i][j] > 0):
            #             count_boundary += 1
            #         if (cropped[i][cropped.shape[1] - j - 1] > 0):
            #             count_boundary += 1
            # count_corner = 0
            # for i in range(h):
            #     for j in range(w):
            #         if (harris[i][j] > harris.max() * 0.001):
            #             cv2.circle(cropped,(j, i), 3, 255, -1)
            #             count_corner += 1
            
            if (count_255 * 1.0 / (w * h)  > 0.05 and count_255 * 1.0 / (w * h) < 0.4):
                
                # test use model of textdetection - opencv 
                # regions = cv2.text.detectRegions(gray[y :y +  h , x : x + w],er1,er2)
                # print len(regions)
                # if len(regions) > 0:
                #     # rects = cv2.text.erGrouping(img[y :y +  h , x : x + w],cropped,[r.tolist() for r in regions])
                #     # #rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

                #     # #Visualization
                #     # for r in range(0,np.shape(rects)[0]):
                #     #     rect = rects[r]
                #     #     cv2.rectangle(cropped, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
                #     #     cv2.rectangle(cropped, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
        
                contours.append(cv2.boundingRect(contour))
                index = index + 1


    # for contour in contours:
    #     index += 1
    #     [x, y, w, h] = contour
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)       

    contours = remove_redundant(contours) 
    for contour in contours:
        index += 1
        [x, y, w, h] = contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) 
        
    cv2.imwrite("./Result/" + file_name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


for fn in os.listdir('./Dataset/'):
    solve(fn)