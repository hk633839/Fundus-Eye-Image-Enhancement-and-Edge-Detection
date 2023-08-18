import cv2 as cv
import numpy as np

#function to resize the frame of image and the image was scaled to 25%
def rescaleframe(frame,scale=0.25):
	w=int(frame.shape[1]*scale)
	h=int(frame.shape[0]*scale)
	dimensions=(w,h) 
	return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

#defining the colour boundaries

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

#declaration of kernels for dilation and erosion

kernel1=np.ones((5,5),np.uint8)
kernel2=np.ones((2,2),np.uint8)

#reading of an image

img=cv.imread("fundus.jpeg")

#resizing the image

img=rescaleframe(img)

#adding noise with blur filter to avoid of detecting unwated minor egges

img=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#cv.imshow("hsv",hsv)
mask = cv.inRange(hsv,lower_red, upper_red)
res = cv.bitwise_and(img,img, mask= mask)
#cv.imshow("Normal",img)

#detecting the edges of the fundus retina image using canny edge detection

canny=cv.Canny(img,30,32)

#dilating the image with kernal size (3,3) with 3 iterations

dilation=cv.dilate(canny,kernel2,iterations=3)

#applying erosion on the image with kernal size (3,3) with 2 iterations

erosion=cv.erode(dilation,kernel2,iterations=2)


#cv.imshow("dilate1",dilation)
#dilating the image with kernal size (3,3) with 1 iterations

dilation=cv.dilate(erosion,kernel2,iterations=1)
#cv.imshow("canny",canny)
cv.imshow("dilate",dilation)
#cv.imshow("erode",erosion)
cv.waitKey(0)
