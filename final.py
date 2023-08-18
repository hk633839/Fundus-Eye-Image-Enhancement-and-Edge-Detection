import cv2 as cv
import numpy as np

#function to resize the frame of image and the image was scaled to 25%
def rescaleframe(frame,scale=0.25):
	w=int(frame.shape[1]*scale)
	h=int(frame.shape[0]*scale)
	dimensions=(w,h) 
	return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

#declaration of kernels for dilation and erosion

kernel1=np.ones((1,1),np.uint8)
kernel2=np.ones((2,2),np.uint8)

#reading of an image

img=cv.imread("fundus.jpeg")

#resizing the image

img=rescaleframe(img)

#adding noise with blur filter to avoid of detecting unwated minor egges

img=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow("gray",img)

# Normalize the image to enhance the contrast
img_norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
#cv.imshow("norm",img_norm)

#using adaptive threshold to highlight the edges
img_thresh = cv.adaptiveThreshold(img_norm, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2)
img_thresh = cv.bitwise_not(img_thresh)

#cv.imshow("thres",img_thresh)

#canny filter detectes the edges
canny=cv.Canny(img_thresh,30,32)
#cv.imshow("canny",canny)

#dilating the image with kernal size (1,1) with 3 iterations

dilation=cv.dilate(img_thresh,kernel1,iterations=3)

#cv.imshow("dilate1",dilation)
#applying erosion on the image with kernal size (2,2) with 1 iteration

erosion=cv.erode(dilation,kernel2,iterations=1)

#dilating the image with kernal size (2,2) with 1 iterations
dilation=cv.dilate(erosion,kernel2,iterations=1)

cv.imshow("dilate",dilation)
cv.imshow("erode",erosion)
output = cv.hconcat([erosion, dilation])
#cv.imshow("out",output)
cv.imwrite("final_output.jpeg",output)

cv.waitKey(0)



