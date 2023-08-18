import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('fundus.jpeg')
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.imread('fundus.jpeg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
# Normalize the image to enhance the contrast
img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Apply Gaussian filter to smooth the image

img_smooth = cv2.GaussianBlur(img_norm, (5,5), 0)
plt.imshow(img_smooth)
# Apply thresholding to segment the blood vessels
img_thresh = cv2.adaptiveThreshold(img_smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)

plt.imshow(img_thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
plt.imshow(img_morph)
img=img_morph.copy()
kernel= np.ones((6,6),np.uint8)
img_erode=cv2.erode(img,kernel,iterations = 3)
kernel=kernel = np.ones((5,5),np.uint8)
img=cv2.dilate(img_erode,kernel,iterations =5)
plt.imshow(img)
plt.show() 
img1=img
cv2.imwrite('binary_image_.jpg', img1)
img=img1.copy()
img_edges = cv2.Canny(img, 10, 15)
plt.imshow(img_edges)
plt.show()
kernel= np.ones((10,10),np.uint8)
img_erode=cv2.dilate(img_edges,kernel,iterations = 2)
plt.imshow(img_erode)
plt.show()

kernel=kernel = np.ones((6,6),np.uint8)
img=cv2.erode(img_erode,kernel,iterations = 3)
plt.imshow(img)
plt.show() 
 

cv2.imwrite('binary_image.jpg', img)