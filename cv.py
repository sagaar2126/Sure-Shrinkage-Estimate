import cv2
import skimage.io 

image=skimage.io.imread('bike.jpg')
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()