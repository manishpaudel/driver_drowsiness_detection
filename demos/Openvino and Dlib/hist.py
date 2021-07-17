import cv2
import numpy as np
import math

def equalizeHist(img):
	hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
	channels=cv2.split(hls)
	# print(np.mean(channels[0]))
	if(np.mean(channels[1] ) < 127):
		# clahe = cv2.createCLAHE(clipLimit=16.0,tileGridSize=(8,8))
		# channels[1] = clahe.apply(channels[1])
		cv2.equalizeHist(channels[1],channels[1])
		cv2.merge(channels,hls)
		cv2.cvtColor(hls,cv2.COLOR_HLS2BGR,img)
	
	cv2.imwrite("hsi-harsh.jpg",hls)


if __name__ == '__main__':
	img = cv2.imread("harsh.jpg")
	equalizeHist(img)