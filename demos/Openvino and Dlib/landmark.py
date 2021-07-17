import cv2
from imutils import face_utils
import dlib
import numpy as np


#pretrained model
model = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

cap = cv2.VideoCapture(0)
color = (255, 0, 0)

# image_path = "smalleyes.jpg"
# image = cv2.imread(image_path)

# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rects = detector(img, 0)

# for (i,rect) in enumerate(rects):
# 	#make prediction and transform to array
# 	shape = predictor(img, rect)
# 	shape = face_utils.shape_to_np(shape)
# 	#print(shape)

# 	#eye aspect ratio
# 	ear_l = ((shape[41][1]-shape[37][1]) + (shape[40][1]-shape[38][1]))/(2*(shape[39][0]-shape[36][0]))
		

# 	ear_r = ((shape[47][1]-shape[43][1]) + (shape[46][1]-shape[44][1]))/(2*(shape[45][0]-shape[42][0]))
# 	#print(str(ear_l)+' '+str(ear_r) )

# 	if(ear_l < 0.15 and ear_r <0.15):
# 		cv2.putText(image, "closed", (10,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
# 	else:
# 		cv2.putText(image, "open", (10,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

# 	#draw circles on image for all coordinates
# 	for(x,y) in shape:
# 		cv2.circle(image, (x,y), 2, color, -1)

# cv2.imwrite(image_path+"-op.jpg", image)

def calculateAngle(point1, point2):
	#corner is point1 and the arc point is point2
	perpn = abs(point1[1] - point2[1])
	base = abs(point2[0] - point1[0])
	angle = np.arctan(perpn/base) # in radian
	return angle

while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	

	#get faces
	rects = detector(gray, 0)


	#for each detected faces,find landmark
	for (i,rect) in enumerate(rects):

		
		print(rect)
		xmin = rect.left()
		ymin = rect.top()
		xmax = rect.right()
		ymax = rect.bottom()

		cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color,1)
		
		#make prediction and transform to array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		#print(shape)

		#eye aspect ratio
		ear_l = ((shape[41][1]-shape[37][1]) + (shape[40][1]-shape[38][1]))/(2*(shape[39][0]-shape[36][0]))
		

		ear_r = ((shape[47][1]-shape[43][1]) + (shape[46][1]-shape[44][1]))/(2*(shape[45][0]-shape[42][0]))
		#print(str(ear_l)+' '+str(ear_r) )

		#angle between lowest and highest point of eye
		angle_l = calculateAngle(shape[36], shape[38])
		angle_r = calculateAngle(shape[42], shape[44])

		if(ear_l < 0.22 and ear_r <0.22):
			cv2.putText(frame, "closed", (10,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
		else:
			cv2.putText(frame, "open", (10,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

		cv2.putText(frame, str(angle_l), (30,90),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
		cv2.putText(frame, str(angle_r), (30,150),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
		#draw circles on image for all coordinates
		for(x,y) in shape:
			cv2.circle(frame, (x,y), 2, color, -1)


	cv2.imshow("Landmarks Detection",frame)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break


cv2.destroyAllWindows()
cap.release()

