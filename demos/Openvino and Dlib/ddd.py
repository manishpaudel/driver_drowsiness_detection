import cv2 as cv
# from inference import preprocessing
from imutils import face_utils
import dlib
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import pygame


dlib_model = "shape_predictor_68_face_landmarks.dat"
	


#for dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model)

	

def hisEqulColor(img):
	hls=cv.cvtColor(img,cv.COLOR_BGR2HLS)
	channels=cv.split(hls)
	# print(np.mean(channels[0]))
	if(np.mean(channels[1] ) < 127):
		# clahe = cv.createCLAHE(clipLimit=16.0,tileGridSize=(8,8))
		# channels[1] = clahe.apply(channels[1])
		cv.equalizeHist(channels[1],channels[1])
		cv.merge(channels,hls)
		cv.cvtColor(hls,cv.COLOR_HLS2BGR,img)
		# print("after equ "+str(np.mean(cv.split(yuv)[0])))
	
	return img
	
	


def main():
	
	total_count = 0

	#for sound
	pygame.mixer.init()
	pygame.mixer.set_num_channels(8)
	voice = pygame.mixer.Channel(5)

	sound = pygame.mixer.Sound("warn.wav")

	# if(cap.isOpened()==False):
	# 	print("error while streaming")
	#eye counters
	time = datetime.now()
	one_min_start_time = time
	eye_closed_counter = 0
	eye_closed = False
	eye_close_time = time

	#mouth counters
	mouth_open_time = time
	mouth_open = False
	yawn_count = 0

	#nod counter
	nod_time = time
	nodding = False
	nod_count = 0

	m_EAR_left, m_EAR_right = get_mEARS()
	cap = cv.VideoCapture(0)
	while True:
		_, frame = cap.read()
		#for mirror image
		frame = cv.flip(frame, 1)
		#print(frame.shape)

		frame = hisEqulColor(frame)
		
		
		
		#cv.putText(frame,"is doing",(100,100),cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
		# preprocessed_image = preprocessing(frame, h, w)
		
		
		#setting bounding box color, image height and width
		color = (0,0,255)
		height = frame.shape[0]
		width = frame.shape[1]

		
		#incase no face detected, crop will be unresolved variable
		crop = frame[0:1, 0:1]
		#iterating over the detected boxes
		

		

		# processed_crop = preprocessing(crop, hh, ww)
		# new_res = sync_inference(exec_net_headpose, image = processed_crop)

		# yaw = new_res['angle_y_fc'][0]
		# pitch = new_res['angle_p_fc'][0]
		# roll = new_res['angle_r_fc'][0]

		# cv.putText(frame,"yaw: {}".format(str(yaw)), (20,30),cv.FONT_HERSHEY_COMPLEX, 1, color,1)
		# cv.putText(frame,"pitch: {}".format(str(pitch)), (20,60),cv.FONT_HERSHEY_COMPLEX, 1, color,1)
		# cv.putText(frame,"roll: {}".format(str(roll)), (20,90),cv.FONT_HERSHEY_COMPLEX, 1, color,1)

		# if pitch > -15 and pitch < 25:
		# 	cv.putText(frame,"straight face", (20,60),cv.FONT_HERSHEY_COMPLEX, 1, color,1)
		# 	nodding = False


		# elif pitch > 25:
		# 	cv.putText(frame,"facing downwards", (20,60),cv.FONT_HERSHEY_COMPLEX, 1, color,1)
		# 	if not nodding:
		# 		nodding = True
		# 		nod_time = time.now()

		# 	if nodding:
		# 		if datetime.now() - nod_time >= timedelta(seconds = 2):
		# 			nod_count += 1
		# 			print("nod count= "+ str(nod_count))
		# 			nod_time = time.now()
		# 			if voice.get_busy() == 0:
		# 				voice.play(sound)


		# elif pitch < -15:
		# 	cv.putText(frame,"facing upwards", (20,60),cv.FONT_HERSHEY_COMPLEX, 1, color,1)
		# 	nodding = False
		

		#feed frame face to dlib
		# print(crop.shape)
		nod_count = 0
		crop_dlib = cv.resize(frame, (300,300))
		gray = cv.cvtColor(crop_dlib, cv.COLOR_BGR2GRAY)
		clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
		gray = clahe.apply(gray)
		 
		rects = detector(gray,0)
		for (i, rect) in enumerate(rects):
			xmin = rect.left()
			ymin = rect.top()
			xmax = rect.right()
			ymax = rect.bottom()

			cv.rectangle(gray,(xmin,ymin),(xmax,ymax),color,1)
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			#EAR

			ear_l = ((shape[41][1]-shape[37][1]) + (shape[40][1]-shape[38][1]))/(2*(shape[39][0]-shape[36][0]))
		

			ear_r = ((shape[47][1]-shape[43][1]) + (shape[46][1]-shape[44][1]))/(2*(shape[45][0]-shape[42][0]))

			

			if(ear_l < m_EAR_left and ear_r < m_EAR_right):
				cv.putText(frame, "closed", (100,80),cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

				#if eye closed for first frame set the eye close time(eye was not closed in last frame)
				if not eye_closed:
					eye_close_time = datetime.now()

				#if eye closed for more than 2 sec straight
				if eye_closed and (datetime.now() - eye_close_time >= timedelta(seconds = 2)):
					if voice.get_busy() == 0:
						voice.play(sound) 
					
				eye_closed = True

				
			else:
				cv.putText(frame, "open", (100,80),cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

				#previous frame was eye closed and now eye opened, increase counter
				if eye_closed:
					eye_closed_counter += 1
					print("eye_closed_counter = "+ str(eye_closed_counter))
				eye_closed = False


			#mouth ratio

			mouth_ratio = ((shape[58][1] - shape[50][1]) + (shape[56][1] - shape[52][1])) / (2*(shape[54][0] - shape[48][0]))

			for (x,y) in shape:
				cv.circle(gray, (x,y), 2, color, -1)

			if(mouth_ratio>0.35):
				cv.putText(frame,"Mouth open", (20,90),cv.FONT_HERSHEY_COMPLEX, 1, color, 1)
				#if previous frame was closed, then first frame to open mouth note time
				if not mouth_open:
					mouth_open_time = datetime.now()
					mouth_open = True
					#print(mouth_open_time)
				#check if more than 4.5 sec opened
				if mouth_open and (datetime.now() - mouth_open_time > timedelta(seconds = 4.5)):
					print(datetime.now() - mouth_open_time)
					mouth_open_time = datetime.now()
					yawn_count += 1
					print("yawn count= "+ str(yawn_count))
					if voice.get_busy() == 0:
						voice.play(sound)

			else:
				cv.putText(frame,"Mouth close", (20,90),cv.FONT_HERSHEY_COMPLEX, 1, color, 1)
				if mouth_open:
					mouth_open = False

		#check 1 min timer:
		if(datetime.now() - one_min_start_time >= timedelta(minutes = 1)):
			
			if eye_closed_counter > 25 or eye_closed_counter < 10:
				total_count += 1
				if(voice.get_busy() == 0):
					voice.play(sound)


			total_count += yawn_count + nod_count
			print("drowsy_count = "+ str(total_count))
			print("total eye_closed_counter= "+ str(eye_closed_counter))
			print("total yawn = " + str(yawn_count))
			print("total nod_count = " + str(nod_count))

			if total_count >= 3:
				voice.play(sound)


			one_min_start_time = datetime.now()
			print(eye_closed_counter)
			yawn_count = 0
			eye_closed_counter = 0
			nod_count = 0

		cv.imshow("Face detection", frame)
		cv.imshow("Cropped face", gray)
		
		#to stop
		interrupt = cv.waitKey(10)
		if interrupt & 0xFF == 27:
			break
	cap.release()
	cv.destroyAllWindows()



def output_processing(result, image, threshold = 0.5):
	#setting bounding box color, image height and width
	color = (0,0,255)
	height = image.shape[0]
	width = image.shape[1]

	#we get output blob as [1*1*N*7] that means result[0][0][i] will give result of i-th box.
	#Each box has 7 numbers (last digit(7) of the output blob). 
	#3rd index numbered 2 gives the confidence
	#4th, 5th, 6th, 7th indices numbered 3,4,5,6 resp. give xmin, ymin, xmax, ymax

	#iterating over the detected boxes
	for i in range(len(result[0][0])):
		box = result[0][0][i]	#result of i-th box
		confidence = box[2]
		if confidence > threshold:
			xmin = int(box[3] * width)
			ymin = int(box[4] * height)
			xmax = int(box[5] * width)
			ymax = int(box[6] * height)

			#Drawing the box with image
			cv.rectangle(image,(xmin,ymin),(xmax,ymax),color,1)

	return image


def get_mEARS():
	cap = cv.VideoCapture(0)
	start_time = datetime.now()

	l_Amin = 100
	l_Amax = 0
	l_Bmin = 100
	l_Bmax = 0
	l_Cmin = 100
	l_Cmax = 0
	r_Amin = 100
	r_Amax = 0
	r_Bmin = 100
	r_Bmax = 0
	r_Cmin = 100
	r_Cmax = 0

	while True:
		_, frame = cap.read()
		#for mirror image
		frame = cv.flip(frame, 1)
		
		
		# preprocessed_image = preprocessing(frame, h, w)
		# result = sync_inference(exec_net, image = preprocessed_image)
		
		# result = result['detection_out']
		
		
		#setting bounding box color, image height and width
		color = (0,0,255)
		height = frame.shape[0]
		width = frame.shape[1]

		crop = frame[0:1, 0:1]
		#iterating over the detected boxes
		# for i in range(len(result[0][0])):
		# 	box = result[0][0][i]	#result of i-th box
		# 	confidence = box[2]
		# 	if confidence > 0.5:
		# 		xmin = int(box[3] * width)
		# 		ymin = int(box[4] * height)
		# 		xmax = int(box[5] * width)
		# 		ymax = int(box[6] * height)
		# 		#print(str(xmin)+", "+str(ymin)+", "+str(xmax)+", "+str(ymax))

		# 		#Drawing the box with image
		# 		cv.rectangle(frame,(xmin,ymin),(xmax,ymax),color,1)
		# 		#cv.putText(frame, str(i), (50,50),cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
		# 		if xmin<0:
		# 			xmin = 0
		# 		if ymin<0:
		# 			ymin = 0
		# 		crop = frame[ymin-0:ymax+30, xmin-30:xmax+30]
				


		

		#feed frame face to dlib
		# print(crop.shape)
		crop_dlib = cv.resize(frame, (300,300))
		gray = cv.cvtColor(crop_dlib, cv.COLOR_BGR2GRAY)
		clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
		gray = clahe.apply(gray)
		 
		rects = detector(gray,0)
		for (i, rect) in enumerate(rects):
			xmin = rect.left()
			ymin = rect.top()
			xmax = rect.right()
			ymax = rect.bottom()

			cv.rectangle(gray,(xmin,ymin),(xmax,ymax),color,1)
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			#EAR

			ear_l = ((shape[41][1]-shape[37][1]) + (shape[40][1]-shape[38][1]))/(2*(shape[39][0]-shape[36][0]))
		

			ear_r = ((shape[47][1]-shape[43][1]) + (shape[46][1]-shape[44][1]))/(2*(shape[45][0]-shape[42][0]))

			
			left_A = shape[41][1]-shape[37][1]
		
			left_B = shape[40][1]-shape[38][1]
			
			left_C = shape[39][0]-shape[36][0]
			
			right_A = shape[47][1]-shape[43][1]
			
			right_B = shape[46][1]-shape[44][1]
			
			right_C = shape[45][0]-shape[42][0]

			#left
			# print(left_A)
			# print(l_Amin)
			if(left_A < l_Amin):
				l_Amin = left_A


			if(left_A > l_Amax):
				l_Amax = left_A

			if(left_B < l_Bmin):
				l_Bmin = left_B

			if(left_B > l_Bmax):
				l_Bmax = left_B

			if(left_C < l_Cmin):
				l_Cmin = left_C

			if(left_C > l_Cmax):
				l_Cmax = left_C

			#right
			if(right_A < r_Amin):
				r_Amin = right_A

			if(right_A > r_Amax):
				r_Amax = right_A

			if(right_B < r_Bmin):
				r_Bmin = right_B

			if(right_B > r_Bmax):
				r_Bmax = right_B

			if(right_C < r_Cmin):
				r_Cmin = right_C

			if(right_C > r_Cmax):
				r_Cmax = right_C



			cv.putText(frame, "l_Amin\tl_Amax\tl_Bmin\tl_Bmax\tl_Cmin\tl_Cmax", (100,50),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
			cv.putText(frame, str(l_Amin)+"\t"+str(l_Amax)+"\t"+str(l_Bmin)+"\t"+str(l_Bmax)+"\t"
				+str(l_Cmin)+"\t"+str(l_Cmax), (100,65),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

			ear_close_left = (l_Amin + l_Bmin) / (2 * l_Cmax)
			ear_open_left = (l_Amax + l_Bmax) / (2 * l_Cmin)
			m_EAR_left = (ear_open_left - ear_close_left) / 2

			ear_close_right = (r_Amin + r_Bmin) / (2 * r_Cmax)
			ear_open_right = (r_Amax + r_Bmax) / (2 * r_Cmin)
			m_EAR_right = (ear_open_right - ear_close_right) / 2

			if(ear_l < m_EAR_left and ear_r < m_EAR_right):
				cv.putText(frame, "closed", (100,80),cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
				
			else:
				cv.putText(frame, "open", (100,80),cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255))


			for (x,y) in shape:
				cv.circle(gray, (x,y), 2, color, -1)


		#cv.imshow("Face detection", frame)
		cv.imshow("Cropped face", gray)

		
		#to stop
		interrupt = cv.waitKey(10)
		if interrupt & 0xFF == 27:
			break
		if(datetime.now() - start_time > timedelta(seconds=10)):
			break
	cap.release()
	cv.destroyAllWindows()
	print(m_EAR_left)
	print(m_EAR_right)
	return m_EAR_left, m_EAR_right


if __name__ == '__main__':
	
	
	main()
	
	
	