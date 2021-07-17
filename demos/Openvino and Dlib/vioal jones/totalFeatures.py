import numpy as np
import math

img = np.zeros((24,24))

# print(img.shape[1])
#for feature a
total_a = 0
for i in range(1, img.shape[0]+1):
	for j in range (1, img.shape[1]+1):
		for w in range(1,math.ceil((img.shape[0]+1)/2)):
			for h in range(1,img.shape[1]+1):
				if (i+h-1<=24) and (j+2*w-1<=24):
					total_a += 1
print(total_a)

# for feature b
total_b = 0
for i in range(1, img.shape[0]+1):
	for j in range (1, img.shape[1]+1):
		for w in range(1,math.ceil((img.shape[0]+1)/3)):
			for h in range(1,img.shape[1]+1):
				if (i+h-1<=24) and (j+3*w-1<=24):
					total_b += 1
print(total_b)


#for feature c
total_c = 0
for i in range(1, img.shape[0]+1):
	for j in range (1, img.shape[1]+1):
		for w in range(1,img.shape[1]+1):
			for h in range(1,math.ceil((img.shape[0]+1)/2)):
				if (i+2*h-1<=24) and (j+w-1<=24):
					total_c += 1
print(total_c)

# for feature d
total_d = 0
for i in range(1, img.shape[0]+1):
	for j in range(1,img.shape[1]+1):
		for w in range (1, img.shape[0]+1):
			for h in range(1,math.ceil((img.shape[1]+1)/3)):
				if (i+3*h-1<=24) and (j+w-1<=24):
					total_d += 1
print(total_d)

# for feature e
total_e = 0
for i in range(1, img.shape[0]+1):
	for j in range(1,img.shape[1]+1):
		for w in range(1,math.ceil((img.shape[0]+1)/2)):
			for h in range(1,math.ceil((img.shape[1]+1)/2)):
				if (i+2*h-1<=24) and (j+2*w-1<=24):
					total_e += 1
print(total_e)

total = total_a + total_b + total_c + total_d + total_e

print("The total number of Haar like features are: ", total)