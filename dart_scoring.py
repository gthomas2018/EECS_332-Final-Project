import numpy as np
import cv2 as cv
import os, os.path
import itertools
from shapely.geometry import LineString
from shapely.geometry import Point


video = "center"
circle_props = [6.7/8.5, 6.25/8.5, 4.25/8.5, 3.75/8.5, 0.75/8.5, 0.25/8.5]


def polar_to_cart_line(rho, theta) :
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*a)
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*a)

	return x1, y1, x2, y2


def center_intersect(x1, y1, x2, y2, draw_crop):
	if float(x1 - x2) == 0:
		return None
	m = float(y1 - y2)/float(x1 - x2)
	b = int(y1 - float(m*x1))

	area_buffer = 25
	center = (draw_crop.shape[1]/2, draw_crop.shape[0]/2)
	min_y = center[1] - area_buffer
	max_y = center[1] + area_buffer
	min_x = center[0] - area_buffer
	max_x = center[0] + area_buffer

	cv.rectangle(draw_crop, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

	y = int(m*(center[0]+100) + b)
	if m == 0:
		return None
	x = int((center[1]-b)/m)

	if y in range(min_y, max_y) or x in range(min_x, max_x):
		y = int(m*draw_crop.shape[1] + b)
		
		line = [0, np.clip(0, b, draw_crop.shape[0]), draw_crop.shape[1], np.clip(0, y, draw_crop.shape[0])]
		return line
	else:
		return None


def stream() : 
	cam = cv.VideoCapture(1)
	cam.set(3, 1920)
	cam.set(4, 1080)

	while cam.isOpened():
		ret, img = cam.read()

		img = cv.GaussianBlur(img, (5,5), 0)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		'''
		GET CIRCLES
		'''
		circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=200, param2=200, minRadius=250, maxRadius=10000)
		
		if circles == None:
			continue

		outer_circle = circles[0][0]
		outer_circle_rad = int(outer_circle[2])
		outer_circle_x = int(outer_circle[0])
		outer_circle_y = int(outer_circle[1])

		crop_buffer = 0
		board_min = (outer_circle_x - outer_circle_rad - crop_buffer, outer_circle_y - outer_circle_rad - crop_buffer)
		board_max = (outer_circle_x + outer_circle_rad + crop_buffer, outer_circle_y + outer_circle_rad + crop_buffer)
		board_crop = img[board_min[1] : board_max[1], board_min[0] : board_max[0]]
		gray_crop = gray[board_min[1] : board_max[1], board_min[0] : board_max[0]]
		draw_crop = board_crop.copy()

		outer_circle_x = draw_crop.shape[1]/2
		outer_circle_y = draw_crop.shape[0]/2

		cv.circle(draw_crop, (outer_circle_x, outer_circle_y), outer_circle_rad - 1, (0,255,0), 1)

		for prop in circle_props:
			cv.circle(draw_crop, (outer_circle_x, outer_circle_y), int(outer_circle_rad*prop), (0,255,0), 1)

		'''
		GET LINES
		'''
		edges_crop = cv.Canny(gray_crop, 65, 130)

		#CV_HOUGH_MULTI_SCALE = 2
		#lines = cv.HoughLines(edges_crop, 10, 1.0/180.0, 650)
		lines = cv.HoughLinesP(edges_crop, rho=10, theta=1.0/180, threshold=400, minLineLength=300, maxLineGap=50)

		if lines == None:
			cv.imshow('frame', draw_crop)
			if cv.waitKey(1) & 0xFF == ord('q'):
				break
			continue

		'''for line in lines:
			for rho, theta in line:
				x1, y1, x2, y2 = polar_to_cart_line(rho, theta)
				cv.line(draw_crop, (x1, y1), (x2, y2), (0, 0, 255), 1)
		'''
		for line in lines:
			for x1, y1, x2, y2 in line:
				line = center_intersect(x1, y1, x2, y2, draw_crop)
				if line != None:
					cv.line(draw_crop, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)


		'''
		GET NUMBERS
		'''

		'''
		GET SEGMENTS
		'''

		cv.imshow('frame', draw_crop)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv.destroyAllWindows()


def grab_folder_imgs() :
	return sorted(os.listdir("vids/center/"))


def main_stream() :
	stream()

def main_folder() : 
	img_paths = grab_folder_imgs()
	
	for img_path in img_paths:
		img = cv.imread(os.path.join("vids", video, img_path))
		img = cv.GaussianBlur(img, (5,5), 0)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		#edges = cv.Canny(gray, 100, 200)
		circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=200, param2=200, minRadius=250, maxRadius=10000)
		
		if circles == None:
			cv.imshow('frame', img)
			if cv.waitKey(1) & 0xFF == ord('q'):
				break
			continue

		circle = circles[0][0]
		max_rad = circle[2]
		cv.circle(img, (circle[0], circle[1]), circle[2], (0,255,0), 2)

		for prop in circle_props:
			cv.circle(img, (circle[0], circle[1]), int(max_rad*prop), (0,255,0), 2)

		cv.imshow('frame', img)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
        

if __name__ == '__main__':
    main_stream()
