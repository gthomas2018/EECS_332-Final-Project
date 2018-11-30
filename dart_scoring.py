import numpy as np
import cv2 as cv
import os, os.path


video = "center"
circle_props = [6.7/8.5, 6.25/8.5, 4.25/8.5, 3.75/8.5, 0.75/8.5, 0.25/8.5]


def stream() : 
	cam = cv.VideoCapture(1)
	cam.set(3, 1920)
	cam.set(4, 1080)

	while cam.isOpened():
		ret, img = cam.read()

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
    main_folder()
