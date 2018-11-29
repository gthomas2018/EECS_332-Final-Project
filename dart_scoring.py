import numpy as np
import cv2 as cv
import os, os.path


def stream() : 
    
    cam = cv.VideoCapture(1)
    
    while True:
        ret, frame = cam.read()
        
        frame = cv.GaussianBlur(frame, (5,5), 0)
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        edges = cv.Canny(gray, 100, 200)
        
        #combined = cv.Add(frame, edges)
        
        cv.imshow('frame', edges)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv.destroyAllWindows()


def grab_folder_imgs() :
	img_paths = sorted(os.listdir("imgs"))
	imgs = []

	for img_path in img_paths:
		imgs.append(cv.imread(os.path.join("imgs", img_path)))

	return imgs	


def main() : 
	imgs = grab_folder_imgs()
	circle_props = [6.7/8.5, 6.25/8.5, 4.25/8.5, 3.75/8.5, 0.75/8.5, 0.25/8.5]

	for img in imgs:
		img = cv.GaussianBlur(img, (5,5), 0)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		#edges = cv.Canny(gray, 100, 200)

		circles = []
		circles.append(cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=200, param2=200, minRadius=250, maxRadius=10000)[0][0])
		max_rad = circles[0][2]

		for i in range(len(circle_props)):
			print(i)
			circles.append(cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=200, param2=200, minRadius=int(max_rad*circle_props[i])-20, maxRadius=int(max_rad*circle_props[i])+20)[0][0])
			#print(cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=200, param2=20, minRadius=int(max_rad*circle_props[i])-20, maxRadius=int(max_rad*circle_props[i])+20)[0][0])
			print(circles)
			cv.circle(img, (circles[i][0], circles[i][1]), circles[i][2], (0,255,0), 2)
			cv.imshow('circles', img)
			if cv.waitKey() & 0xFF == ord('q'):
				break
        

if __name__ == '__main__':
    main()
