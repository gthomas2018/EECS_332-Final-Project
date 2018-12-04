import numpy as np
import cv2 as cv
import random
import math


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


def center_intersect(x1, y1, x2, y2, draw_crop) :
	if float(x1 - x2) == 0:
		return None
	m = float(y1 - y2)/float(x1 - x2)
	b = int(y1 - float(m*x1))

	area_buffer = 10
	center = (draw_crop.shape[1]/2, draw_crop.shape[0]/2)
	min_y = int(center[1] - area_buffer)
	max_y = int(center[1] + area_buffer)
	min_x = int(center[0] - area_buffer)
	max_x = int(center[0] + area_buffer)

	#cv.rectangle(draw_crop, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

	y = int(m*(center[0]) + b)
	if m == 0:
		return None
	x = int((center[1]-b)/m)

	if y in range(min_y, max_y) or x in range(min_x, max_x):
		y = int(m*draw_crop.shape[1] + b)
		line = [0, b, draw_crop.shape[1], y]
		return line
	else:
		return None


def sort_lines(lines) :
	lines = sorted(lines, key=lambda line: line[1], reverse=True)
	good_lines = []

	for i in range(len(lines)-1):
		if abs(lines[i][1] - lines[i+1][1]) > 50:
			good_lines.append(lines[i])
	if not len(good_lines) == 10:
		good_lines.append(lines[-1])

	return good_lines


def get_score(lines, circles, dart_point) :
	center = (circles[0][0], circles[0][1])
	x1 = dart_point[0]
	y1 = dart_point[1]
	x2 = center[0]
	y2 = center[1]

	m = float(y1 - y2)/float(x1 - x2)
	b = int(y1 - float(m*x1))

	scores = [(1, 19), (18, 7), (4, 16), (13, 8), (6, 11), (14, 10), (9, 15), (12, 2), (5, 17), (20, 3)]
	score = 0

	for i in range(len(lines)-1):
		if b <= lines[i][1] and b >= lines[i+1][1]:
			if i == 4:
				# horizontal dart_point x val
				if dart_point[0] >= center[0]:
					score = scores[i][0]
				else:
					score = scores[i][1]
			else:
				if dart_point[1] >= center[1]:
					score = scores[i][1]
				else:
					score = scores[i][0]
			break
		else:
			if dart_point[1] >= center[1]:
				score = scores[-1][1]
			else:
				score = scores[-1][0]

	dist = math.sqrt((dart_point[0] - center[0])**2 + (dart_point[1] - center[1])**2)

	if dist > circles[1][2]:
		score *= 0
	elif dist <= circles[1][2] and dist > circles[2][2]:
		score *= 2
	elif dist <= circles[3][2] and dist > circles[4][2]:
		score *= 3
	elif dist <= circles[5][2] and dist > circles[6][2]:
		score = 25
	elif dist <= circles[6][2]:
		score = 50

	return score


def find_hsv_settings() :
	cam = cv.VideoCapture(1)
	cam.set(3, 1280)
	cam.set(4, 720)

	def nothing(x):
		pass

	cv.namedWindow('image')
	cv.createTrackbar("Hue_lower", 'image', 0, 180, nothing)
	cv.createTrackbar("Hue_upper", 'image', 0, 180, nothing)
	cv.createTrackbar("Sat_lower", 'image', 0, 255, nothing)
	cv.createTrackbar("Sat_upper", 'image', 0, 255, nothing)
	cv.createTrackbar("Val_lower", 'image', 0, 255, nothing)
	cv.createTrackbar("Val_upper", 'image', 0, 255, nothing)

	while cam.isOpened():
		ret, img = cam.read()
		img = cv.GaussianBlur(img, (5,5), 0)

		HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

		H_thresh_lower = cv.getTrackbarPos("Hue_lower", 'image')
		H_thresh_upper = cv.getTrackbarPos("Hue_upper", 'image')
		S_thresh_lower = cv.getTrackbarPos("Sat_lower", 'image')
		S_thresh_upper = cv.getTrackbarPos("Sat_upper", 'image')
		V_thresh_lower = cv.getTrackbarPos("Val_lower", 'image')
		V_thresh_upper = cv.getTrackbarPos("Val_upper", 'image')

		min_hsv = (H_thresh_lower, S_thresh_lower, V_thresh_lower)
		max_hsv = (H_thresh_upper, S_thresh_upper, V_thresh_upper)

		mask = cv.inRange(HSV_img, min_hsv, max_hsv)

		cv.imshow('image', mask)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv.destroyAllWindows()


def find_dart(board_crop) :
	# PINK
	Hue_min = 160
	Hue_max = 180
	Sat_min = 99
	Sat_max = 200
	Val_min = 134
	Val_max = 255

	# Blue
	'''Hue_min = 92
	Hue_max = 111
	Sat_min = 77
	Sat_max = 125
	Val_min = 90
	Val_max = 163'''

	min_hsv = (Hue_min, Sat_min, Val_min)
	max_hsv = (Hue_max, Sat_max, Val_max)

	HSV_crop = cv.cvtColor(board_crop, cv.COLOR_BGR2HSV)

	mask = cv.inRange(HSV_crop, min_hsv, max_hsv)

	kernel = np.ones((5,5), np.uint8)
	mask = cv.erode(mask, kernel, iterations=1)
	mask = cv.dilate(mask, kernel, iterations=2)

	params = cv.SimpleBlobDetector_Params()

	params.minThreshold = 0
	params.maxThreshold = 256

	params.filterByColor = False
	#params.blobColor = 255

	params.filterByArea = False
	#params.minArea = 1

	params.filterByCircularity = False
	#params.minCircularity = 0.5

	params.filterByConvexity = False
	#params.minConvexity = 0.6

	params.filterByInertia = False
	#params.minInertiaRatio = 0.8

	detector = cv.SimpleBlobDetector_create(params)

	mask = mask
	points = detector.detect(mask)

	if not points:
		return None
	else:
		clean_points = []

		for point in points:
			clean_points.append((int(point.pt[0]), int(point.pt[1])))
		return clean_points


def main() : 
	cam = cv.VideoCapture(1)
	cam.set(3, 1920)
	cam.set(4, 1080)

	'''
	GET GOOD BOARD PROPOSAL LINES
	'''
	while cam.isOpened():
		while True:
			ret, img = cam.read()

			img = cv.GaussianBlur(img, (5,5), 0)
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

			'''
			GET CIRCLES
			'''
			circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=200, param2=200, minRadius=250, maxRadius=10000)
			
			if circles is None:
				continue

			outer_circle = circles[0][0]
			outer_circle_rad = int(outer_circle[2])
			outer_circle_x = int(outer_circle[0])
			outer_circle_y = int(outer_circle[1])

			board_min = (outer_circle_x - outer_circle_rad, outer_circle_y - outer_circle_rad)
			board_max = (outer_circle_x + outer_circle_rad, outer_circle_y + outer_circle_rad)
			board_crop = img[board_min[1] : board_max[1], board_min[0] : board_max[0]]
			gray_crop = gray[board_min[1] : board_max[1], board_min[0] : board_max[0]]
			draw_crop = board_crop.copy()

			outer_circle_x = int(draw_crop.shape[1]/2)
			outer_circle_y = int(draw_crop.shape[0]/2)

			saved_circles = [[outer_circle_x, outer_circle_y, outer_circle_rad]]
			cv.circle(draw_crop, (outer_circle_x, outer_circle_y), int(outer_circle_rad), (0,0,255), 2)

			for prop in circle_props:
				saved_circles.append([outer_circle_x, outer_circle_y, int(outer_circle_rad*prop)])
				cv.circle(draw_crop, (outer_circle_x, outer_circle_y), int(outer_circle_rad*prop), (0,0,255), 2)

			'''
			GET LINES
			'''
			edges_crop = cv.Canny(gray_crop, 65, 130)

			lines = cv.HoughLinesP(edges_crop, rho=10, theta=1.0/180, threshold=100, minLineLength=300, maxLineGap=50)

			if lines is None:
				cv.imshow('frame', draw_crop)
				if cv.waitKey(1) & 0xFF == ord('q'):
					break
				continue

			saved_lines = []

			for line in lines:
				for x1, y1, x2, y2 in line:
					line = center_intersect(x1, y1, x2, y2, draw_crop)
					if line != None:
						saved_lines.append(line)

			saved_lines = sort_lines(saved_lines)

			if len(saved_lines) < 10:
				continue

			for line in saved_lines:
				cv.line(draw_crop, (line[0], line[1]), (line[2], line[3]), (0,0,255), 2)

			cv.imshow('board proposal', draw_crop)
			key = cv.waitKey() & 0xFF
			if key == ord('y'):
				break
			elif key == ord('n'):
				continue
			elif key == ord('q'):
				cam.release()
				cv.destroyAllWindows()
				return
		break

	'''
	GET SCORE LOOP
	'''
	cnt = 0
	while True:
		ret, img = cam.read()

		img = cv.GaussianBlur(img, (5,5), 0)

		board_crop = img[board_min[1] : board_max[1], board_min[0] : board_max[0]]
		
		dart_points = find_dart(board_crop)

		if dart_points == None:
			cv.imshow('result', board_crop)
			if cv.waitKey(1) & 0xFF == ord('q'):
				break
			continue

		score = 0

		for dart_point in dart_points:
			score += get_score(saved_lines, saved_circles, dart_point)
			cv.circle(board_crop, (dart_point[0], dart_point[1]), 5, (0,255,0), 2)

		cv.putText(board_crop, "SCORE: " + str(score), (0, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
		
		cv.imshow('result', board_crop)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
		cv.imwrite("demo2/" + str(cnt).zfill(4) + ".jpg", board_crop)
		cnt += 1

	cam.release()
	cv.destroyAllWindows()
        


if __name__ == '__main__':
	#find_hsv_settings()
    main()