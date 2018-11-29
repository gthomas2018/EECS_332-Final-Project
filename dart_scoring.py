import numpy as np
import cv2 as cv



if __name__ == '__main__':


	cam = cv.VideoCapture(1)

	while True:
		ret, frame = cam.read()

		frame = cv.GaussianBlur(frame, (5.5), 0)

		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		edges = cv.Canny(gray, 100, 200)

		combined = cv.Add(frame, edges)

		cv.imshow('frame', combined)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cam.release()
	cv.destroyAllWindows()