# TO RUN:
# python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream   # Import the packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()                      # Construct the argument parse and parse the arguments
ap.add_argument("-p", "--prototxt", required=True,  # Path to the Caffe prototxt file
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,     # Path to the pre-trained model
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,      # Minimum probability threshold
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",        # All the class labels
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))              # Set of bounding box colors

print("[INFO] loading model...")                        # load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")            # Initialize video frame
vs = VideoStream(src=0).start()                     # Start camera sensor
time.sleep(2.0)
fps = FPS().start()                                 # Initializing FPS counter

# loop over the frames from the video stream
while True:
	frame = vs.read()               # Read a frame
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]        # Grab the frame dimensions and convert it to a blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	net.setInput(blob)          # Pass through our network
	detections = net.forward()

# At this point, we have detected objects in the input frame
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:                             # Filter out weak detections

			idx = int(detections[0, 0, i, 1])                           # Extract the index of the class label
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])     # Compute x,y coordicates of bounding box
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],                  # Draw the prediction on the frame
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),        # Draw colored bounding box
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),                      # If there isn’t room, we’ll display label just below the top of the rectangle
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Frame", frame)                  # Show the output frame
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):                         # If the `q` key is pressed, break from the loop
		break
	fps.update()

fps.stop()                                      # Stop the timer
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))      # Display FPS information
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
