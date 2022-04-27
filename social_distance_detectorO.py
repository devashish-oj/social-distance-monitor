import social_distancing_config as config
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
from gtts import gTTS
from playsound import playsound
import argparse
import imutils
import cv2
import os
from pygame import mixer

mixer.init()
sound = mixer.Sound('beep.mp3')

arg = argparse.ArgumentParser()
arg.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
arg.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
arg.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(arg.parse_args())
label = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(label).read().strip().split("\n")
weight = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weight)
if config.USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
video = cv2.VideoCapture('college.mp4')
writer = None
count = 0
while True:
	(grabbed, frame) = video.read()
	if grabbed:
		count += 5
		video.set(cv2.CAP_PROP_POS_FRAMES,count)
	else:
		break
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame,frame,frame])
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))
	violate_red = set()
	violate_yellow = set()
	if len(results) >= 2:
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if (config.MIN_DISTANCE)*(0.8) < D[i, j] < config.MIN_DISTANCE:
					violate_yellow.add(i)
					violate_yellow.add(j)
				elif  D[i,j] < config.MIN_DISTANCE*(0.8):
					violate_red.add(i)
					violate_red.add(j)
	for (i, (prob, bbox, centroid)) in enumerate(results):
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)
		if i in violate_red:
			color = (0, 0, 255)
		if i in violate_yellow:
			color = (0,255,255)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)
	text_red = "Maximum Risk: {}".format(len(violate_red))
	cv2.putText(frame, text_red, (10, frame.shape[0] - 25),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	text_yellow = "Medium Risk: {}".format(len(violate_yellow))
	cv2.putText(frame, text_yellow, (10, frame.shape[0] - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

	if (len(violate_red) > int(len(results) * 0.3)):
		sound.play(maxtime=200)

	if args["display"] > 0:
		frame = cv2.resize(frame,(1280,720))
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	if args["output"] != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)
	if writer is not None:
		writer.write(frame)
