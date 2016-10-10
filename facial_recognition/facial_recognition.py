"""
Secured-Pi, 2016
This is a script for the Secured-Pi project dealing with the machine learning aspects of the facial recognition.

This script relies on the OpenCV 3.1 library installed with the opencv-contrib face module.
If you do not have OpenCV installed, I highly recommend that you attempt to do so using a virtual box.

For training, create a directory called 'training' inside of this directory.  Each person's photo should
be named like "member-1-a", where 1 identifies the person to whom the image belongs to, and the letter
increments for each photo.  For example, a training set of 2 people might contain the images:
member-1-a.gif
member-1-b.gif
member-1-c.gif
member-1-d.gif
member-2-a.gif
member-2-b.gif .. and so on ..

For accuracy, we recommend a minimum of __ pictures for each member.  It is also important that
no other files be contained in the training directory besides the pictures.
"""

import os
import cv2
import numpy as np
import re
from PIL import Image


CASCADE_MODEL = 'haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_MODEL)
TRAINING_SET_PATH = 'training'


def train_recognizer(recognizer=cv2.face.createLBPHFaceRecognizer, image_path=TRAINING_SET_PATH, save_file='recog_brain.yml', recog_model=None):
	"""Train the facial recognition software with some training photos.
	
	This will train the recognizer, and save the training.  This saved file
	will be used by test_individual to verify membership.
	"""
	recognizer = recognizer()
	if recog_model is not None:
		recognizer.load(recog_model)
	tr_files = os.listdir(image_path)
	tr_files = [image_path + '/' + f for f in tr_files]
	images = []		# image arrays of face cut-outs
	members = []	# this will be the id of the members
	
	for tr_f in tr_files:
		print('training with file:' + tr_f)
		tr_img = np.array(Image.open(tr_f).convert('L'), 'uint8')
		curr_member = int(re.search('-(\d+)-', os.path.split(tr_f)[1]).group(1))
		curr_face = FACE_CASCADE.detectMultiScale(tr_img)

		for (x, y, w, h) in curr_face:
			images.append(tr_img[y: y + h, x: x + w])
			members.append(curr_member)
			# to see the images while training(demo purposes), uncomment lines below
			# cv2.imshow("Training...", tr_img[y: y + h, x: x + w])
			# cv2.waitKey(10)

	# cv2.destroyAllWindows()
	recognizer.train(images, np.array(members))
	recognizer.save(save_file)


def test_individual(image_to_test, threshold=40, recognizer=cv2.face.createLBPHFaceRecognizer, recog_model='recog_brain.yml'):
	"""Test if an individual has access to the lock.

	Return true or false."""
	recognizer = recognizer()
	recognizer.load(recog_model)
	
	image_array = np.array(Image.open(image_to_test).convert('L'), 'uint8')
	curr_face = FACE_CASCADE.detectMultiScale(image_array)[0]
	x, y, w, h = curr_face
	test_image = image_array[y: y + h, x: x + w]

	member_prediction, confidence = recognizer.predict(test_image)
	print('member number: {}, confidence: {}'.format(member_prediction, confidence))
	if confidence > threshold:
		return False
	return True