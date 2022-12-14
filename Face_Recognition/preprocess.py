import cv2
import os
import numpy as np

class Preprocessor():
	def __init__(self, training_path, testing_path, cascade_path, image_vector_size = 120):
		"""
		Called when class is created with the variables of data
		"""
		self.training_path = training_path
		self.testing_path = testing_path
		self.cascade_path = cascade_path
		self.image_vector_size = image_vector_size
	
	def __detect_face__(self, img):
		"""
		Method which perform face detection using haarcascade tool.
		img: cv2 BGR plane image
		returns: coordinated face of the given image, coordinates
		"""
		detector = cv2.CascadeClassifier(self.cascade_path)
		face = detector.detectMultiScale(img, minNeighbors=3)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if len(face) == 0:
				return (-1, -1)
		(x, y, w, h) = face[0]
		return (img[y:y+w, x:x+h], face[0])

	def __prepare_data__(self, path, type = "Training"):
			"""
			path: string where training image folders are present
			returns: detected faces and their labels(names of folders where group of images are present)
			"""
			detected_faces = []
			face_labels = []
			img_dir = os.listdir(path)
			for dir_name in img_dir:
					label = str(dir_name)
					if not label.startswith("."):
							img_names = os.listdir(path + '/' + dir_name)
							for img_name in img_names:
									if not img_name.startswith("."):
											img_path = path + '/' + dir_name + '/' + img_name
											img = cv2.imread(img_path)
											face, _ = self.__detect_face__(img)
											resized_face = cv2.resize(face, (self.image_vector_size, self.image_vector_size), interpolation = cv2.INTER_AREA)
											p = "Build/Face_data" + '/' + type + '/' + label
											if not os.path.exists(p):
													os.makedirs(p)
											cv2.imwrite(p + '/' + img_name, face)
											detected_faces.append(resized_face)
											face_labels.append(label)
			return (np.array(detected_faces), np.array(face_labels))
	
	def get_train_data(self):
		"""
		Method which performs face detection task using detect face function for every image in the training set
		returns: detected faces and their labels(names of folders where group of images are present)
		"""
		return self.__prepare_data__(self.training_path, type="Training")
	
	def get_test_data(self):
		"""
		Method which performs face detection task using detect face function for every image in the testing set
		returns: detected faces and their labels(names of folders where group of images are present)
		"""
		return self.__prepare_data__(self.testing_path, type="Testing")