import keras
from keras.models import model_from_json  
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os, uuid
import glob

model = model_from_json(open('my_model_architecture.json').read())  
model.load_weights('my_model_weights.h5')  



def predit(img):
	# im =  load_img(img,grayscale=True)
	im =  load_img(img)
	x = img_to_array(im)
	x = np.rollaxis(x,1)
	x = x.reshape((1,) + x.shape)
	predit_result = model.predict_classes(x)
	# print "path:",path
	path = str(predit_result)
	if os.path.exists("./test/"+path)==False:
		os.mkdir("./test/"+path)
	filename = "./test/"+path+"/" + str(uuid.uuid4()) + ".png"
	print filename
	im.save(filename)
	print "save ", filename
	# return pre_temp1


img_list = [img for img in glob.glob("./test/*png")]
for file in img_list:
	predit(file)
	# path = predit(file)




