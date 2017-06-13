
import glob,os
l = [l1 for l1 in glob.glob("./data/validation/*")]#train
for dir in l:
	img = [im for im in glob.glob(dir+"/*png")]
	# count = len(img)%4
	# print len([im for im in glob.glob(dir+"/*png")])%4
	# for img in [im for im in glob.glob(dir+"/*png")][:]:

	# for x in range(count):	
	# 	del_img = img.pop()
	# 	os.remove(del_img)
	# 	print del_img
	for img in img[8:]:
		os.remove(img)



	# print count
