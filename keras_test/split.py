# -*- coding: utf-8 -*-
import PIL
from PIL import Image
import itertools
import time
import uuid

def binarizing(img,threshold): #input: gray image，灰度低于threshold的灰度设置为白色
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img

def connection(img,num):   #input: gray image 连通域算法
	pixdata = img.load()
	w,h = img.size
	letters = []
	for x in range(2,w-2):
		for y in range(2,h-2):
			point = (x,y)
			marked_point = [i for letter in letters for i in letter]
			if (pixdata[x,y]==0) and (point not in marked_point):
				letters.append(_in_list(point,pixdata,letters))
	letters.sort(key=lambda x:len(x),reverse=True)
	for i in letters:print len(i)
	im = Image.new("P",img.size,255)
	pixdate = im.load()
	l2 = [i for l1 in letters[0:4] for i in l1]
	for i in l2:pixdate[i[0],i[1]]= 0
	return im

def _in_list(point,pixdata,letters):	
	single_erea = [point]
	for point in single_erea:
		x,y=point
		near_point = [(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]
		find = False
		for p in near_point:
			if (pixdata[p[0],p[1]] == 0)and(p not in single_erea):
				single_erea.append(p)
	return single_erea

def depoint(img):   #input: gray image
    pixdata = img.load()
    w,h = img.size
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > 245:
                count = count + 1
            if pixdata[x,y+1] > 245:
                count = count + 1
            if pixdata[x-1,y] > 245:
                count = count + 1
            if pixdata[x+1,y] > 245:
                count = count + 1
            if count > 2:
                pixdata[x,y] = 255
    return img

def some(img1):
	from operator import itemgetter
	img = img1.convert("P")
	his = img.histogram()
	values = {}
	for i in range(256):
		values[i] = his[i]
	# for j,k in sorted(values.items(), key=itemgetter(1), reverse=True)[:10]:
	# 	print j,k
	im2 = Image.new("P",img.size,255)
	for x in range(img.size[1]):
		for y in range(img.size[0]):
			pix = img.getpixel((y,x))
			if pix == 0: # these are the numbers to get
				im2.putpixel((y,x),255)
			else:
				im2.putpixel((y,x),pix)
	return im2

def get_crop_imgs(img):
	'''
	按照图片的特点,进行切割,这个要根据具体的验证码来进行工作. # 见原理图
	:param img:
	:return:
	'''
	child_img_list = []
	for i in range(4):
		x = 15 + i * 15  # 见原理图
		y = 0
		child_img = img.crop((x, y, x + 15, y + 20))
		child_img_list.append(child_img)
	return child_img_list

if __name__ == '__main__':
	# import sys
	# print sys.argv
	import glob
	# train_data = [img for img in glob.glob("./data/train/*/*png")]
	train_data = [img for img in glob.glob("./test/*png")]

	# print train_data
	for file in train_data:
		# print file
		im = Image.open(file)#('./captchas/daffd901-ba35-4770-ac66-1cc9ca518bb0.png')
		# print im.size
	# # im.show()
	# # im2 = some(im)
	# # im2.show()

		# im.show()
		# for i in get_crop_imgs(im):
		# 	filename = str(uuid.uuid4()) + ".png"
		# 	print filename
		# 	i.save(filename)
		# im.show()


		im = im.crop((0,2,15,18))
		im.save(file)




	# im2 = Image.new("P",im.size,255)
	# for x in range(im.size[1]):
	# 	for y in range(im.size[0]):
	# 			im2.putpixel((y,x),227)
	# im2.show()

	# im2=some(im)
	# time.sleep(1)
	# im3 = im2.convert("L") # 转化为灰度图
	# im4 = binarizing(im3,180)
	# im6 = connection(im4, 0)
	# im6.show()


