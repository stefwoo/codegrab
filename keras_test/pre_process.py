# -*- coding: utf-8 -*-
import PIL, glob, queue, itertools
from PIL import Image

#input: RGB image，灰度低于threshold的灰度设置为白色
def binarizing(img_path,threshold): 
    img = Image.open(img_path)
    im_gray = img.convert("L") # 转化为灰度图
    pixdata = im_gray.load()
    w, h = im_gray.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return im_gray
    # im_gray.save

#联通域算法,种子法
def flood_fill(img, x0, y0, blank_color, color_to_fill):
    pix = img.load()
    visited = set()
    q = queue.Queue()
    q.put((x0, y0))
    visited.add((x0, y0))
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while not q.empty():
        x, y = q.get()
        pix[x, y] = color_to_fill
        for xoffset, yoffset in offsets:
            x1, y1 = x + xoffset, y + yoffset
            if ((x1, y1) not in visited)&(pixdata[x1, y1]==0):
			    visited.add((x1, y1))
				q.put((x1, y1))
    return visited

#种子法 input:image path
def seeds_split(im_path):		
	img = Image.open(img_path)
	pixdata = img.load()
	w,h = img.size
    visited_set_list = []

	for x in range(2,w-2):
		for y in range(2,h-2):
			l1 = list(itertools.chain.from_iterable(visited_set_list))
			if (pixdata[x,y]==0)&((x,y) not in l1):
				flood_fill(img,x,y,)

    visited = set()



def connection(img_path):   #input: gray image path 连通域算法
	img = Image.open(img_path)
	pixdata = img.load()
	w,h = img.size
	letters = []
	# set_sum = []
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









if __name__ == '__main__':
	# import sys
	# print sys.argv
	image_list = [img for img in glob.glob("./data/*/*/*png")]
	for file in image_list[:1]:
		# binarizing(file,250).save(file,"PNG")
		connection(file).show()
	# im = Image.open(image_list[5])
	# binarizing(im,245).show()



