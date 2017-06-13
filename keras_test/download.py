# -*- coding: utf-8 -*-
import requests
import uuid
from PIL import Image
import os

url = "http://www.scgh114.com/weixin/drawImage/code"
for i in range(200):
    resp = requests.get(url)
    filename = "./test/" + str(uuid.uuid4()) + ".png"
    with open(filename, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
        f.close()
    im = Image.open(filename)
    if im.size != (90, 20):
        os.remove(filename)
    else:
        print filename

