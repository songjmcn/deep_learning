import numpy as np
import cv
import os
import re
def load_grayimage(path):
    im=cv.LoadImage(path)
    width=im.width
    height=im.height
    array=[]
    for y in range(0,height):
        for x in range(0,width):
            pixel=cv.Get2D(im, y, x)[0]
            array.append(pixel)
    return np.array(array)
def resize_grayimage(src_path,dst_path,width,height,filter=''):
    files=os.listdir(src_path)
    use_filter=False
    pattern=''
    if len(filter)>0:
        use_filter=True
        pattern=re.compile(filter)
    
    for f in files:
        if use_filter:
            if pattern.match(f):
                continue
        img=cv.LoadImage(src_path+'/'+f,0)
        tmp_image=cv.CreateImage((width,height), 8, 1)
        cv.Resize(img, tmp_image)
        cv.SaveImage(dst_path+'/'+f,tmp_image)
    return