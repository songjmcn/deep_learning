import Image
import numpy
import cv
img=Image.open('D:/data/train/001.png')
arr=numpy.asarray(img, dtype='int32')
print(arr)
cvimage=cv.LoadImage('D:/data/train/001.png')
array=numpy.asarray(cvimage.tostring(),dytpe='byte')
print('\n')
print(array)
