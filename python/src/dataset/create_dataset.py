import cv
import img.image as img
import numpy as np
import gzip
import cPickle
import os
def trim(sstr):
    ystr=sstr.lstrip()
    ystr=ystr.rstrip()
    ystr=ystr.strip()
    ystr=ystr.strip('\n')
    return ystr
def label_to_int(label):
    if label=='centromere':
        return 1
    elif label=='coarse_speckled':
        return 2
    elif label=='cytoplasmatic':
        return 3
    elif label=='fine_speckled'   :
        return 4
    elif label=='homogeneous':
        return 5
    elif label=='nucleolar':
        return 6

def load_data(image_dir,label_file_path):
    f=open(label_file_path)
    image_label=[]
    for line in f:
        line=trim(line)
        label=line.split(';')
        tmp=[]
        tmp.append(int(label[0]))
        tmp.append(label_to_int(label[1]))
        image_label.append(tmp)
    f.close()
    files=os.listdir(image_dir)
    input=[]
    output=[]
    for fp in files:
        pixels=img.load_grayimage(image_dir+'/'+fp)
        names=fp.split('.')
        l=int(names[0])
        input.append(pixels)
        for data in image_label:
            if data[0]==l:
                output.append(data[1])
                break
    #input=np.asanyarray(input, dtype='float32')
    #output=np.asanyarray(output, dtype='int32')
    return (input,output)
def create_dataset(train_dir,train_label,test_dir,test_label,dataset_path):
    train_dataset=load_data(train_dir, train_label)
    test_dataset=load_data(test_dir, test_label)
    zip=gzip.open(dataset_path,'wb')
    cPickle.dump(train_dataset, zip, -1)
    cPickle.dump(test_dataset,zip,-1)
    zip.close()
def load_dateset(path):
    f=gzip.open(path, 'rb')
    train_data=cPickle.load(f)
    test_data=cPickle.load(f)
    f.close()
    return [train_data,test_data]