from xml.etree import  ElementTree
import os
def trim(text):
    s=text.lstrip()
    s=s.rstrip()
    s=s.strip()
    return s
def read_xml(src,dst_path):
    f_out=open(dst_path,'w')
    root=ElementTree.fromstring(src)
    width_node=root.getiterator("width")
    height_node=root.getiterator("height")
    depth_node=root.getiterator("data")
    width=int(width_node[0].text)
    height=int(height_node[0].text)
    depth_str=depth_node[0].text
    depth_str=trim(depth_str)
    depth_str=depth_str.replace('    ',' ')
    depth_str=depth_str.replace('\r\n',' ')
    depth_str=depth_str.replace('\n',' ') 
    pixels=depth_str.split(' ')
    index=0
    for y in range(0,height):
        for x in range(0,width):
#            print('index:%s data:%s'%(index,pixels[index]))
            try:
              pixel=float(pixels[index])
              f_out.write(str(pixel)+' ')
              index=index+1
            except ValueError,e:
              index=index+1
              pixel=float(pixels[index])
              f_out.write(str(pixel)+' ')
              index=index+1     
        f_out.write('\n')
    f_out.close()
    return
def list_dir(src_path,dst_path):
    all_files=os.listdir(src_path)
    for f in all_files:
        if  os.path.isdir(f):
            continue
        else:
            dst_name=dst_path+'/'+f+'.txt'
            read_xml(open(src_path+'/'+f).read(),dst_name)
            print('file:%s converted'%(f))
    return
src_dir_path='D:/s01_e01'
dst_dir_path='D:/s01_e01'
list_dir(src_dir_path,dst_dir_path)