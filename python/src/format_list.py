def trim(src_str):
    ystr=src_str.strip()
    ystr=ystr.lstrip()
    ystr=ystr.rstrip()
    return ystr
def format_list(src_file_path,dst_file_path):
    src_f=open(src_file_path, 'r')
    dst_f=open(dst_file_path,'w')
    for line in src_f:
        line=trim(line)
        print(str.format("get data:%s\n"%(line)))
        file_name=line.split('.')[0]
        format_str=line+" "+file_name+".txt\n"
        print("output data:%s\n" % (format_str))
        dst_f.write(format_str)
    src_f.close()
    dst_f.close()
format_list(r"d:\file_list.txt",r"d:\file_list.list")