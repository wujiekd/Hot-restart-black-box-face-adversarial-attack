import os
import shutil
import os.path
from pathlib import Path

def delete_folder(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            delete_folder(path_file)


def softdir(path,to_path):
    my_file = Path(to_path)
    if my_file.is_dir():
        delete_folder(to_path)
    else:
        os.makedirs(to_path)

    current_path = path
    filename_list = os.listdir(current_path)

    print('正在分类整理进文件夹ing...')
    for filename in filename_list:
        try:
            name1, name2 = filename.split('.')
            if name2 == 'jpg':
                try:
                    dir_name = filename.split('_')[0]
                    os.mkdir(to_path + '/' + dir_name)
                    print('创建文件夹'+dir_name)
                except:
                    pass
                try:
                    shutil.copy(current_path+'/'+filename, to_path+'/'+dir_name)
                    print(filename+'转移成功！')
                except Exception as e:
                    print('移动失败:' + e)
        except:
            pass

    print('整理完毕！')


if __name__ == '__main__':
    softdir('../result_data/adv_images','../result_data/images')



