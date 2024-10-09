import glob
import os
import shutil
from sklearn.model_selection import train_test_split

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_files(source_files, destination_root):
    for file in source_files:
        file_class = os.path.basename(os.path.dirname(file))
        file_name = os.path.basename(file)
        destination_class = os.path.join(destination_root, file_class)
        create_directory(destination_class)
        shutil.copy(file, os.path.join(destination_class, file_name))

image_list = glob.glob('data/*/*.*')
print(image_list)

file_dir = 'dataset'
create_directory(file_dir)

shutil.rmtree(file_dir)
os.makedirs(file_dir)

trainval_files, val_files = train_test_split(image_list, test_size=0.2, random_state=42)

train_dir = 'train'
val_dir = 'val'
train_root = os.path.join(file_dir, train_dir)
val_root = os.path.join(file_dir, val_dir)
create_directory(train_root)
create_directory(val_root)

copy_files(trainval_files, train_root)
copy_files(val_files, val_root)
