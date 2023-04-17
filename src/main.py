from typing import List, Tuple, Set, Dict

import os, os.path

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

BATCH_SIZE: int  = 32
LABELS_TO_NUMBER = {
    "biological": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4
}
NUMBER_TO_LABEL = {
    0:"biological",
    1:"glass",
    2:"metal",
    3:"paper",
    4:"plastic"
}




def get_orginal_dataset_size() -> int:
    
    count: int = 0
        
    current_dir: str = os.getcwd()
    data_dir: str =os.path.join(os.path.dirname(current_dir), 'data')
    
    if os.path.isdir(data_dir):
        list_class_dirs_names: List[str] = os.listdir(data_dir)
        for current_dir_name in list_class_dirs_names:
            
            current_dir: str = os.path.join(data_dir, current_dir_name)
            
            list_of_files: List[str] = os.listdir(current_dir)
        
            count += len(list_of_files)
            
    return count
    
def calculate_orginal_distribution(path: str) -> int:

    list_of_files: List[str] = os.listdir(path)
    return len(list_of_files)

def load_dataset_and_prepare():
    current_dir: str = os.getcwd()
    data_dir: str =os.path.join(os.path.dirname(current_dir), 'data')
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    data.as_numpy_iterator().next()
    

    #preprocessing sacling to <0,1>
    #data = data.map(lambda x,y: (x/255, y))
    #data.as_numpy_iterator().next()
    
    #print(f"NUMBER OF BATCHES: {len(data)}")
    
    #train_size = int(len(data)* 0.7)
    #val_size = int(len(data)* 0.2)
    #test_size = int(len(data)* 0.1)

    #print("BATCH SIZES:")
    #print(f"TRAIN NUMBER OF BATCHES: {train_size}")
    #print(f"VALIDATION NUMBER OF BATCHES: {val_size}")
    #print(f"TEST NUMBER OF BATCHES: {test_size}")

    #train = data.take(train_size)
    #val = data.skip(train_size).take(val_size)
    #test = data.skip(train_size+val_size).take(test_size)


    dataset_image_count: int = get_orginal_dataset_size()

    per_class_orginal_distribution = []
    
    current_dir: str = os.getcwd()
    data_dir: str =os.path.join(os.path.dirname(current_dir), 'data')

    if os.path.isdir(data_dir):
        list_class_dirs_names: List[str] = os.listdir(data_dir)
        for current_dir_name in list_class_dirs_names:
            current_dir: str = os.path.join(data_dir, current_dir_name)
            result = calculate_orginal_distribution(current_dir)
            per_class_orginal_distribution.append(result)

    for it, value in enumerate(per_class_orginal_distribution):
        print(f"{NUMBER_TO_LABEL[it]}  =  {value}")


    #SPLIT 1
    per_class_proportions = []
    for it in per_class_orginal_distribution:
        per_class_proportions.append(int(it/dataset_image_count*100))

    split_1_dataset_size = 0
    for it, value in enumerate(per_class_proportions):
        split_1_dataset_size+=value
        print(f"{NUMBER_TO_LABEL[it]}  =  {value}")

    print(f"SPLIT 1 SIZE: {split_1_dataset_size}")
    
    #SPLIT 2    
    uniform_distribution: int  = dataset_image_count/ len(NUMBER_TO_LABEL)
    print(f"UNIFORM DISTRIBUTION: {uniform_distribution}")

def check_image_sizes() -> List[Tuple[int, int]]:
    
    sizes: Set[Tuple[int, int]] = set()
        
    current_dir: str = os.getcwd()
    data_dir: str =os.path.join(os.path.dirname(current_dir), 'data')
    
    if os.path.isdir(data_dir):
        list_class_dirs_names: List[str] = os.listdir(data_dir)
        for current_dir_name in list_class_dirs_names:
            
            current_dir: str = os.path.join(data_dir, current_dir_name)
            
            list_of_files: List[str] = os.listdir(current_dir)
        
            for current_file_name in list_of_files:
                
                path_to_file: str = os.path.join(current_dir, current_file_name)
                
                image = cv2.imread(path_to_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #changing to rgb
                plt.show()
                height, width, channels = image.shape
                
                sizes.add((width, height))
    
    image_sizes_list = list(sizes) 

    return image_sizes_list
    
def find_max_image_size(image_sizes: List[Tuple[int,int]]) -> Tuple[int,int]:
    
    max_width_tuple = max(image_sizes, key=lambda x: x[0])
    max_height_tuple = max(image_sizes, key=lambda x: x[1])

    max_width = max_width_tuple[0]
    max_height = max_height_tuple[1]
    return max_width, max_height

def find_min_image_size(image_sizes: List[Tuple[int,int]]) -> Tuple[int,int]:
    

    min_width_tuple = min(image_sizes, key=lambda x: x[0])
    min_height_tuple = min(image_sizes, key=lambda x: x[1])

    min_width = min_width_tuple[0]
    min_height = min_height_tuple[1]
    
    return min_width, min_height

if __name__ == '__main__':
    
    image_sizes: List[Tuple[int,int]] = check_image_sizes()
    

    image_size_count: int = len(image_sizes)
    
    print(f"COUNT OF IMAGE SIZES: {image_size_count}")
    
    for size in image_sizes:
        
        width, height = size

    max_image_size = find_max_image_size(image_sizes)
    print(max_image_size)


    min_image_size = find_min_image_size(image_sizes)
    print(min_image_size)

    load_dataset_and_prepare()





