from typing import List, Tuple, Set, Dict

import os, os.path

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

IMAGE_WIDTH: int = 224
IMAGE_HEIGHT: int = 224

CATEGORIES: Dict[int, str] = { 0: 'biological', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic'}

def preprocess_image():
    pass

def load_dataset():
    current_dir: str = os.getcwd()
    data_dir: str =os.path.join(os.path.dirname(current_dir), 'data')
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    data_iterator = data.as_numpy_iterator()

    

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

        #print(f"{width}, {height}")

    max_image_size = find_max_image_size(image_sizes)
    print(max_image_size)


    min_image_size = find_min_image_size(image_sizes)
    print(min_image_size)

    load_dataset()





