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

def normalise_data(x,y):
    return (x/255, y)

def histrogram(path_to_class: str):

    current_dir: str = os.getcwd()
    data_dir: str =os.path.join(os.path.dirname(current_dir), f"data/{path_to_class}")
    
    files = os.listdir(data_dir)

    images = []
    for i, file_name in enumerate(files):
        file_path = os.path.join(data_dir, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #changing to rgb
        images.append(image)

    histograms = []
    r = []
    g = []
    b = []
    for j, image in enumerate(images):
        histogram_r = cv2.calcHist([image], [0], None, [255], [0, 255])
        histogram_g = cv2.calcHist([image], [1], None, [255], [0, 255])
        histogram_b = cv2.calcHist([image], [2], None, [255], [0, 255])
        r.append(histogram_r)
        g.append(histogram_g)
        b.append(histogram_b)
        histograms.append((histogram_r, histogram_g, histogram_b))

    mean_r = np.mean(r, axis=0)
    mean_g = np.mean(g, axis=0)
    mean_b = np.mean(b, axis=0)

    plt.title(f"Histogram for {path_to_class}")
    plt.plot(mean_r, color='r')
    plt.plot(mean_g, color='g')
    plt.plot(mean_b, color='b')
    plt.grid(True)
    plt.savefig(f"{path_to_class}_histogram.png")
    plt.show()

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
    
    print(f"NUMBER OF BATCHES: {len(data)}")

    dataset_image_count: int = get_orginal_dataset_size()

    per_class_orginal_distribution = []
    
    if os.path.isdir(data_dir):
        list_class_dirs_names: List[str] = os.listdir(data_dir)
        for current_dir_name in list_class_dirs_names:
            current_dir: str = os.path.join(data_dir, current_dir_name)
            result = calculate_orginal_distribution(current_dir)
            per_class_orginal_distribution.append(result)

    #for it, value in enumerate(per_class_orginal_distribution):
        #print(f"{NUMBER_TO_LABEL[it]}  =  {value}")

    #SPLIT 1
    per_class_proportions = []
    for it in per_class_orginal_distribution:
        per_class_proportions.append(int(it/dataset_image_count*100))

    split_1_dataset_size = 0
    for it, value in enumerate(per_class_proportions):
        split_1_dataset_size+=value
        #print(f"{NUMBER_TO_LABEL[it]}  =  {value}")

    print(f"SPLIT 1 SIZE: {split_1_dataset_size}")
    
    train_size = int(split_1_dataset_size*0.7)
    val_size = int(split_1_dataset_size*0.2)
    test_size = int(split_1_dataset_size*0.1)
    
    train_data_split_1 = data.take(train_size)
    val_data_split_1 = data.skip(train_size).take(val_size)
    test_data_split_1 = data.skip(train_size+val_size).take(test_size)

    #SPLIT 2    
    uniform_distribution: int  = int(dataset_image_count/ len(NUMBER_TO_LABEL))
    #print(f"UNIFORM DISTRIBUTION: {uniform_distribution}")

    per_class_proportions_split_2 = []

    for _ in list(LABELS_TO_NUMBER.keys()):
        per_class_proportions_split_2.append(int(uniform_distribution/dataset_image_count*100))


    split_2_dataset_size = 0
    for it, value in enumerate(per_class_proportions_split_2):
        split_2_dataset_size+=value
        #print(f"{NUMBER_TO_LABEL[it]}  =  {value}")
    
    print(f"SPLIT 2 SIZE: {split_2_dataset_size}")

        
    train_size = int(split_2_dataset_size*0.7)
    val_size = int(split_2_dataset_size*0.2)
    test_size = int(split_2_dataset_size*0.1)
    
    train_data_split_2 = data.take(train_size)
    val_data_split_2 = data.skip(train_size).take(val_size)
    test_data_split_2 = data.skip(train_size+val_size).take(test_size)


    train_data_split_2 =train_data_split_2.map(normalise_data)
    val_data_split_2=val_data_split_2.map(normalise_data)
    test_data_split_2=test_data_split_2.map(normalise_data)

    train_data_split_3 = train_data_split_2.concatenate(val_data_split_2)
    val_data_split_3 = val_data_split_2
    test_data_split_3 = test_data_split_2

    batch = train_data_split_3.as_numpy_iterator().next()
    
    #print(batch[0])
    #print(batch[0].max())

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


    #for it in list(LABELS_TO_NUMBER.keys()):
    #    histrogram(f"{it}")
    #
    load_dataset_and_prepare()





