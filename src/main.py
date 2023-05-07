from typing import List, Tuple, Set, Dict, Optional

import time
import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import keras.utils as ku

from keras.callbacks import History
from keras.callbacks import Callback
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
#from keras_visualizer import visualizer

#
# # testing code
# test_img = os.path.join(data_dir, "paper/cardboard1.jpg")
# img = ku.load_img(test_img, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
# img = ku.img_to_array(img, dtype=np.uint8)
# img = np.array(img) / 255.0
# prediction = model.predict(img[np.newaxis, ...])
# print("Probability:", np.max(prediction[0], axis=-1))
# predicted_class = NUMBER_TO_LABEL[np.argmax(prediction[0], axis=-1)]
# print("Classified:", predicted_class, '\n')
#
# plt.axis('off')
# plt.imshow(img.squeeze())
# plt.title("Loaded Image")
# plt.show()


IMAGE_WIDTH: int = 256
IMAGE_HEIGHT: int = 256
BATCH_SIZE: int = 32
EPOCHS: int = 1

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

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        epoch_time = time.time() - self.epoch_start_time
        self.times.append(epoch_time)

    def get_times(self):
        return self.times


CALLBACK_TO_INDEX = {
    "HISTORY": 0,
    "TIME": 1
}
class ModelData:
    def __init__(self, model, num_of_epochs, train_data_split, val_data_split, test_data_split = None):
        self.model = model
        self.model_history = History()
        self.num_of_epochs = num_of_epochs
        self.train_data_split = train_data_split
        self.val_data_split = val_data_split
        self.test_data_split = test_data_split
        self.model_result = None
        self.callback_list = None
    def train_model(self, callback_list = None):

        if callback_list is not None:
            self.callback_list = [self.model_history] + callback_list
        else:
            self.callback_list = self.model_history
        model_result = self.model.fit(self.train_data_split,
                                      steps_per_epoch=len(self.train_data_split),
                                      epochs=self.num_of_epochs,
                                      validation_data=self.val_data_split,
                                      validation_steps=len(self.val_data_split),
                                      callbacks = self.callback_list
                                    )
        self.model_result = model_result

    def save_training_history(self, index = None):
        train_loss = self.model_history.history['loss']
        val_loss = self.model_history.history['val_loss']

        train_acc = self.model_history.history['accuracy']
        val_acc = self.model_history.history['val_accuracy']

        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        current_dir: str = os.getcwd()
        if index is not None:
            os.mkdir(os.path.join(current_dir, f"model_{index}"))
        else:
            os.mkdir(os.path.join(current_dir, f"model"))

        if index is not None:
            plt.grid(True)
            plt.savefig(os.path.join(f"model_{index}", f"model_loss_{index}.png"))
        else:
            plt.grid(True)
            plt.show()

            plt.savefig(os.path.join(f"model", f"model_loss.png"))

        plt.clf()
        plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if index is not None:
            plt.grid(True)
            plt.savefig(os.path.join(f"model_{index}", f"model_accuracy_{index}.png"))
        else:
            plt.grid(True)
            plt.show()
            plt.savefig(os.path.join(f"model", f"model_accuracy.png"))

        plt.clf()
    def save_model(self, index = 0):
        self.model.save(f"weights/model_{index}.h5")

    def get_callbacks(self):
        return self.callback_list

def find_best_split_1(models_data_list: List[ModelData]):

    best_model_data: Optional[ModelData] = models_data_list[0]

    for item in models_data_list:

        if item.model_result.history['val_accuracy'][-1] >= best_model_data.model_result.history['val_accuracy'][-1] \
                and item.model_result.history['val_loss'][-1] <= best_model_data.model_result.history['val_loss'][-1]:

            best_model_data = item

    return best_model_data


def find_best_split_2(models_data_list: List[ModelData]) -> Optional[int]:
    pass

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


def create_cnn_model(visualize = False):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(256, 256, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    model.summary()

    #if visualize:
       #visualizer(model, file_format='png', view=True)

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_to_file(path, data):
    with open(path, 'w') as f:
        f.write(str(data))

def calculate_orginal_distribution(path: str) -> int:

    list_of_files: List[str] = os.listdir(path)
    return len(list_of_files)

def load_dataset_and_prepare():
    current_dir: str = os.getcwd()
    data_dir: str = os.path.join(os.path.dirname(current_dir), 'data')

    data = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                       class_names=list(LABELS_TO_NUMBER.keys()),
                                                       image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                       batch_size=BATCH_SIZE)

    data.as_numpy_iterator().next()
    
    dataset_image_count: int = get_orginal_dataset_size()

    per_class_orginal_distribution = []
    
    if os.path.isdir(data_dir):
        list_class_dirs_names: List[str] = os.listdir(data_dir)
        for current_dir_name in list_class_dirs_names:
            current_dir: str = os.path.join(data_dir, current_dir_name)
            result = calculate_orginal_distribution(current_dir)
            per_class_orginal_distribution.append(result)

    #SPLIT 1
    per_class_proportions = []
    for it in per_class_orginal_distribution:
        per_class_proportions.append(int(it/dataset_image_count*100))

    split_1_dataset_size = 0
    for it, value in enumerate(per_class_proportions):
        split_1_dataset_size+=value

    train_size = int(split_1_dataset_size*0.7)
    val_size = int(split_1_dataset_size*0.2)
    test_size = int(split_1_dataset_size*0.1)


    train_data_split_1 = data.take(train_size)
    val_data_split_1 = data.skip(train_size).take(val_size)
    test_data_split_1 = data.skip(train_size+val_size).take(test_size)

    #SPLIT 2    
    uniform_distribution: int = int(dataset_image_count / len(NUMBER_TO_LABEL))

    per_class_proportions_split_2 = []

    for _ in list(LABELS_TO_NUMBER.keys()):
        per_class_proportions_split_2.append(int(uniform_distribution/dataset_image_count*100))


    split_2_dataset_size = 0
    for it, value in enumerate(per_class_proportions_split_2):
        split_2_dataset_size+=value
        #print(f"{NUMBER_TO_LABEL[it]}  =  {value}")

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

    best_models_list = []
    models_to_check_list = []

    model_1_epoch_10 = create_cnn_model()


    for i in range(1,3):
        models_to_check_list.append(
            ModelData(
                model_1_epoch_10,
                5*i,
                train_data_split_1,
                val_data_split_1
            )
        )

    #item.train_model([TimeHistory()])

    for index, item in enumerate(models_to_check_list):
        item.train_model()
        item.save_training_history(index)

    best_model_split_1 = find_best_split_1(models_to_check_list)
    print(f"Best model for split_1 based on epoch: {best_model_split_1.num_of_epochs}")
    save_to_file(os.path.join(os.getcwd(), "best_result_epoch.txt"), best_model_split_1.num_of_epochs)

    best_models_list.append(best_model_split_1)

    for index, item in enumerate(best_models_list):
        item.save_model(index)

    #list_of_callbacks = best_model_split_1.get_callbacks()
    #test = list_of_callbacks[CALLBACK_TO_INDEX["TIME"]]


    #Verify best and save
    #found best epoch number and over fit it

    # train_loss, train_acc = model.evaluate(train_data_split_1)
    # val_loss, val_acc = model.evaluate(test_data_split_1)
    #
    # print('Training loss:', train_loss)
    # print('Training accuracy:', train_acc)
    # print('Validation loss:', val_loss)
    # print('Validation accuracy:', val_acc)

    # #model_results = model.evaluate()
    # #model.save('weights/model.h5')

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





