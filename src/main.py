import copy
from typing import List, Tuple, Set, Dict, Optional
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

import time
import os, os.path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from colorama import Fore
from keras import regularizers


import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import keras.utils as ku

from keras.callbacks import History, ModelCheckpoint
from keras.callbacks import Callback
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras_visualizer import visualizer
from keras.models import load_model
from tensorflow.keras.metrics import CategoricalAccuracy

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
NUMBER_OF_EPOCHS: int = 50

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
    def __init__(self, model, num_of_epochs, train_data_split, val_data_split, test_data_split = None , model_index = 0):
        self.model_index = model_index
        self.model = model
        self.model_history = History()
        self.num_of_epochs = num_of_epochs
        self.train_data_split = train_data_split
        self.val_data_split = val_data_split
        self.test_data_split = test_data_split
        self.model_result = None
        self.callback_list = None
        self.accuracy_tested = None
        self.loss_tested = None
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
                                      callbacks = self.callback_list)
        self.model_result = model_result

        return model_result

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

    @staticmethod
    def create_checkpoint_callback():

        cur_dir = os.getcwd()

        path = os.path.join(os.path.dirname(cur_dir), "checkpoints_models")
        checkpoint_filepath = f"{path}//MODEL_epoch{{epoch:02d}}.h5"
        checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy',
                                              save_weights_only=False, save_best_only=True,
                                              verbose=1)
        return checkpoint_callback

    def save_model(self, name):
        cur_dir = os.getcwd()
        path = os.path.join(os.path.dirname(cur_dir), "models")
        self.model.save(os.path.join(path, f"MODEL_{name}.h5"))

    def get_callbacks(self):
        return self.callback_list

    def compare_result(self, best_model_previous_split_history,best_model_previous_split_callbacks, index = 0):

        train_loss = self.model_history.history['loss']
        val_loss = self.model_history.history['val_loss']

        train_acc = self.model_history.history['accuracy']
        val_acc = self.model_history.history['val_accuracy']

        previous_split_train_loss = best_model_previous_split_history.history['loss']
        previous_val_loss = best_model_previous_split_history.history['val_loss']

        previous_split_train_acc = best_model_previous_split_history.history['accuracy']
        previous_split_val_acc = best_model_previous_split_history.history['val_accuracy']


        size = min(len(previous_split_train_loss), len(train_loss))

        epochs = range(1, size + 1)

        plt.plot(epochs, train_loss[:size], 'b', label='Training Loss')
        plt.plot(epochs, val_loss[:size], 'r', label='Validation Loss')
        plt.plot(epochs, previous_split_train_loss[:size], "yellow", label='Previous Split Train Loss')
        plt.plot(epochs, previous_val_loss[:size], "orange", label='Previous Split Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if index is not None:
            plt.grid(True)
            plt.savefig(os.path.join(f"model_{index}", f"model_loss_compare_{index}.png"))
        else:
            plt.grid(True)
            plt.show()

            plt.savefig(os.path.join(f"model", f"model_loss_compare.png"))

        plt.clf()
        plt.plot(epochs, train_acc[:size], 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc[:size], 'r', label='Validation Accuracy')
        plt.plot(epochs, previous_split_train_acc[:size], "yellow", label='Previous Split Train Accuracy')
        plt.plot(epochs, previous_split_val_acc[:size], "orange", label='Previous Split Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if index is not None:
            plt.grid(True)
            plt.savefig(os.path.join(f"model_{index}", f"model_accuracy_compare_{index}.png"))
        else:
            plt.grid(True)
            plt.show()
            plt.savefig(os.path.join(f"model", f"model_accuracy_compare.png"))

        plt.clf()

        time_spend_per_epoch = self.callback_list[CALLBACK_TO_INDEX["TIME"]].times
        previous_time_spend_per_epoch = best_model_previous_split_callbacks[CALLBACK_TO_INDEX["TIME"]].times


        plt.plot(epochs, time_spend_per_epoch[:size], 'r', label='Time')
        plt.plot(epochs, previous_time_spend_per_epoch[:size], 'b', label='Time')
        plt.title('Time comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Time')
        plt.legend()

        if index is not None:
            plt.grid(True)
            plt.savefig(os.path.join(f"model_{index}", f"model_time_comparison_{index}.png"))
        else:
            plt.grid(True)
            plt.show()

            plt.savefig(os.path.join(f"model", f"model_time_comparison.png"))
        plt.clf()


    def test_model(self, data_split):
            self.loss_tested, self.accuracy_tested = self.model.evaluate(data_split)
            return self.accuracy_tested

def find_best_model(models_data_list: List[ModelData]):

    best_model_data: Optional[ModelData] = models_data_list[0]

    for item in models_data_list:

        if item.accuracy_tested >= best_model_data.accuracy_tested:

            best_model_data = item

    return best_model_data


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


def create_first_cnn_model(visualize = False):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), padding='same', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.summary()

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_second_cnn_model(visualize = False):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), padding='same', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))


    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))

    model.summary()

    #if visualize:
       #visualizer(model, file_format='png', view=True)

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_third_cnn_model(visualize=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
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


    #training split 1
    #perform_training_on_split_1(train_data_split_1, val_data_split_1)

    #overfiting
    #perform_overfitting_split_1(test_data_split_1, train_data_split_1)

    #training split 2
    #perform_training_on_split_2(train_data_split_2, val_data_split_2)

    #training split 3
    #perform_training_on_split_3(train_data_split_3, val_data_split_3)

    #perform_model_testing(f"../result_models/split_1/MODEL_epoch39.h5", train_data_split_1, "Train_Data_Split_1")
    #perform_model_testing(f"../result_models/split_1/MODEL_epoch39.h5", val_data_split_1, "Val_Data_Split_1")
    #perform_model_testing(f"../result_models/split_1/MODEL_epoch39.h5", test_data_split_1, "Test_Data_Split_1")


    #perform_model_testing(f"../result_models/split_2/MODEL_epoch49.h5", train_data_split_2, "Train_Data_Split_2")
    #perform_model_testing(f"../result_models/split_2/MODEL_epoch49.h5", val_data_split_2, "Val_Data_Split_2")
    #perform_model_testing(f"../result_models/split_2/MODEL_epoch49.h5", test_data_split_2, "Test_Data_Split_2")


    #perform_model_testing(f"../result_models/split_3/MODEL_epoch30.h5", train_data_split_3, "Train_Data_Split_3")
    #perform_model_testing(f"../result_models/split_3/MODEL_epoch30.h5", val_data_split_3, "Val_Data_Split_3")
    #perform_model_testing(f"../result_models/split_3/MODEL_epoch30.h5", test_data_split_3, "Test_Data_Split_3")

def perform_model_testing(model_path, data_set, data_set_name):

    model_epoch_39 = load_model(model_path)

    predictions = model_epoch_39.predict(data_set, batch_size=BATCH_SIZE, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    original_labels = []
    for images, labels in data_set:
        for i in range(len(images)):
            original_labels.append(labels[i].numpy())

    report = classification_report(np.array(original_labels), predicted_classes, output_dict=True)

    class_labels = list(report.keys())
    class_labels.remove('accuracy')
    class_labels.remove('macro avg')
    class_labels.remove('weighted avg')

    metrics = ['precision', 'recall', 'f1-score']

    class_metrics = {metric: [] for metric in metrics}

    for label in class_labels:
        for metric in metrics:
            class_metrics[metric].append(report[label][metric])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_labels))
    bar_width = 0.15

    for i, metric in enumerate(metrics):
        ax.bar(x + (i * bar_width), class_metrics[metric], bar_width, label=metric)

    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Score')
    ax.set_title(f"Classification Report Metrics for: {data_set_name}")

    ax.set_xticks(x)
    ax.set_xticklabels([NUMBER_TO_LABEL[int(e)] for e in class_labels])

    ax.legend()

    plt.grid(True)
    plt.savefig(f"{data_set_name}_metrics_results.png")

def perform_training_on_split_3(train_data_split, val_data_split):
    model_to_train = ModelData(create_first_cnn_model(),
                               NUMBER_OF_EPOCHS,
                               train_data_split,
                               val_data_split,
                               model_index=0)

    checkpoint_callback = ModelData.create_checkpoint_callback()

    model_to_train.train_model([TimeHistory(), checkpoint_callback])
    model_to_train.save_training_history(0)
    model_to_train.test_model(val_data_split)


def perform_overfitting_split_1(train_data_split, val_data_split):
    model_test = ModelData(create_first_cnn_model(),
                           50,
                           train_data_split,
                           val_data_split
                           )
    model_test.train_model([TimeHistory()])
    model_test.save_training_history(0)


def perform_training_on_split_1(train_data_split, val_data_split):

    model_to_train = ModelData(create_first_cnn_model(),
                               NUMBER_OF_EPOCHS,
                               train_data_split,
                               val_data_split,
                               model_index=0)

    checkpoint_callback = ModelData.create_checkpoint_callback()

    model_to_train.train_model([TimeHistory(), checkpoint_callback])
    model_to_train.save_training_history(0)
    model_to_train.test_model(val_data_split)

def perform_training_on_split_2(second_train_data_split, second_val_data_split):
    model_to_train = ModelData(create_first_cnn_model(),
                               NUMBER_OF_EPOCHS,
                               second_train_data_split,
                               second_val_data_split,
                               model_index=0)

    checkpoint_callback = ModelData.create_checkpoint_callback()

    model_to_train.train_model([TimeHistory(), checkpoint_callback])
    model_to_train.save_training_history(0)
    model_to_train.test_model(second_val_data_split)

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


def count_images_per_class():

    current_dir: str = os.getcwd()
    data_dir: str = os.path.join(os.path.dirname(current_dir), 'data')
    dir_list = os.listdir(data_dir)
    class_image_count = {}
    for index, directory in enumerate(dir_list):
        class_image_count[NUMBER_TO_LABEL[index]] = len(os.listdir(os.path.join(data_dir, directory)))

    return class_image_count

if __name__ == '__main__':
    res = count_images_per_class()

    #for index, class_name in enumerate(list(res.keys())):
    #    print(f"{index} : {class_name} : {res[class_name]}")

    load_dataset_and_prepare()







