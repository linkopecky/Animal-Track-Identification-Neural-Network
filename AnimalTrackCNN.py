''' Animal Track Classification CNN Project '''
## Linda Kopecky - kopec039@umn.edu
## Psy5038W - Neural Networks - UMN Fall 2022

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

# import tensorflow_hub as hub

TRAIN_TO_TEST_RATIO = 0.8
RANDOM_SEED = 2022
LEARNING_RATE = 0.01
IMG_SIZE = (100, 76)
RSZ_DIR = 'C:/Users/kopec/Documents/UMN/Fall 2022/Psy5038W/Project/GrayResizedImages/'
IMG_DIR = 'C:/Users/kopec/Documents/UMN/Fall 2022/Psy5038W/Project/Images/'
CUR_DIR = 'C:/Users/kopec/Documents/UMN/Fall 2022/Psy5038W/Project/'

'''
IMAGE PRE-PROCESSING:
    - Download images and metadata
    - Convert images to black and white?
    - Remove images with descriptions containing 'scat' or 'droppings' 'skull' 'bones' 'antler' 'eggs'
    - Resize?
    ---- End up with location, image, try to get common name
    ---- Potentially, later, see if taxa were aight
'''

## Optional - change current working directory
os.chdir(CUR_DIR)

# Read in metadata file
# metadata_all = pd.read_csv('midwestandcanada.csv',dtype={'sound_url': str, 'taxon_subspecies_name': str, 'taxon_form_name': str})
metadata_all = pd.read_csv('metadata_file_full.csv',dtype={'sound_url': str, 'taxon_subspecies_name': str, 'taxon_form_name': str})
# metadata_all = pd.read_csv('metadata_file_snowtracks.csv',dtype={'sound_url': str, 'taxon_subspecies_name': str, 'taxon_form_name': str})

def label_i(df, index_list, index, label):
    # print(index, label, df.at[index,'label'])
    df.at[index, 'label'] = label
    index_list.append(index)

def extract_my_categories(df):
    df.insert(0,'label', np.chararray(len(df)), True)
    new_indices = []
    for i in df.index:
        if df['quality_grade'][i] != "research":
            continue
        clss = df['taxon_class_name'][i]
        order = df['taxon_order_name'][i]
        family = df['taxon_family_name'][i]
        genus = df['taxon_genus_name'][i]
        if clss == "Aves":
            label_i(df, new_indices, i, "Bird")
            continue
            if order in ["Falconiformes", "Strigiformes", "Accipitriformes"]:
                label_i(df, new_indices, i, "Bird of prey")
            elif order == "Passeriformes":
                label_i(df, new_indices, i, "Small bird")
        elif clss == "Mammalia":
            if order == "Artiodactyla":
                # continue
                label_i(df, new_indices, i, "Deer/Moose")
            elif order == "Lagomorpha":
                label_i(df, new_indices, i, "Rabbit")
            elif order == "Carnivora":
                if family == "Canidae":
                    # label_i(df, new_indices, i, "Dog/Fox/Cat")
                    if genus == "Canis":
                        label_i(df, new_indices, i, "Dog")
                    # elif genus in ["Urocyon", "Vulpes"]:
                    #     # continue
                    #     label_i(df, new_indices, i, "Fox")
                elif family == "Felidae":
                    # label_i(df, new_indices, i, "Dog/Fox/Cat")
                    # if genus == "Felis":
                    #     continue
                    #     label_i(df, new_indices, i, "Domestic cat")
                    # else:
                    label_i(df, new_indices, i, "Cat")
                elif family == "Procyonidae":
                    label_i(df, new_indices, i, "Raccoon")
                elif family == "Ursidae":
                    # continue
                    label_i(df, new_indices, i, "Bear")
            elif order == "Rodentia":
                if family == "Sciuridae":
                    label_i(df, new_indices, i, "Squirrel")
    print(new_indices[:20])
    new_df = df[df.index.isin(new_indices)]
    new_df.reset_index(drop=True, inplace=True)
    return new_df

# Remove samples with descriptions that hint at non-track
# image subject matter (scat, skeleton, eggs, nest)
def remove_scat(df):
    blacklist_words = [ "scat", "dropping", "poop", "fecal", "pellet",
                        "skeleton", "bone", "skull", "jawbone", "antler", "fossa"
                        "egg", "nest", "dam", "feather",
                        "dead", "body", "decapitated", "roadkill", "carcass", "killed", "impailed"]
    pattern = r'(\b{}\b)'.format('|'.join(blacklist_words))
    new_df = df[~df["description"].str.contains(pattern, case=False, na=False)]
    new_df.reset_index(drop=True, inplace=True)
    print(str(len(df) - len(new_df)) + " samples trimmed due to description.")
    return new_df

def filter_data(metadata):
    if os.path.exists("filtered_categorized_researchgrade.csv"):
        data = pd.read_csv('filtered_categorized_researchgrade.csv',dtype={'sound_url': str, 'taxon_subspecies_name': str, 'taxon_form_name': str})
        return data
    data_categorized = extract_my_categories(metadata)
    print(str(len(data_categorized))+" categorized research grade samples.")
    data_filtered = remove_scat(data_categorized)    # 3383 samples trimmed due to description. 24210 total samples remaining.
    # data_filtered.to_csv("filtered_categorized_researchgrade.csv")
    # data_filtered.to_csv("filtered_categorized_researchgrade_midc.csv")
    return data_filtered
    
data = filter_data(metadata_all)
NUM_SAMPLES, NUM_TABS = data.shape
# print(  str(len(metadata_all) - len(data)) + " samples trimmed due to description. " +
#         str(NUM_SAMPLES)+" total samples remaining."  )
print(str(NUM_SAMPLES)+" total samples remaining.")

''' Crop to center '''
def center_crop(img, new_size):
    new_w, new_h = new_size
    w,h = img.size
    left = w//2 - new_w//2
    upper = h//2 - new_h//2
    right = w//2 + new_w//2
    bottom = h//2 + new_h//2
    return img.crop((left,upper,right,bottom))

def download_images(file_path, data):
    if len(os.listdir(file_path)) > 0:
        # Assume images have been loaded if non-empty directory
        return os.listdir(file_path)
    img_filenames = []
    for i in range(NUM_SAMPLES): # 13118, 
        failures = []
        filename = str(i) + '.jpg'
        try:
            response = requests.get(data['image_url'][i])
        except requests.exceptions.Timeout:
            failures.append((i,data['id'][i]))
        except requests.exceptions.TooManyRedirects:
            failures.append((i,data['id'][i]))
        except requests.exceptions.RequestException as e:
            # failures.append((i,metadata['id'][i]))
            raise SystemExit(e)
        img = open(file_path+filename, "wb")
        result = img.write(response.content)    ## error handle this later
        img.close()
        img_filenames.append(filename)
    return img_filenames
# og_img_filenames= [str(i)+".jpg" for i in range(NUM_SAMPLES)]
og_img_filenames = download_images(IMG_DIR, data)

# data = data.drop("orig_image_filenames", axis=1)
data.insert(0,"orig_image_filenames", og_img_filenames, True)
data.reset_index(drop=True, inplace=True)

''' Resize all images for input consistency
    Convert to grayscale 
    Save to new directory '''
def process_images(new_dir, orig_dir, new_size):
    img_filepaths = [new_dir+str(i)+".jpg" for i in range(NUM_SAMPLES)]
    if len(os.listdir(new_dir)) > 0:
    # Assume images have been loaded if non-empty directory
        return img_filepaths
    for i in range(NUM_SAMPLES):  # img in os.list... results in list out of order
    # for i in range(len(metadata)):
        # img = (str(i)+'.jpg')
        # if not os.path.exists(new_dir+img):
        filename = str(i)+".jpg"    # kind of had to hard-code this
        im = Image.open(orig_dir+filename)
        # im.show()
        im_gray = im.convert(mode="L")
        im_gray.thumbnail((max(new_size)+25,max(new_size)+25))
        if im_gray.size[1] > im_gray.size[0]:
            # im_gray.show()
            im_gray = im_gray.rotate(90, expand=True)
            # im_gray.show()
        im_resized = center_crop(im_gray, new_size)
        # im_resized.show()
        im_resized.save(new_dir+filename)
    return img_filepaths

rtt = [rsz_img_filepaths[i] for i in range(NUM_SAMPLES) if data['label'][i] not in ["Bird of prey", "Small bird","Deer/Moose","Fox","Bear","Domestic cat"]]
rsz_img_filepaths = rtt

rsz_img_filepaths = process_images(RSZ_DIR, IMG_DIR, IMG_SIZE)
len(rsz_img_filepaths)
rsz_img_filepaths[-1]

# data = data.loc[:,~data.columns.duplicated()].copy()
# data = data.drop("rsz_image_filepaths", axis=1)
data = data.drop("label", axis=1)
data.insert(0,"rsz_image_filepaths", rsz_img_filepaths, True)
data.reset_index(drop=True, inplace=True)
# data['rsz_image_filepaths']

def get_classes(df, column):
    labels_class = df[column]
    # classes = labels_class.unique()
    classes = labels_class.unique()
    labels_class.reset_index(drop=True, inplace=True)
    class_dict = {}
    for i in range(len(classes)):
        class_dict[classes[i]] = i
    labels_num = np.array([class_dict[label] for label in labels_class])
    return labels_num, labels_class, classes

labels_num, labels_class, classes = get_classes(data, "label")
NUM_CLASSES = len(classes)

def show_classes(labels_class):
    # labels_class.value_counts().index.tolist()
    labels_class.value_counts().plot(kind='bar',color='rosybrown')
    plt.ylabel("# of Entries")
    plt.xlabel("Label")
    plt.tight_layout()
    plt.show()

show_classes(labels_class)

''' Load images as 2D numpy arrays
    Normalize ?? '''
def load_images(img_paths, load_file_path, use_load_file=False):
    if use_load_file:
        if os.path.exists(load_file_path):
            return np.load(load_file_path)
    # If use_load_file option is true but no load file exists
    # at the path, create new load file
    new_set = []
    for i in range(len(img_paths)):
        if i%5000 == 0:
            print(str(i)+" images loaded.")
        img_path = img_paths[i]
        img = Image.open(img_path)
        img_array = np.array(img)
        new_set.append(img_array / 255.0)
    # print("done")
    arr = np.array(new_set)
    np.save(load_file_path, arr)
    return arr

# loaded_images = load_images(rsz_img_filepaths, CUR_DIR+"loaded_images_array")
# loaded_images = load_images(rsz_img_filepaths, CUR_DIR+"loaded_images_array.npy", use_load_file=True)
loaded_images = load_images([], CUR_DIR+"loaded_images_array.npy", use_load_file=True)


def add_channels(set):
    new_set = []
    for img in set:
        img_arr = np.expand_dims(img, axis=-1)
        img_tensor = img_arr.repeat(3, axis=-1)
        new_set.append(img_tensor)
    return np.array(new_set)

''' Necessary for grayscale images into CNN with 3x3 kernel '''
def get_3ch_images(images):
    if os.path.exists(CUR_DIR+"images_3ch_array"):
        return np.load(CUR_DIR+"images_3ch_array")
    arr = add_channels(images)
    np.save(CUR_DIR+"images_3ch_array",images_3ch)
    return arr

images_3ch = get_3ch_images(loaded_images)
images_3ch.shape

images_3ch = np.load(CUR_DIR+"images_3ch_array")

def get_weights():
    weights = []
    counts = labels_class.value_counts()
    for cls in classes:
        # weights[classes[i]] = 1.0/counts[i]
        weights.append(1.0/counts[cls])
    norm_factor= np.mean(weights)
    for i in range(NUM_CLASSES):
        weights[i]= weights[i]/norm_factor
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    rounded = [round(w,4) for w in weights]
    text = np.array([classes,rounded]).T
    text.shape
    # rC = ['white']*NUM_CLASSES
    # rC[3] = 'lightgrey'
    # df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
    ax.table(cellText=text, colLabels=("Class","Weights"), loc='center', cellLoc='center', # str(classes[3]),str(rounded[3])
            colColours=['rosybrown']*2)#, colWidths=[20,20])
    # fig.tight_layout()
    plt.show()
    weights_dict= {}
    for i in range(NUM_CLASSES):
        weights_dict[i] = weights[i]
    return weights_dict

weights = get_weights()

''' Split input data into training and testing sets '''
training_images, testing_images, training_labels, testing_labels = train_test_split(images_3ch, labels_num, test_size=.2)
# training_images, testing_images, training_labels, testing_labels = train_test_split(loaded_images, labels_num, test_size=.2)
training_images_temp, testing_images, training_labels_temp, testing_labels = train_test_split(loaded_images, labels_num, test_size=.2)
training_images, validation_images, training_labels, validation_labels = train_test_split(training_images_temp, training_labels_temp, test_size=.2)
len(training_images)
len(validation_images)
len(testing_images)
# training_images.reset_index(drop=True, inplace=True)
# training_labels.reset_index(drop=True, inplace=True)
# testing_images.reset_index(drop=True, inplace=True)
# testing_labels.reset_index(drop=True, inplace=True)

# y_train_short = y_train.head(5000)
# y_train_short.reset_index(drop=True, inplace=True)
# y_test_short = y_test.head(5000)
# y_test_short.reset_index(drop=True, inplace=True)
# X_train_short = X_train_paths.head(5000)
# X_train_short.reset_index(drop=True, inplace=True)
# X_test_short = X_test_paths.head(5000)
# X_test_short.reset_index(drop=True, inplace=True)

# plt.imshow(images_3ch[7], cmap=plt.cm.binary_r)
# plt.xlabel(labels_class[7])
# plt.show()

# l = [20,424,606,410,303,296,308,312,304,220,196,672,700,263,502,511,522,500,413,149,211,685,670,210,205]
# len(l)
# np.save("ffffffftrainimg.npy",training_images)
# np.save("fffffftraininglab.npy",training_labels)
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     j = i+666
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(training_images[l[i]], cmap=plt.cm.binary_r)
#     plt.xlabel(classes[training_labels[l[i]]])
#     # plt.imshow(training_images[j], cmap=plt.cm.binary_r)
#     # plt.xlabel(classes[training_labels[j]]+str(j))
#     # plt.tight_layout()
# plt.show()

''' Basic Neural Network '''
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(76, 100)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(NUM_CLASSES)
# ])

# inp_shape = [IMG_SIZE[1], IMG_SIZE[0], 3]

''' Basic CNN '''
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape = [IMG_SIZE[1], IMG_SIZE[0], 1]),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation ='softmax')
])

''' Edited CNN '''
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = [IMG_SIZE[1], IMG_SIZE[0], 1]),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUM_CLASSES, activation ='softmax')
])

model.compile(optimizer='adam',
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #'binary_crossentropy', # 
              metrics=['accuracy'])

# tf.keras.utils.plot_model(
#     model#,
#     # to_file="model.png",
#     # show_shapes=True,
#     # show_layer_names=True,
#     # rankdir="TB",
#     # expand_nested=True,
#     # dpi=96,
# )
# plt.show()

from time import time

# training_images.shape
# len(training_labels)
start = time()
# history = model.fit(training_images, training_labels, epochs=25)
history = model.fit(training_images, training_labels, epochs=125, validation_data=(validation_images, validation_labels), class_weight=weights, validation_steps=48, steps_per_epoch=190)
end = time()
print("Time elapsed: "+str(end-start)) # 1461.156991481781
metrics_df = pd.DataFrame(history.history).plot(figsize=(8,5))
# metrics_df[["loss"]].plot()
# plt.show()
# metrics_df[["accuracy"]].plot()
plt.xlabel("Epochs")
plt.show()

from tensorflow.keras.models import load_model

model.save("C:/Users/kopec/Documents/UMN/Fall 2022/Psy5038W/Project/24000model125epochs")
# y_train[0]
# 'White-tailed Deer'
model.summary()
# test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=2)
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=2)
print('\nTest accuracy:', test_acc)
# Test accuracy: 0.072720967233181 0.1636

# tf.keras.utils.plot_model(
#     model,
#     to_file='modelo.png',
#     show_shapes=False,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=False
# )

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(testing_images)

print("Evaluate on test data")
results = model.evaluate(testing_images, testing_labels, batch_size=128)
print("test loss, test acc:", results)

# predictions = probability_model.predict(testing_images)

# np.save("C:/Users/kopec/Documents/UMN/Fall 2022/Psy5038W/Project/predictions", predictions)
# c = enumerate(y_train)

# tf.keras.utils.plot_model(
#     model,
#     to_file='C:/Users/kopec/Documents/UMN/Fall 2022/Psy5038W/Project/modelll.jpg',
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=False
# )

# class_dict = { i : classes[i] for i in range(NUM_CLASSES)}
# class_dict = dict.fromkeys(classes)
# class_dict = { classes[i] : i for i in range(NUM_CLASSES)}
# class_dict = {}
# for i in range(NUM_CLASSES):
#     class_dict[classes[i]] = i
# y_train_short_nums = [class_dict[label] for label in y_train_short]
# y_test_short_nums = [class_dict[label] for label in y_train_short]


'''
Test accuracy: 0.0754
>>> 1/NUM_CLASSES
0.0009560229445506692
>>>
'''

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                100*np.max(predictions_array),
                                classes[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(NUM_CLASSES))
  plt.yticks([])
  thisplot = plt.bar(range(NUM_CLASSES), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

for i in range(73,75):
# i = 51
    # if testing_labels[i] == 1:
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], testing_labels, testing_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  testing_labels)
    plt.show()

# from sklearn.metrics import plot_confusion_matrix
# plot_confusion_matrix(model, testing_images, testing_labels)

class estimator:
  _estimator_type = ''
#   classes_=[]
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self.classes_ = classes
  def predict(self, X):
    y_prob= self.model.predict(X)
    y_pred = y_prob.argmax(axis=1)
    return y_pred

classifier = estimator(model, classes)
# 13/13 - 0s - loss: 8.1242 - accuracy: 0.1990 - 436ms/epoch - 34ms/step
# plot_confusion_matrix(estimator=classifier, X=testing_images, y_true=testing_labels)

from sklearn.metrics import plot_confusion_matrix

figsize = (8,8)
plot_confusion_matrix(estimator=classifier, X=testing_images, include_values=False, y_true=testing_labels, cmap='Blues', normalize='true', ax=plt.subplots(figsize=figsize)[1])
# plot_confusion_matrix(estimator=classifier, X=testing_images, y_true=testing_labels, cmap='Blues', normalize='true', ax=plt.subplots(figsize=figsize)[1])
plt.tight_layout()
plt.show()
counts
# batch_size = 32
# # train_dl = DataLoader()

from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = tf.keras.applications.vgg16.VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

input_shape = (76, 100, 3)
optim_1 = tf.keras.optimizers.Adam(learning_rate=0.001)

vgg16_model = create_model(input_shape, NUM_CLASSES, optim_1, fine_tune=0)


vgg16_model = tf.keras.applications.VGG16(
              weights="imagenet", 
              input_shape=(76, 100, 3),
              include_top = False)
vgg16_model.summary()
vgg_history = vgg16_model.fit(training_images)
predictions = vgg16_model.predict(testing_images)
predicted_class = tf.keras.applications.vgg16.decode_predictions(predictions, top=5)
# predicted_class
from livelossplot.inputs.keras import PlotLossesCallback

plot_loss_1 = PlotLossesCallback()

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = tf.keras.callbacks.ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')

vgg_history = vgg16_model.fit(training_images, training_labels)


# model = models.resnet50(pretrained=True)
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 5)

# def forward(xb):
#     return torch.sigmoid(model(xb))

# epochs = 10
# opt_func = torch.optim.Adam
# lr = 0.001
# data_dir ="../input/flowers-recognition/flowers/flowers"
# data_dir = new_dir

# database = ImageFolder(data_dir)

# evaluate(model, test_dl)

# train,test = make_sets()
# train.reset_index(drop=True, inplace=True)
# test.reset_index(drop=True, inplace=True)
# train.shape
# test.shape

# # train_loader = DataLoader(train, batch_size=10, shuffle=False)
# # test_loader = DataLoader(test, 10)
# # for (images, labels) in train_loader:
# #     print(images)

# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(250,300))
# ])

# # expand_grayscale_image_channels
# def add_channels(img):
#     img_arr = np.array(img)
#     img_arr = np.expand_dims(img_arr, axis=-1)
#     img_tensor = img_arr.repeat(3, axis=-1)
#     return img_tensor
# pp = Image.open(img_filepaths[0])
# xx = add_channels(pp)
# # pp = im_resized

# def tf_load_process_image(filename):
#   img_size = 224
#   # Load image (in PIL image format by default)
# #   image_original = load_img(filename, target_size=(img_size, img_size))
#   image_original = Image.open(filename)
#   print("Image size after loading", image_original.size)
# #   image_original.show()
#   # Convert from numpy array
#   image_array = add_channels(image_original)
#   print("Image size after expanding channels", image_array.shape)
# #   image_array.show()
#   # Expand dims to add batch size as 1
#   image_batch = np.expand_dims(image_array, axis=0)
#   print("Image size after expanding dimension", image_batch.shape)

#   # Preprocess image
#   image_preprocessed = tf.keras.applications.resnet50.preprocess_input(image_batch)

#   return image_preprocessed

# img_preprocessed = tf_load_process_image("C:/Users/kopec/Documents/UMN/Fall 2022/Psy5038W/Project/GrayResizedImages/0.jpg")

# IMG_SHAPE = (250, 300, 3) # IMG_SIZE + (3,)

# XX = tf.keras.applications.resnet50.preprocess_input(XX)

# base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False, classes=NUM_CLASSES)
# base_model.summary()
# predictions = base_model.predict(img_preprocessed)

# predicted_class = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)

# # image_batch = Image.open(next(iter(X_train)))
# image_batch = X_train[0:5]
# XX = [add_channels(Image.open(p)) for p in image_batch]
# feature_batch = base_model(img_preprocessed)
# print(feature_batch.shape)










# # Load a pre-trained TF-Hub module for extracting features from images. We've
# # chosen this particular module for speed, but many other choices are available.
# image_module = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2')

# # import tensorflow.compat.v1 as tf
# # tf.disable_eager_execution()

# # # Preprocessing images into tensors with size expected by the image module.
# # encoded_images = tf.placeholder(tf.string, shape=[None])
# # image_size = hub.get_expected_image_size(image_module)

# # def decode_and_resize_image(encoded):
# #   decoded = tf.image.decode_jpeg(encoded, channels=3)
# #   decoded = tf.image.convert_image_dtype(decoded, tf.float32)
# #   return tf.image.resize_images(decoded, image_size)

# # batch_images = tf.map_fn(decode_and_resize_image, encoded_images, dtype=tf.float32)

# # # The image module can be applied as a function to extract feature vectors for a
# # # batch of images.
# ''' Extract features '''
# # features = image_module(batch_images)
# features = image_module(train)

# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# import numpy as np
# model = VGG16(weights='imagenet', include_top=False)
# model.summary()
# img_path = metadata['image_filepaths'][543]
# # img = image.load_img(img_path, target_size=(400,400))
# img = Image.open(img_path)
# img_data = np.array(img)
# img_data = np.expand_dims(img_data, axis=0)

# vgg16_feature = model.predict(img_data)


# def create_model(features):
#   """Build a model for classification from extracted features."""
#   # Currently, the model is just a single linear layer. You can try to add
#   # another layer, but be careful... two linear layers (when activation=None)
#   # are equivalent to a single linear layer. You can create a nonlinear layer
#   # like this:
#   # layer = tf.layers.dense(inputs=..., units=..., activation=tf.nn.relu)
#   layer = tf.layers.dense(inputs=features, units=NUM_CLASSES, activation=None)
#   return layer

# # For each class (kind of flower), the model outputs some real number as a score
# # how much the input resembles this class. This vector of numbers is often
# # called the "logits".
# logits = create_model(features)
# labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# # Mathematically, a good way to measure how much the predicted probabilities
# # diverge from the truth is the "cross-entropy" between the two probability
# # distributions. For numerical stability, this is best done directly from the
# # logits, not the probabilities extracted from them.
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
# cross_entropy_mean = tf.reduce_mean(cross_entropy)

# # Let's add an optimizer so we can train the network.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
# train_op = optimizer.minimize(loss=cross_entropy_mean)

# # The "softmax" function transforms the logits vector into a vector of
# # probabilities: non-negative numbers that sum up to one, and the i-th number
# # says how likely the input comes from class i.
# probabilities = tf.nn.softmax(logits)

# # We choose the highest one as the predicted class.
# prediction = tf.argmax(probabilities, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))

# # The accuracy will allow us to eval on our test set. 
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## Train the network

# # How long will we train the network (number of batches).
# NUM_TRAIN_STEPS = 100
# # How many training examples we use in each step.
# TRAIN_BATCH_SIZE = 10
# # How often to evaluate the model performance.
# EVAL_EVERY = 10

# def get_batch(batch_size=None, test=False):
#   """Get a random batch of examples."""
#   examples = TEST_EXAMPLES if test else TRAIN_EXAMPLES
#   batch_examples = random.sample(examples, batch_size) if batch_size else examples
#   return batch_examples

# def get_images_and_labels(batch_examples):
#   images = [get_encoded_image(e) for e in batch_examples]
#   one_hot_labels = [get_label_one_hot(e) for e in batch_examples]
#   return images, one_hot_labels

# def get_label_one_hot(example):
#   """Get the one hot encoding vector for the example."""
#   one_hot_vector = np.zeros(NUM_CLASSES)
#   np.put(one_hot_vector, get_label(example), 1)
#   return one_hot_vector

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   for i in range(NUM_TRAIN_STEPS):
#     # Get a random batch of training examples.
#     train_batch = get_batch(batch_size=TRAIN_BATCH_SIZE)
#     batch_images, batch_labels = get_images_and_labels(train_batch)
#     # Run the train_op to train the model.
#     train_loss, _, train_accuracy = sess.run(
#         [cross_entropy_mean, train_op, accuracy],
#         feed_dict={encoded_images: batch_images, labels: batch_labels})
#     is_final_step = (i == (NUM_TRAIN_STEPS - 1))
#     if i % EVAL_EVERY == 0 or is_final_step:
#       # Get a batch of test examples.
#       test_batch = get_batch(batch_size=None, test=True)
#       batch_images, batch_labels = get_images_and_labels(test_batch)
#       # Evaluate how well our model performs on the test set.
#       test_loss, test_accuracy, test_prediction, correct_predicate = sess.run(
#         [cross_entropy_mean, accuracy, prediction, correct_prediction],
#         feed_dict={encoded_images: batch_images, labels: batch_labels})
#       print('Test accuracy at step %s: %.2f%%' % (i, (test_accuracy * 100)))

# def show_confusion_matrix(test_labels, predictions):
#   """Compute confusion matrix and normalize."""
#   confusion = sk_metrics.confusion_matrix(
#     np.argmax(test_labels, axis=1), predictions)
#   confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
#   axis_labels = list(CLASSES.values())
#   ax = sns.heatmap(
#       confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
#       cmap='Blues', annot=True, fmt='.2f', square=True)
#   plt.title("Confusion matrix")
#   plt.ylabel("True label")
#   plt.xlabel("Predicted label")

# show_confusion_matrix(batch_labels, test_prediction)






# model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
#                             tf.keras.layers.Dense(128, activation='relu'),
#                             tf.keras.layers.Dense(1)
#                             ])

# X_train_imgs.shape
# (N_samples, width, height, channels)
# X_train_meta.shape  # add in latitude and longitude data
# (N_samples, features)

# y_train.shape   # common name
# (N_samples, outputs)

# model.fit(inputs = [X_train_imgs, X_train_meta])


# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img

# model = ResNet50(weights="imagenet", include_top=False)
