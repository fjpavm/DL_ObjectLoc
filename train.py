import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
import ProfilingUtils
import JsonUtils
import random

dataset_path = dict()
dataset_path['syn'] = "../EdIntelligence-Neurolabs-Hackathon/common/train/syn/"
dataset_path['real'] = "../EdIntelligence-Neurolabs-Hackathon/challenge-2/train/real/"
image_sub_path = "images/"
challenge1_syn_csv = "annotations-detection.csv"
challenge2_syn_csv = "annotations-localization.csv"
challenge2_real_csv = "annotations.csv"

Image_Width = 1920
Image_Height = 1080
NumClasses = 330 + 1

BATCH_SIZE = 1

# Reads annotation file into a full annotations list, as well as by image and by class dicts
# annotation is dict with keys:
#   image: image name string
#   class: class name string
#   centre: np.array with x and y float pixel coordinates
#   size: np.array with x and y float pixel dimentions
#   source: where data comes from (default is syn)
# returns annotations_list, annotations_by_image, annotations_by_class
def read_annotation_file(path: str, source = 'syn'):
    annotations_by_image = dict()
    annotations_by_class = dict()
    annotations_list = list()
    with open(path) as fp:
        lines = fp.readlines()
        for line in lines:
            try:
                image_name, xmin, ymin, xmax, ymax, class_name = line.strip().split(',')
                xmin = float(xmin)
                xmax = float(xmax)
                ymin = float(ymin)
                ymax = float(ymax)
                annotation = {  "image":image_name, 
                                "centre" : np.array([(xmax+xmin)/2, (ymax+ymin)/2]), 
                                "size" : np.array([(xmax-xmin), (ymax-ymin)]), 
                                "class" : class_name,
                                "source" : source
                            }
                if image_name not in annotations_by_image:
                    annotations_by_image[image_name] = list()
                annotations_by_image[image_name].append(annotation)
                if class_name not in annotations_by_class:
                    annotations_by_class[class_name] = list()
                annotations_by_class[class_name].append(annotation)
                annotations_list.append(annotation)
            except:
                print(f"{path} : problem reading line: {line}")
    return annotations_list, annotations_by_image, annotations_by_class


def createFigure(title=''):
    fig = plt.figure()
    #fig = matplotlib.figure.Figure
    fig.suptitle(title,fontsize=14, fontweight='bold')
    graph = fig.add_subplot(1,1,1)
    #graph = matplotlib.axes.Axes 
    graph.axis("off") 

    return fig

# Add a 5 pixel border in red channel
def highlightAnnotation(in_image, annotation):
    BBMin = annotation["centre"]-annotation["size"]/2
    BBMax = annotation["centre"]+annotation["size"]/2
    BBMin = np.around(BBMin).astype(int)
    BBMax = np.around(BBMax).astype(int)

    for y in range(BBMin[1], BBMax[1]):
        for x in range(BBMin[0], BBMax[0]):
            if x-BBMin[0] < 5 or BBMax[0]-x < 5 or y-BBMin[1] < 5 or BBMax[1]-y < 5 :
                in_image[y][x][0] = 1.0

def extractBestAnnotationForBlock(in_blockCentre, in_image_annotations):
    minOverlap = 16 #pixels
    blockRangeHalfSize = (32.0*7)/2
    bestAnnotation = None
    bestDistance = np.inf
    # check for objects in block range with at least 16 pixel width and height overlap
    # and from these choose object annotation closest to centre as best
    for annotation in in_image_annotations:
        annotationCentre = annotation['centre']
        annotationHalfSize = annotation['size']/2
        centreDiff = annotationCentre-in_blockCentre
        if (abs(centreDiff[0]) < blockRangeHalfSize + annotationHalfSize[0] - minOverlap) and (abs(centreDiff[1]) < blockRangeHalfSize + annotationHalfSize[0] - minOverlap): 
            centreDistance = np.linalg.norm(annotationCentre-in_blockCentre)
            if centreDistance < bestDistance:
                bestAnnotation = annotation
    return bestAnnotation

def createTrainingForImage(in_imageName, in_annotations_by_image, in_classMap):
    image_annotations = in_annotations_by_image[in_imageName]
    #TODO: use input image dimentions to calculate prediction shapes
    BBPrediction = np.zeros((34,60,4))
    ObjectPrediction = np.zeros((34,60),dtype=int)
    ClassPrediction = np.zeros((34,60),dtype=int)

    for blockY in range(34):
        for blockX in range(60):
            blockCentre = np.array([blockX*32 +16, blockY*32+16], dtype=float)
            bestAnnotation = extractBestAnnotationForBlock(blockCentre, image_annotations)
            if bestAnnotation == None:
                # default no object to X and Y size of block (32 pixels)
                BBPrediction[blockY][blockX][2] = 32
                BBPrediction[blockY][blockX][3] = 32
            else:
                centre = bestAnnotation['centre']-blockCentre
                size = bestAnnotation['size']
                BBPrediction[blockY][blockX][0] = centre[0] 
                BBPrediction[blockY][blockX][1] = centre[1]
                BBPrediction[blockY][blockX][2] = size[0]
                BBPrediction[blockY][blockX][3] = size[1]
                ObjectPrediction[blockY][blockX] = 1
                ClassPrediction[blockY][blockX] = in_classMap['toInt'][bestAnnotation['class']]
    return BBPrediction, ObjectPrediction, ClassPrediction

class DatasetFromImageNames_Generator(keras.utils.Sequence) :

    def __init__(self, in_image_names, in_batch_size, in_annotations_by_image, in_classMap) :
        self.m_image_names = in_image_names
        self.m_batch_size = in_batch_size        
        self.m_annotations_by_image = in_annotations_by_image
        self.m_classMap = in_classMap

    def __len__(self) :
        return (np.ceil(len(self.m_image_names) / float(self.m_batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        batch_image_names = self.m_image_names[idx * self.m_batch_size : (idx+1) * self.m_batch_size]

        batch_images = list()
        batch_BBoxes = list()
        batch_Objects = list()
        batch_Classes = list()

        #Create batch training data from filenames
        for image_name in batch_image_names:
            source = self.m_annotations_by_image[image_name][0]['source']
            image_path = os.path.join(dataset_path[source], image_sub_path, image_name) 
            image = plt.imread(image_path)
            preprocessed = keras.applications.mobilenet_v2.preprocess_input(image)
            batch_images.append(preprocessed)

            BBPrediction, ObjectPrediction, ClassPrediction = createTrainingForImage(image_name, self.m_annotations_by_image, self.m_classMap)
            batch_BBoxes.append(BBPrediction)
            batch_Objects.append(ObjectPrediction)
            batch_Classes.append(ClassPrediction)

        return np.array(batch_images), [np.array(batch_BBoxes), np.array(batch_Objects), np.array(batch_Classes)]

def separateData(list_of_data, numTrain=200, numValidate=30, numTest=200):
    list_of_data_copy = list(list_of_data)
    random.shuffle(list_of_data_copy)
    train_list = list_of_data_copy[0:numTrain]
    validate_list = list_of_data_copy[numTrain:numTrain+numValidate]
    test_list = list_of_data_copy[numTrain+numValidate:numTrain+numValidate+numTest]
    return train_list, validate_list, test_list



if __name__ == '__main__':
    challenge1_csv_path = os.path.join(dataset_path['syn'], challenge1_syn_csv) 
    annotations_list, annotations_by_image, annotations_by_class = read_annotation_file(challenge1_csv_path)
    # create classMap for this run
    classMap = dict()
    toIntMap = dict()
    toClassMap = ['NO_OBJECT'] + list(annotations_by_class.keys())
    for classInt in range(len(toClassMap)):
        toIntMap[toClassMap[classInt]] = classInt
    classMap['toInt'] = toIntMap
    classMap['toClass'] = toClassMap

    print(f"Number of annotations: {len(annotations_list)}")
    print(f"Number of images: {len(annotations_by_image.keys())}")
    print(f"Number of classes: {len(annotations_by_class.keys())}")
 #   min_area =  np.inf
 #   min_size = []
 #   max_area = -np.inf
 #   max_size = []
 #   max_height = max_width = -np.inf
 #   size_list = list()
 #   line = 0
 #   too_small = 0
 #   for annotation in annotations_list:
 #       line+=1
 #       size = annotation["size"]
 #       size_list.append(size)
 #       if size[0] * size[1] <= 500:
 #           too_small += 1
 #           #print(f"{line} : {annotation}")
 #           continue
 #       area = size[0]*size[1]
 #       if area < min_area:
 #           min_size = size
 #           min_area = area
 #       if area > max_area:
 #           max_size = size
 #           max_area = area
 #       max_width = np.fmax(max_width, size[0])
 #       max_height = np.fmax(max_height, size[1])
 #   print(f"Num too small: {too_small}")
 #   print(f"Avg size: {np.mean(size_list, axis=0)}")
 #   print(f"Min area size: {min_size}")
 #   print(f"Max area size: {max_size}")
 #   print(f"Max size: {np.array([max_width,max_height])}")

   # Avg size: [51.80558886 82.77852721]
   # Min area size: [72.  7.]
   # Max area size: [280. 196.]
   # Max size: [352. 236.]

   # mobileNetV2 predictions shape: (1, 34, 60, 1280)
   # BBmodel predictions shape: (1, 34, 60, 4)
   # ObjModel predictions shape: (1, 34, 60, 2)
   # ClassModel predictions shape: (1, 34, 60, NumClasses)
   # image shape: (1080, 1920, 3)
   # preprocessed image shape: (1080, 1920, 3)

    input_layer = keras.Input(shape=(Image_Height, Image_Width, 3))    
    mobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_tensor=input_layer)
    mobileNetV2.trainable = False
    mobileNet_out = mobileNetV2(input_layer, training=False) 
    intermediate_net_1 = keras.layers.Conv2D(400, (1,1), padding='same', activation='relu', name='learning_input_reduction')(mobileNet_out)
    boundingBox_out = keras.layers.Conv2D(4, (7,7), padding='same', activation='linear', name='BB_out')(intermediate_net_1)
    object_out = keras.layers.Conv2D(2, (7,7), padding='same', activation='softmax', name='obj_out')(intermediate_net_1)
    intermediate_net_2 = keras.layers.DepthwiseConv2D((7,7), padding='same', activation='relu', name='class_net')(intermediate_net_1)
    class_out = keras.layers.Conv2D(NumClasses, (1,1), padding='same', activation='softmax', name='class_out')(intermediate_net_2)

    #mobileNetV2.summary()
    
    BBmodel = keras.Model(input_layer, boundingBox_out)
    ObjModel = keras.Model(input_layer, object_out)
    ClassModel = keras.Model(input_layer, class_out)

    model = keras.Model(input_layer, [boundingBox_out, object_out, class_out])
    model.summary()
    sparce_top5 = keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    
    model.compile(optimizer='adam', 
                    loss=['mse', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], 
                    loss_weights=[0.5, 1.0, 2.0],
                    metrics={'BB_out':keras.metrics.RootMeanSquaredError(), 'obj_out':'sparse_categorical_accuracy', 'class_out':['sparse_categorical_accuracy', sparce_top5]})

    if(not os.path.exists('test_names.json')):
        train_names, validate_names, test_names = separateData(annotations_by_image.keys())
        test_names_json = dict()
        test_names_json['name_list'] = test_names
        JsonUtils.writeJsonToFile('test_names.json',test_names_json)
    else:
        test_names_json = JsonUtils.readJsonFromFile('test_names.json')
        test_names = test_names_json['name_list']
        names_list = [image_name for image_name in annotations_by_image.keys() if image_name not in test_names]
        train_names, validate_names, tmp = separateData(annotations_by_image.keys(),numTest=0)

    train_gen = DatasetFromImageNames_Generator(train_names,BATCH_SIZE,annotations_by_image,classMap)
    val_gen = DatasetFromImageNames_Generator(validate_names,BATCH_SIZE,annotations_by_image,classMap)
    test_gen = DatasetFromImageNames_Generator(test_names,BATCH_SIZE,annotations_by_image,classMap)

    model_checkpoint=keras.callbacks.ModelCheckpoint('../weights/EdIntel_weights{epoch:02d}.h5',save_weights_only=True)

    if(os.path.exists('last_weights.h5')):
        model.load_weights('last_weights.h5')
    model.fit(train_gen, batch_size=BATCH_SIZE, epochs=10, validation_data=val_gen, callbacks=[model_checkpoint])
    model.save_weights('last_weights.h5')
    """  img_count = 0
    images = dict()
    image_path = os.path.join(dataset_path['syn'], image_sub_path) 
    for image_name in annotations_by_image.keys():
        image = plt.imread(os.path.join(image_path, image_name))
        img_count+=1
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(image)
        batch_preprocessed = np.array([preprocessed])
        predictions = ObjModel.predict(batch_preprocessed)

        BBPrediction, ObjectPrediction, ClassPrediction = createTrainingForImage(image_name, annotations_by_image, classMap)

        annotation = dict()
        blockX = 10
        blockY = 10
        blockCentre = np.array([blockX*32 +16, blockY*32+16], dtype=float)
        annotation['class'] = classMap['toClass'][ClassPrediction[blockY][blockX]]
        annotation['centre'] = blockCentre + np.array([BBPrediction[blockY][blockX][0], BBPrediction[blockY][blockX][1]])
        annotation['size'] = np.array([BBPrediction[blockY][blockX][2], BBPrediction[blockY][blockX][3]])

        print(f"{annotation['class']} at {annotation['centre']}")

        highlightAnnotation(preprocessed, annotation)

        print(f"new model predictions shape: {np.shape(predictions)}")
        print(f"image shape: {np.shape(image)}")
        print(f"preprocessed image shape: {np.shape(preprocessed)}")
        #images[image_name] = image
        #plt.close()
        fig = createFigure('image '+image_name)
        plt.imshow(image)
        # TODO: draw label text
        plt.show(block=False)
        fig = createFigure('preprocessed '+image_name)
        plt.imshow(preprocessed)
        plt.show()
        if(img_count % 1000 == 0):
            print(f"num images loaded: {img_count}")
            #print(f"image memory: {ProfilingUtils.getsize(images)}")
        
    print(f"all images loaded: {img_count}") """
