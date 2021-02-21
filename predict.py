import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
import JsonUtils
import random

dataset_path = dict()
dataset_path['syn'] = "../EdIntelligence-Neurolabs-Hackathon/common/train/syn/"
dataset_path['real'] = "../EdIntelligence-Neurolabs-Hackathon/challenge-2/train/real/"
test_path = dict()
test_path['syn'] = "../EdIntelligence-Neurolabs-Hackathon/challenge-1/test/"
test_path['real'] = "../EdIntelligence-Neurolabs-Hackathon/challenge-2/test/"
image_sub_path = "images/"
challenge1_syn_csv = "annotations-detection.csv"
challenge2_syn_csv = "annotations-localization.csv"
challenge2_real_csv = "annotations.csv"

Image_Width = 1920
Image_Height = 1080
NumClasses = 330 + 1

BATCH_SIZE = 1

trainType = 'syn2syn'
#trainType = 'sr2real'

locOnly = False

testInReal = False
testInSyn = False
#testInReal = True
testInSyn = True


# adapted from https://stackoverflow.com/questions/2225564/get-a-filtered-list-of-files-in-a-directory
def listImagesInFolder(in_path):
    included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
    file_names = [fn for fn in os.listdir(in_path)
              if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def writeAnnotation(in_file, in_annotation):
    image_name = in_annotation['image']
    centre = in_annotation['centre']
    halthSize = in_annotation['size']/2
    BBMin = centre - halthSize
    BBMax = centre + halthSize
    xmin = int(max(0, min(Image_Width, BBMin[0])))
    ymin = int(max(0, min(Image_Width, BBMin[1])))
    xmax = int(max(0, min(Image_Width, BBMax[0])))
    ymax = int(max(0, min(Image_Width, BBMax[1])))
    class_name = in_annotation['class']
    if locOnly == True:
        class_name = 'object'
    
    in_file.write(f"{image_name}, {xmin}, {ymin}, {xmax}, {ymax}, {class_name} \n") 

def createAnnotationsForImage(in_imageName, in_BBPrediction, in_ObjectPrediction, in_ClassPrediction, in_classMap):
    #TODO: use input image dimentions to calculate prediction shapes

    annotations_list_for_img = list()

    for blockY in range(34):
        for blockX in range(60):
            blockCentre = np.array([blockX*32 +16, blockY*32+16], dtype=float)
            predictedTop2 = findTopNPredict(2, in_ClassPrediction[blockY][blockX])
            predictedClass = predictedTop2[0] 
            if predictedClass == 0  and in_ClassPrediction[blockY][blockX][0] < 0.5:
                predictedClass = predictedTop2[1] 
            if in_ObjectPrediction[blockY][blockX][1] > 0.5 and predictedClass > 0:
            #if predictedClass > 0:
                annotation = dict()
                annotation['image'] = in_imageName
                annotation['class'] = classMap['toClass'][predictedClass]
                annotation['centre'] = blockCentre + np.array([in_BBPrediction[blockY][blockX][0], in_BBPrediction[blockY][blockX][1]])
                annotation['size'] = np.array([in_BBPrediction[blockY][blockX][2], in_BBPrediction[blockY][blockX][3]])
                annotations_list_for_img.append(annotation)     
    return annotations_list_for_img

def writeAnnotationsForImage(in_file, in_imageName, in_BBPrediction, in_ObjectPrediction, in_ClassPrediction, in_classMap):
    annotations_list_for_img = createAnnotationsForImage(in_imageName, in_BBPrediction, in_ObjectPrediction, in_ClassPrediction, in_classMap)
    for annotation in annotations_list_for_img:
        writeAnnotation(in_file, annotation)

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
            if BBMin[0] < 0 or BBMin[1] < 0 or BBMax[0] >= Image_Width or BBMax[1] >= Image_Height:
                continue
            if x-BBMin[0] < 5 or BBMax[0]-x < 5 or y-BBMin[1] < 5 or BBMax[1]-y < 5 :
                in_image[y][x][0] = 1.0

def findTopNPredict(in_N, in_pred):
    n = in_N
    top = np.argsort(in_pred)[-n:]
    return top

def highlightAllInPredictions(in_image, in_BBPrediction, in_ObjectPrediction, in_ClassPrediction, in_classMap):
    #TODO: use input image dimentions to calculate prediction shapes

    for blockY in range(34):
        for blockX in range(60):
            blockCentre = np.array([blockX*32 +16, blockY*32+16], dtype=float)
            predictedTop2 = findTopNPredict(2, in_ClassPrediction[blockY][blockX])
            predictedClass = predictedTop2[0] 
            if predictedClass == 0  and in_ClassPrediction[blockY][blockX][0] < 0.5:
                predictedClass = predictedTop2[1] 
            if in_ObjectPrediction[blockY][blockX][1] > 0.5 and predictedClass > 0:
                annotation = dict()
                if locOnly == True:
                    annotation['class'] = 'object'
                else:
                    annotation['class'] = classMap['toClass'][predictedClass]
                annotation['centre'] = blockCentre + np.array([in_BBPrediction[blockY][blockX][0], in_BBPrediction[blockY][blockX][1]])
                annotation['size'] = np.array([in_BBPrediction[blockY][blockX][2], in_BBPrediction[blockY][blockX][3]])
                #print(f"{annotation['class']} at {annotation['centre']}")
                highlightAnnotation(in_image, annotation)
    return 

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

    input_layer = keras.Input(shape=(Image_Height, Image_Width, 3))    
    mobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, input_tensor=input_layer)
    mobileNetV2.trainable = False
    mobileNet_out = mobileNetV2(input_layer, training=False) 
    intermediate_net_1 = keras.layers.Conv2D(400, (1,1), padding='same', activation='relu', name='learning_input_reduction')(mobileNet_out)
    intermediate_net_2 = keras.layers.DepthwiseConv2D((7,7), padding='same', activation='relu', name='class_net')(intermediate_net_1)
    boundingBox_out = keras.layers.Conv2D(4, (7,7), padding='same', activation='linear', name='BB_out')(intermediate_net_2)
    object_out = keras.layers.Conv2D(2, (7,7), padding='same', activation='softmax', name='obj_out')(intermediate_net_2)
    class_out = keras.layers.Conv2D(NumClasses, (1,1), padding='same', activation='sigmoid', name='class_out')(intermediate_net_2)
    count_out = keras.layers.Conv2D(1, (9,9), activation='linear', padding='same', name='count_out')(object_out)

    #mobileNetV2.summary()
    
    BBmodel = keras.Model(input_layer, boundingBox_out)
    ObjModel = keras.Model(input_layer, object_out)
    ClassModel = keras.Model(input_layer, class_out)

    model = keras.Model(input_layer, [boundingBox_out, object_out, class_out, count_out])
 #   sparce_top5 = keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    
#    model.compile(optimizer='adam', 
#                    loss=['mse', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], 
#                    loss_weights=[0.05, 1.0, 5.0],
#                    metrics={'BB_out':keras.metrics.RootMeanSquaredError(), 'obj_out':'sparse_categorical_accuracy', 'class_out':['sparse_categorical_accuracy', sparce_top5]})

    #if(not os.path.exists('test_names.json')):
    #    print('no test dataset')
    #else:
    #    test_names_json = JsonUtils.readJsonFromFile('test_names.json')
    #    test_names = test_names_json['name_list']
    #    print('remembered to exclude test dataset')
    #image_path = os.path.join(dataset_path['syn'], image_sub_path) 

    if testInSyn == True:
        image_path = os.path.join(test_path['syn'], image_sub_path) 
        test_names = listImagesInFolder(image_path)

    if testInReal == True:
        image_path = os.path.join(test_path['real'], image_sub_path) 
        test_names = listImagesInFolder(image_path)


    if(os.path.exists('last_weights_'+trainType+'.h5')):
        model.load_weights('last_weights_'+trainType+'.h5')
        print('loaded last saved weights')
    else:
        print('ERROR: No saved weights')
        exit(-1)
    model.summary()
    
    img_count = 0
    images = dict()
    
    out_file_name = trainType
    if locOnly == True:
        out_file_name += '_Loc' 
    else:
        out_file_name += '_Class' 
    if testInReal:
        out_file_name += '_TestInReal'
    if testInSyn:
        out_file_name += '_TestInSyn'
    
    out_file_name += '.csv'
    fp = open(out_file_name, 'w')

    for image_name in test_names:
        image = plt.imread(os.path.join(image_path, image_name))
        img_count+=1
        
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(image)
        batch_preprocessed = np.array([preprocessed])
        
        
        BBPrediction = BBmodel.predict(batch_preprocessed)[0]
        ObjectPrediction = ObjModel.predict(batch_preprocessed)[0] 
        ClassPrediction = ClassModel.predict(batch_preprocessed)[0] 

        writeAnnotationsForImage(fp,image_name, BBPrediction, ObjectPrediction, ClassPrediction, classMap)

        if False:
            highlightAllInPredictions(preprocessed, BBPrediction, ObjectPrediction, ClassPrediction, classMap)
            fig = createFigure('image '+image_name)
            plt.imshow(image)
            # TODO: draw label text
            plt.show(block=False)
            fig = createFigure('preprocessed '+image_name)
            plt.imshow(preprocessed)
            plt.show()

        if(img_count % 100 == 0):
            print(f"num images loaded: {img_count}")
            #print(f"image memory: {ProfilingUtils.getsize(images)}")
    
    fp.close() 

    print(f"all images loaded: {img_count}")
