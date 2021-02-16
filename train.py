import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
import ProfilingUtils

syn_dataset_path = "../EdIntelligence-Neurolabs-Hackathon/common/train/syn/"
real_dataset_path = "../EdIntelligence-Neurolabs-Hackathon/challenge-2/train/real/"
image_sub_path = "images/"
challenge1_syn_csv = "annotations-detection.csv"
challenge2_syn_csv = "annotations-localization.csv"
challenge2_real_csv = "annotations.csv"

Image_Width = 1920
Image_Height = 1080

# Reads annotation file into a full annotations list, as well as by image and by class dicts
# annotation is dict with keys:
#   image: image name string
#   class: class name string
#   centre: np.array with x and y float pixel coordinates
#   size: np.array with x and y float pixel dimentions
# returns annotations_list, annotations_by_image, annotations_by_class
def read_annotation_file(path: str):
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
                annotation = {"image":image_name, "centre" : np.array([(xmax+xmin)/2, (ymax+ymin)/2]), "size" : np.array([(xmax-xmin), (ymax-ymin)]), "class" : class_name}
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
               

if __name__ == '__main__':
    challenge1_csv_path = os.path.join(syn_dataset_path, challenge1_syn_csv) 
    annotations_list, annotations_by_image, annotations_by_class = read_annotation_file(challenge1_csv_path)
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

    input_layer = keras.Input(shape=(Image_Height, Image_Width, 3))    
    mobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2(input_tensor=input_layer, include_top=False)

    img_count = 0
    images = dict()
    image_path = os.path.join(syn_dataset_path, image_sub_path) 
    for image_name in annotations_by_image.keys():
        image = plt.imread(os.path.join(image_path, image_name))
        img_count+=1
        preprocessed = keras.applications.mobilenet_v2.preprocess_input(image)
        batch_preprocessed = np.array([preprocessed])
        predictions = mobileNetV2.predict(batch_preprocessed)

        print(f"predictions shape: {np.shape(predictions)}")
        print(f"image shape: {np.shape(image)}")
        print(f"preprocessed image shape: {np.shape(preprocessed)}")
        #images[image_name] = image
        #plt.close()
        fig = createFigure('image '+image_name)
        plt.imshow(image)
        plt.show(block=False)
        fig = createFigure('preprocessed '+image_name)
        plt.imshow(preprocessed)
        plt.show()
        if(img_count % 1000 == 0):
            print(f"num images loaded: {img_count}")
            #print(f"image memory: {ProfilingUtils.getsize(images)}")
        
    print(f"all images loaded: {img_count}")
