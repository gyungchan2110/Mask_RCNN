"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import random
import numpy as np
import cv2

from config import Config
import utils
import DataSets
import os


def get_Data_Set(datasetConfig, datasetBase, dataType):
    dataset = datasetConfig.DATASET 
    classes = datasetConfig.CLASSES

    ImagePath = datasetBase + "/" + dataset + "/Imgs"
    MasksPath = datasetBase + "/" + dataset + "/Masks"

    imgpath = ImagePath + "/" + dataType
    maskPaths = []
    for cl in classes:
        maskPaths.append(MasksPath + "/" + cl + "/" + dataType)

    data = Xray_CardioMegalyDataset()
    data.load_classes(imgpath, maskPaths, classes)

    return data


class Xray_CardioMegalyDataSetConfig():
    ConfigType = "DataSet"
    NAME = ""  
    NUM_CLASS = 0
    IMG_SHAPE = (1024,1024,1)
    CLASSES = []
    DATASET = ""

    def __init__(self, Name, Img_Shape, Classes, Dataset):
        self.NAME = Name
        self.IMG_SHAPE = Img_Shape
        self.CLASSES = Classes
        self.NUM_CLASS = len(self.CLASSES)
        self.DATASET = Dataset
    def SaveConfig(self, LogPath):
    
        if(not os.path.isdir(LogPath)): 
            os.mkdir(LogPath)

        if(not os.path.isdir(LogPath)): 
            print("Error : Log Path is Empty!")
            return 
        
        fileName = self.NAME + "_" +  self.ConfigType + ".txt"    
        filePath = LogPath + "/" + fileName
    
        f = open(filePath, 'a', encoding='utf-8')
        
        
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                f.writelines("{:30} {}\n".format(a, getattr(self, a)))
        f.close()

class Xray_CardioMegalyModelConfig(Config):
    NAME = "Cardiomegaly"  # Override in sub-classes

    ConfigType = "Model"
    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1
    EPOCHS = 50
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000 / IMAGES_PER_GPU

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 220

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1+ 1 # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = False  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True


    def __init__(self, classCount, ImageSize, Batch = 1):
        """Set values of computed attributes."""
        # Effective batch size
        self.NUM_CLASSES = 1 + classCount
        self.IMAGES_PER_GPU = Batch
        self.STEPS_PER_EPOCH = ImageSize / self.IMAGES_PER_GPU
        super(Xray_CardioMegalyModelConfig, self).__init__()


class Xray_CardioMegalyDataset(DataSets.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_classes(self, path, maskPath, classlabels):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        labels = classlabels
        for i, labelname in enumerate(labels):
            self.add_class("Cardiomegaly", i+1, labelname)

        if(path is None or not os.path.isdir(path)):
            return 

        maskFilePath = []
        width = height = 2048
        for file in os.listdir(path):
            index = len(self.image_info)
            filepath = path + "/" + file
            maskFilePath = []
            for i,filePath in enumerate(maskPath):
                maskFilePath.append(filePath + "/" + file)
            self.add_image("Cardiomegaly", image_id=index, path=filepath, maskPaths = maskFilePath,
                           width=width, height=height)
        



    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        fileaPath = info["path"]
        print(fileaPath)
        image = cv2.imread(fileaPath)
        image = np.array(image, dtype = "int16")
        if(len(image.shape) < 3):
            image = np.expand_dims(image, -1)

        filename = fileaPath.split("/")[-1]

        return image, filename

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        filePaths = info['maskPaths']
        
        masks = []
        classids = [0]
        
        #print(filePaths)
        N = len(filePaths)
        for i, filePath in enumerate(filePaths):           
            #print(filePath)
            if i == 0 and N >= 2:
                continue
            mask = cv2.imread(filePath)
            mask = np.asarray(mask, dtype = "int16")
            if(len(mask.shape)==3):
                mask = mask[:,:,0]
                
            masks.append(mask)
        
        
        image = np.stack(masks, axis=2)
        #print(image.shape)
        class_ids = np.array(classids, dtype=np.int32)
        return image, class_ids

    def showImginfo(self):
        print(len(self.image_info))
        print(self.image_info)

    def getDataSet(self):
        ids = self.image_ids

        Images = []
        Masks = []
        Filenames = []

        for pid in ids : 

            image, filename = self.load_image(pid)
            masks,ids = self.load_mask(pid) 
            Images.append(image)
            Masks.append(masks)
            Filenames.append(filename)

        Images = np.stack(Images, 0)
        Masks = np.stack(Masks, 0)
        print("Get Data Done")
        return Images, Filenames, Masks