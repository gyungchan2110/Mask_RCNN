class_names = ["BG"]
class_names.append("Rt Lower CB")

Epochs = 40
ImageSize = 846 
Batch = 1 
DatasetFolder = "Generated_Data_20180125_103950"

FinalModelFile = ""





import socket 

GPU_ServerType = "Local_83"

hostName = socket.gethostname()
if(hostName == "deepserver"):
    GPU_ServerType = "Local_83"
elif (hostName[:3] == "csi"):
    GPU_ServerType = "Kakao"
elif (hostName == "user-All-Series"):
    GPU_ServerType = "Local_180"    
elif (hostName == "User-PC"):
    GPU_ServerType = "Local_PC" 


import os
import XRay_DataSet
import model as modellib
from datetime import datetime

BaseDataSetPath = ""
GPU_Num = ""

if(GPU_ServerType == "Local_83"):
    BaseDataSetPath = "/data/gyungchan_data/Segmentation"
    GPU_Num = "3"
elif (GPU_ServerType == "Kakao"):
    BaseDataSetPath = "/data/private/cardiomegaly"
    GPU_Num = "0"
elif (GPU_ServerType == "Local_180"):
    BaseDataSetPath = ""
    GPU_Num = "0"
elif (GPU_ServerType == "Local_PC"):
    BaseDataSetPath = ""
    GPU_Num = "0"

os.environ["CUDA_VISIBLE_DEVICES"]= GPU_Num
ROOT_DIR = os.getcwd()
LOG_DIR = ROOT_DIR + "/logs"
MODEL_DIR = ROOT_DIR + "/Models"
FinalModelPath = MODEL_DIR + "/" + FinalModelFile
##########################################################################################################################

config = XRay_DataSet.Xray_CardioMegalyConfig(len(class_names) - 1, ImageSize, Batch)
config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=LOG_DIR, config=config)
model.load_weights(FinalModelPath, by_name=True)



import cv2
import visualize
import numpy as np
import random

IMAGE_DIR = BaseDataSetPath + "/" + DatasetFolder + "/Imgs/test"
file_names = next(os.walk(IMAGE_DIR))[2]
image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
image = np.asarray(image)
print(image.shape)
#image = np.expand_dims(image,-1)
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
GetMostProperMask(r)
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])







def GetMostProperMask(results):
    scores = results['scores']
    masks =  results['masks']

    scores = np.asarray(scores)
    indeces = np.argmax(scores)

    for index in indeces : 
        mask = masks[:, :, index]
        plt.imshow(mask, cmap = "gray")
        plt.show()