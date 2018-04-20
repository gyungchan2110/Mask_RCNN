# In[]
class_names = []
class_names.append("Carina")

Epochs = 20
ImageSize = 846 
Batch = 2 
DatasetFolder = "Generated_Data_20180125_103950_Expand_20pixel"

PretainedModelFIle = "MaskRCNN_Carina_20180326_150620.hdf5"
LEARNIING_MODE = "scratch"
#LEARNIING_MODE = "transfer"

if LEARNIING_MODE == "transfer" : 
    assert(PretainedModelFIle is not None)

#######################################################################

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

currentTime = datetime.now()
strTime = "%04d%02d%02d_%02d%02d%02d.hdf5" %(currentTime.year, currentTime.month, currentTime.day,currentTime.hour, currentTime.minute, currentTime.second)
FinalModelFile = "MaskRCNN_"
for cl in class_names:
    FinalModelFile = FinalModelFile + cl + "_"
FinalModelFile = FinalModelFile + strTime

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
if LEARNIING_MODE == "transfer" : 
    PretainedModelPath = MODEL_DIR +"/" + PretainedModelFIle
FinalModelPath = MODEL_DIR + "/" + FinalModelFile
##########################################################################################################################

config = XRay_DataSet.Xray_CardioMegalyConfig(len(class_names), ImageSize, Batch)
config.IMAGE_PADDING = True
############################################################


ImagePath = BaseDataSetPath + "/" + DatasetFolder + "/Imgs"
MasksPath = BaseDataSetPath + "/" + DatasetFolder + "/Masks"

path = ImagePath + "/train"
maskPaths = []
for cl in class_names:
    maskPaths.append(MasksPath + "/Mask_" + cl + "/train")
    
dataset_train = XRay_DataSet.Xray_CardioMegalyDataset()
dataset_train.load_classes(path,maskPaths, class_names)
dataset_train.prepare()


path = ImagePath + "/validation"
maskPaths = []
for cl in class_names:
    maskPaths.append(MasksPath + "/Mask_" + cl + "/validation")
    
dataset_val = XRay_DataSet.Xray_CardioMegalyDataset()
dataset_val.load_classes(path,maskPaths, class_names)
dataset_val.prepare()

###################################################################################
model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOG_DIR)
if LEARNIING_MODE == "transfer":
    model.load_weights(PretainedModelPath, by_name=True)
    
# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE, 
#             epochs=10, 
#             layers='heads')

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=Epochs, 
            layers="all")

#######################################################################################
model.keras_model.save_weights(MODEL_DIR +"/"+ FinalModelFile)
print(MODEL_DIR +"/"+ FinalModelFile)