from XRay_DataSet import Xray_CardioMegalyModelConfig 
import model as modellib
import os
import visualize
import numpy as np
import PostProcessing
import pandas
import csv
import pickle  
import gc
import visualize  

import cv2
from skimage.morphology import skeletonize
import math

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History


from keras import backend as K

def Train_DataSet(TaskID, datasetBase, datasetConfig, modelConfig, train_data, valid_data):
    
    ROOT_DIR = datasetBase + "/MaskRCNN_LOGS"
    if(not os.path.isdir(ROOT_DIR)):
        os.mkdir(ROOT_DIR)

    LOG_DIR = ROOT_DIR + "/logs"
    MODEL_DIR = ROOT_DIR + "/Models"
        
    LOG_DIR_FOLDER = LOG_DIR + "/Train_" + TaskID
    HistDumpFile = "History_" + TaskID + ".txt"
    HistDumpFileCSV = "History_" + TaskID + ".csv"



    FinalModelFile = "MaskRCNN_" 
    for cl in datasetConfig.CLASSES :
        FinalModelFile = FinalModelFile + cl + "_"
    FinalModelFile = FinalModelFile + TaskID + ".hdf5"

    train_data.prepare()
    valid_data.prepare()
    
    model = modellib.MaskRCNN(mode="training", config=modelConfig, log_dir = LOG_DIR_FOLDER, model_dir=LOG_DIR)
    if modelConfig.LEARNIING_MODE == "transfer":
        model.load_weights(modelConfig.PretainedModelPath, by_name=True)
        
    # model.train(dataset_train, dataset_val, 
    #             learning_rate=config.LEARNING_RATE, 
    #             epochs=10, 
    #             layers='heads')

    history = model.train(train_data, valid_data, 
                learning_rate=modelConfig.LEARNING_RATE / 10,
                epochs = modelConfig.EPOCHS, 
                layers="all")

    model.keras_model.save_weights(MODEL_DIR +"/"+ FinalModelFile)
    print(MODEL_DIR +"/"+ FinalModelFile)

    datasetConfig.SaveConfig(LOG_DIR_FOLDER)
    modelConfig.SaveConfig(LOG_DIR_FOLDER)

    
    #pandas.DataFrame(history.history).to_csv(LOG_DIR_FOLDER + "/" + HistDumpFileCSV)
    save_TrainHistoryLog("Success", TaskID, ROOT_DIR, datasetConfig, modelConfig)
    Result = True

    del model
    gc.collect()
    K.clear_session()
    return Result, FinalModelFile, history


def Test_Dataset(TaskID, datasetBase, datasetConfig, modelConfig, ModelFileName, testData, modelInLogDir = False, showImg = True):
    
    #ROOT_DIR = os.getcwd()
    ROOT_DIR = datasetBase + "/MaskRCNN_LOGS"
    LOG_DIR = ROOT_DIR + "/logs"

    if(modelInLogDir):
        MODEL_DIR = LOG_DIR
    else:
        MODEL_DIR = ROOT_DIR + "/Models"

    LOG_DIR_FOLDER = LOG_DIR + "/Test_" + TaskID
    
    Mask_DIR = LOG_DIR_FOLDER + "/Mask"
    OVERLAY_DIR = LOG_DIR_FOLDER + "/OverLay"

    if(not os.path.isdir(LOG_DIR_FOLDER)):
        os.mkdir(LOG_DIR_FOLDER)

    if(not os.path.isdir(Mask_DIR)):
        os.mkdir(Mask_DIR)
    if(not os.path.isdir(OVERLAY_DIR)):
        os.mkdir(OVERLAY_DIR)

    testData.prepare()
    testImages, testFileNames, testMasks = testData.getDataSet()


    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR + "/" + ModelFileName, config=modelConfig, log_dir = LOG_DIR)
    model.load_weights(MODEL_DIR + "/" + ModelFileName, by_name=True)


    Classes = datasetConfig.CLASSES    
    ##############################################################
    # Run detection
    results = model.detect_All(testImages, verbose=1)      
    results = PostProcessing.GetMostProperMask(results, Classes)
    
    #############################################################
    # Estimation 

    dices = []
    jaccards = []
    ious = []
    distances = []

    try : 

        logfile = LOG_DIR_FOLDER + "/Test_" + TaskID + "_Result.csv"
        f = open(logfile, 'a')
        f_writer = csv.writer(f)
        strline = []
        strline = ["filename", "Dice", "jaccard", "iou_mean", "dis_min", "dis_max", "dis_mean"]
        f_writer.writerow(strline)

        for i, result in enumerate(results):
            bbox = result['rois'][0]
            mask = result['masks'][:,:,0]
            class_ids = result['class_ids']

            truemask = np.asarray(testMasks[i], dtype = "uint8")
            ret, truemask = cv2.threshold(truemask, 127, 255, cv2.THRESH_BINARY)
            _, contours, _ = cv2.findContours(truemask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            true_bbox = cv2.boundingRect(contours[0])
            
            truemask = truemask // 255 
            mask = np.asarray(mask)
            #print(mask.shape, truemask.shape)
            dice = Dice(truemask, mask)
            jaccard = Jaccard_coef(truemask, mask)
            iou = estimate_IoU(true_bbox, (bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]))
            dis = ave_DisofSkeleton(truemask, mask)

            strline = []
            strline = [testFileNames[i], dice, jaccard, iou, dis[0], dis[1], dis[2]]
            f_writer.writerow(strline)
    

            dices.append(dice)
            jaccards.append(jaccard)
            ious.append(iou)
            distances.append(dis[2])

        f.close()
        dice = np.asarray(dices).mean()
        jaccard = np.asarray(jaccard).mean()
        iou = np.asarray(ious).mean()
        dis = np.asarray(distances).mean()

    except Exception as ex: 
        print(ex)
        if (f):
            f.close()
        os.rmdir(LOG_DIR_FOLDER) 

    save_TestHistoryLog(TaskID, ROOT_DIR, ModelFileName, dice, jaccard, iou, dis, datasetConfig)
    
    ############################################################
    # Display & Save 
    for i, result in enumerate(results) : 
        visualize.display_instances(testImages[i], result, Classes, testFileNames[i], auto_show = showImg, truemask = testMasks[i,:,:,:], logDir = LOG_DIR_FOLDER )
    
    print("Test Done ! ")




def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    #length = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    #print(y_pred.shape, y_true.shape)
    assert y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  

    #for i in range(length):

    true_sum= y_true[:,:].sum()
    pred_sum= y_pred[:,:].sum()
    intersection_sum = intersection[:,:].sum()

    #dices = np.zeros(length)

    #for i in range(length):
    dices = (2 * intersection_sum + 1.) / (true_sum + pred_sum + 1.)

    return dices

def estimate_IoU(true_Loc, pred_Loc):
    
#     if(not operator.eq(true_Loc[0], pred_Loc[0])):
#         return

    true_x_start = true_Loc[0]
    true_y_start = true_Loc[1]
    true_x_end = true_x_start + true_Loc[2]
    true_y_end = true_y_start + true_Loc[3]

    pred_x_start = pred_Loc[0]
    pred_y_start = pred_Loc[1]
    pred_x_end = pred_x_start + pred_Loc[2]
    pred_y_end = pred_y_start + pred_Loc[3]

    start_x = min(true_x_start, pred_x_start)
    start_y = min(true_y_start, pred_y_start)
    end_x = max(true_x_end, pred_x_end)
    end_y = max(true_y_end, pred_y_end)

    true_label = np.zeros((end_x - start_x, end_y - start_y))
    pred_label = np.zeros((end_x - start_x, end_y - start_y))

    true_label[(true_x_start -start_x) : (true_x_end - start_x), (true_y_start - start_y):(true_y_end - start_y) ] = 1
    pred_label[(pred_x_start -start_x) : (pred_x_end - start_x), (pred_y_start - start_y):(pred_y_end - start_y) ] = 1

    smooth = 1.
    y_true_f = true_label.flatten()
    y_pred_f = pred_label.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    jac = (intersection) / (union )
    return jac

def Jaccard_coef(y_true, y_pred):
    smooth = 1.

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert  y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  
    union = np.logical_or(y_true, y_pred) 

    
 
    union_sum= union[:,:].sum()
    intersection_sum = intersection[:,:].sum()


    #for i in range(length):
    jacards = (intersection_sum + smooth) / (union_sum + smooth)

    return jacards 

def ave_DisofSkeleton(y_true, y_pred):

    # y_true = np.asarray(y_true) // 255
    # y_pred = np.asarray(y_pred) // 255

    assert  y_true.shape == y_pred.shape
    y_true = skeletonize(y_true)
    y_pred = skeletonize(y_pred)
    
    true_pts = np.where(y_true==1)
    true_pts = np.asarray(true_pts)


    pred_pts = np.where(y_pred==1)
    pred_pts = np.asarray(pred_pts)


    true_len = len(y_true[y_true==1])
    pred_len = len(y_pred[y_pred==1])

    length = min(true_len, pred_len)

    distance = []

    dif = np.array([true_len - length, pred_len - length], dtype = "int")
    start = dif // 2

    for i in range(length): 
        
        pt1 = (true_pts[1][start[0] + i], true_pts[0][start[0] + i])
        pt2 = (pred_pts[1][start[1] + i], pred_pts[0][start[1] + i])
        dis = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])  +   (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        distance.append(dis)

    distance = np.asarray(distance)
    #print(length)
    if length > 0:
        return distance.min(), distance.max(), distance.mean()
    else : 
        return 0, 0, 0


def save_TrainHistoryLog(arg, TaskID, RootDir,  datasetConfig, modelConfig):
    
    HistoryFile = RootDir + "/Segmentation_TrainHistory.csv"
    
    f = open(HistoryFile, 'a')
    f_writer = csv.writer(f)
    
    strLines = [TaskID]
    for a in dir(datasetConfig):
        if not a.startswith("__") and not callable(getattr(datasetConfig, a)):
            strLines.append("{}".format(getattr(datasetConfig, a)))
    for a in dir(modelConfig):
        if not a.startswith("__") and not callable(getattr(modelConfig, a)):
            strLines.append("{}".format(getattr(modelConfig, a)))

    strLines.append(arg)
    f_writer.writerow(strLines)
    f.close()

def save_TestHistoryLog(TaskID, RootDir, ModelFileName, dice, jacard, iou, dis,  datasetConfig):
    HistoryFile = RootDir + "/Segmentation_TestHistory.csv"
    
    f = open(HistoryFile, 'a')
    f_writer = csv.writer(f)
    strLines = []
    #strLines = [TaskID, ModelFileName, datasetConfig.DATASET, str(dice), str(jacard), str(iou), str(dis)]
    strLines.append(datasetConfig.CLASSES)
    strLines.append([TaskID, ModelFileName, datasetConfig.DATASET, str(dice), str(jacard), str(iou), str(dis)])

    f_writer.writerow(strLines)
    f.close()