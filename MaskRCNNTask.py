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
    
    model = modellib.MaskRCNN(mode="training", config=modelConfig, model_dir=LOG_DIR)
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

    #draw_history_graph(history, LOG_DIR_FOLDER + "/" + HistImgFile)
    with open(LOG_DIR_FOLDER + "/" + HistDumpFile, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    pandas.DataFrame(history.history).to_csv(LOG_DIR_FOLDER + "/" + HistDumpFileCSV)
    save_TrainHistoryLog("Success", TaskID, ROOT_DIR, datasetConfig, modelConfig)
    Result = True

    del model
    gc.collect()
    K.clear_session()
    return Result, FinalModelFile


def Test_Dataset(TaskID, datasetBase, datasetConfig, modelConfig, ModelFileName, testData, modelInLogDir = False):
    
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

    testImages, testFileNames, testMasks = testData.getDataSet()


    model = modellib.MaskRCNN(mode="inference", model_dir=LOG_DIR, config=modelConfig)
    model.load_weights(MODEL_DIR + "/" + ModelFileName, by_name=True)


    Classes = datasetConfig.Classes    
    ##############################################################
    # Run detection
    results = model.detect_All(testImages, verbose=1)      
    results = PostProcessing.GetMostProperMask(results, Classes)
    
    #############################################################
    # Estimation 


    
    
    ############################################################
    # Display & Save 
    for i, result in enumerate(results) : 
        visualize.display_instances(testImages[i], result, Classes, testFileNames[i], truemask = testMasks, logDir = LOG_DIR_FOLDER )




def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    length = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    #print(y_pred.shape, y_true.shape)
    assert y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  
    true_sum = np.zeros(length)
    pred_sum = np.zeros(length)
    intersection_sum = np.zeros(length)
    for i in range(length):
        true_sum[i]= y_true[i,:,:].sum()
        pred_sum[i]= y_pred[i,:,:].sum()
        intersection_sum[i] = intersection[i,:,:].sum()

    dices = np.zeros(length)

    #for i in range(length):
    dices[:] = (2 * intersection_sum[:] + 1.) / (true_sum[:] + pred_sum[:] + 1.)

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
    length = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert  y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  
    union = np.logical_or(y_true, y_pred) 
    union_sum = np.zeros(length)
    intersection_sum = np.zeros(length)
    for i in range(length):
        union_sum[i]= union[:,:].sum()
        intersection_sum[i] = intersection[i,:,:].sum()

    jacards = np.zeros(length)

    #for i in range(length):
    jacards[:] = (intersection_sum[:] + smooth) / (union_sum[:] + smooth)

    return jacards 


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