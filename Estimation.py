import numpy as np
import os

def Dice_BBox(y_true, y_pred):
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

def Jaccard_coef_BBox(y_true, y_pred):
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


def Dice_Mask(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    length = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    #print(y_pred.shape, y_true.shape)
    assert len(y_true.shape) == 4 and y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  
    true_sum = np.zeros(length)
    pred_sum = np.zeros(length)
    intersection_sum = np.zeros(length)
    for i in range(length):
        true_sum[i]= y_true[i,:,:,:].sum()
        pred_sum[i]= y_pred[i,:,:,:].sum()
        intersection_sum[i] = intersection[i,:,:,:].sum()

    dices = np.zeros(length)

    #for i in range(length):
    dices[:] = (2 * intersection_sum[:] + 1.) / (true_sum[:] + pred_sum[:] + 1.)

    return dices
    
def Jaccard_coef_Mask(y_true, y_pred):
    smooth = 1.
    length = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true.shape) == 4 and y_true.shape == y_pred.shape

    intersection = np.logical_and(y_true, y_pred)  
    union = np.logical_or(y_true, y_pred) 
    union_sum = np.zeros(length)
    intersection_sum = np.zeros(length)
    for i in range(length):
        union_sum[i]= union[i,:,:,:].sum()
        intersection_sum[i] = intersection[i,:,:,:].sum()

    jacards = np.zeros(length)

    #for i in range(length):
    jacards[:] = (intersection_sum[:] + smooth) / (union_sum[:] + smooth)

    return jacards 