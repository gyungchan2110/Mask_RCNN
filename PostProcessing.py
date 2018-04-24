import numpy as np



#r['rois'], r['masks'], r['class_ids'], class_names, r['scores']

Classlist = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA"]

def Rule_RtUpperCB(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[1] < 512 and box[0] < 512:
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
 
    return Index 

def Rule_RtLowerCB(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[1] < 500 :
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
    
    return Index 

def Rule_LtLowerCB(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[1] > 500 :
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
    
    return Index 

def Rule_LAA(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[1] > 500 :
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
    
    return Index

def Rule_PC(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[1] > 500 :
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
    
    return Index 

def Rule_AK(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[1] > 500 and box[0] < 512:
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
    
    return Index

def Rule_Carina(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[0] < 512:
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
    
    return Index

def Rule_DAO(bboxes, scores, class_ids_indeces):
    
    Index = -1

    candidates = []
    for i in class_ids_indeces:
        box = bboxes[i]
        if box[1] > 400 :
            candidates.append(i) 
    if(len(candidates) == 0) :
        Index = -1
    elif(len(candidates) == 1):
        Index =  candidates[0]
    else:
        score = []
        for i in candidates:
            score.append(scores[i])
        score = np.asarray(score)
        Index = candidates[np.argmax(score)]
    
    return Index 

RulesDic = {"Aortic Knob" : "Rule_AK", 
            "Lt Lower CB" : "Rule_LtLowerCB", 
            "Pulmonary Conus" : "Rule_PC", 
            "Rt Lower CB" : "Rule_RtLowerCB", 
            "Rt Upper CB" : Rule_RtUpperCB, 
            "DAO" : "Rule_DAO", 
            "Carina" : "Rule_Carina", 
            "LAA" : "Rule_LAA", 
            }
            
def Rules(bboxes, scores, class_ids_indeces, classname):
    return RulesDic[classname](bboxes, scores, class_ids_indeces)


def GetMostProperMask(results, classes):
    
    
    # boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    # masks: [height, width, num_instances]
    # class_ids: [num_instances]

    #print(results)
    refeind_results = []
    for result in results : 

        MostProperIndexes = []

        scores = result['scores']
        masks =  result['masks']
        bboxes = result['rois']
        class_ids = result['class_ids']

        refined_scores = []
        refined_masks = []
        refined_bboxes = []
        refined_class_ids = []


        for i,classname in enumerate(classes) : 
            
            if(i == 0):
                continue  

            index = -1
            class_ids_indeces = []
            #class_ids = np.asarray(class_ids)
            for index, ids in enumerate(class_ids): 
                if ids == i :
                    class_ids_indeces.append(index)
           
            index = Rules(bboxes, scores, class_ids_indeces, classname)
            MostProperIndexes.append(index)



        for index in MostProperIndexes:
            
            if(index < 0):
                continue   

            refined_scores.append(scores[index])
            refined_masks.append(masks[:,:,index])        
            refined_bboxes.append(bboxes[index])
            refined_class_ids.append(class_ids[index])


        refined_scores = np.asarray(refined_scores)
        refined_masks = np.stack(refined_masks, -1)
        refined_bboxes = np.asarray(refined_bboxes)
        refined_class_ids = np.asarray(refined_class_ids)

        refeind_result = {'scores':refined_scores, 'masks':refined_masks, 'rois':refined_bboxes, 'class_ids':refined_class_ids}
        refeind_results.append(refeind_result)

    return refeind_results




    

