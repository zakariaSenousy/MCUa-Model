from src import *
from src import datasets1, datasets2
import numpy as np

from src import models_N1, networks_N1
from src import models_N2, networks_N2
from src import models_N3, networks_N3
from src import models_P4, networks_P4
from src import models_P5, networks_P5
from src import models_P6, networks_P6

from src import models1_N1, networks1_N1
from src import models1_N2, networks1_N2
from src import models1_P4, networks1_P4
from src import models1_P5, networks1_P5
from src import models1_P6, networks1_P6
from src import models1_P8, networks1_P8

from src import models2_N1, networks2_N1
from src import models2_N2, networks2_N2
from src import models2_N3, networks2_N3
from src import models2_P4, networks2_P4
from src import models2_P5, networks2_P5
from src import models2_P6, networks2_P6

#------------------------

import torch
from torch.distributions import Categorical
import re


args = ModelOptions().parse()


LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
LabelAbbrev = ['n', 'b', 'is', 'iv']



def Single_Feature_extractor(p, mode='test'):
    results = []
    for i in range(len(p)):     
        model_pred = np.array(p[i][0])
        highest_index = 3 - np.argmax(model_pred[::-1])
        label = LABELS[highest_index]
        image_name = p[i][1]
        results.append([label,image_name])
    
    print('Feature Extractor model:')
    print('--------------------------------------')
    for i in results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(results)):
            if results[i][0] == LABELS[0] and LabelAbbrev[0] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[1] and LabelAbbrev[1] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[2] and LabelAbbrev[2] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[3] and LabelAbbrev[3] in results[i][1]:
                correct+=1
        val_acc = (correct / len(results)) *100
        print('Validation Accuracy = ', val_acc,'%')
        
    
        
def Single_ContextAware(p, mode = 'test'):
    results = []
    for i in range(len(p)):     
        model_pred = np.array(p[i][0])
        std = np.array(p[i][1])
        logp = np.log2(model_pred)
        entropy = np.sum(-model_pred * logp)
        highest_index = np.argmax(model_pred)
        label = LABELS[highest_index]
        image_name = p[i][3]
        results.append([label, image_name, model_pred, std])
    
    print('Context Aware model:')
    print('--------------------------------------')
    for i in results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(results)):
            if results[i][0] == LABELS[0] and LabelAbbrev[0] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[1] and LabelAbbrev[1] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[2] and LabelAbbrev[2] in results[i][1]:
                correct+=1
            elif results[i][0] == LABELS[3] and LabelAbbrev[3] in results[i][1]:
                correct+=1
        val_acc = (correct / len(results)) *100
        print('Validation Accuracy = ', val_acc,'%')
    
        
def Feature_extractor_Ensemble (p, p1, p2, mode = 'test'):
    ens_results = []
    for i in range(len(p)):     
        model_pred = np.array(p[i][0])
        model_pred1 = np.array(p1[i][0])
        model_pred2 = np.array(p2[i][0])
        
        final = np.divide(model_pred + model_pred1 + model_pred2, 3)
        highest_index = 3 - np.argmax(final[::-1])
        label = LABELS[highest_index]
        image_name = p[i][1]
        ens_results.append([label,image_name])
    
    print('Ensemble of Feature Extractor models:')
    print('--------------------------------------')
    for i in ens_results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(ens_results)):
            if ens_results[i][0] == LABELS[0] and LabelAbbrev[0] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[1] and LabelAbbrev[1] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[2] and LabelAbbrev[2] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[3] and LabelAbbrev[3] in ens_results[i][1]:
                correct+=1
        val_acc = (correct / len(ens_results)) *100
        print('Validation Accuracy = ', val_acc,'%')
    
     



def ContextAware_Ensemble (p, p1, p2, p3, p4, p5, p6, p7, p8,
                           p9, p10, p11, p12, p13, p14, p15, p16, p17,
                           threshold = 0, mode = 'test'):
    ens_results = []
    excluded_imgs = []    
    for i in range(len(p)):
        chosen_models = []
        #0
        uncert = np.mean(p[i][1])
        if uncert < threshold:
            chosen_models.append(p[i][0])
        
        #1
        uncert = np.mean(p1[i][1])
        if uncert < threshold:
            chosen_models.append(p1[i][0])
            
        #2
        uncert = np.mean(p2[i][1])
        if uncert < threshold:
            chosen_models.append(p2[i][0])
        #3
        uncert = np.mean(p3[i][1])
        if uncert < threshold:
            chosen_models.append(p3[i][0])
        
        #4
        uncert = np.mean(p4[i][1])
        if uncert < threshold:
            chosen_models.append(p4[i][0])
            
        #5
        uncert = np.mean(p5[i][1])
        if uncert < threshold:
            chosen_models.append(p5[i][0])
            
        #6
        uncert = np.mean(p6[i][1])
        if uncert < threshold:
            chosen_models.append(p6[i][0])
            
        #7
        uncert = np.mean(p7[i][1])
        if uncert < threshold:
            chosen_models.append(p7[i][0])
            
        #8
        uncert = np.mean(p8[i][1])
        if uncert < threshold:
            chosen_models.append(p8[i][0])
            
        #9
        uncert = np.mean(p9[i][1])
        if uncert < threshold:
            chosen_models.append(p9[i][0])
            
        #10
        uncert = np.mean(p10[i][1])
        if uncert < threshold:
            chosen_models.append(p10[i][0])
        
        #11
        uncert = np.mean(p11[i][1])
        if uncert < threshold:
            chosen_models.append(p11[i][0])
        
        #12
        uncert = np.mean(p12[i][1])
        if uncert < threshold:
            chosen_models.append(p12[i][0])
            
        #13
        uncert = np.mean(p13[i][1])
        if uncert < threshold:
            chosen_models.append(p13[i][0])
            
        #14
        uncert = np.mean(p14[i][1])
        if uncert < threshold:
            chosen_models.append(p14[i][0])
        
        #15
        uncert = np.mean(p15[i][1])
        if uncert < threshold:
            chosen_models.append(p15[i][0])
            
        #16
        uncert = np.mean(p16[i][1])
        if uncert < threshold:
            chosen_models.append(p16[i][0])
            
        #17
        uncert = np.mean(p17[i][1])
        if uncert < threshold:
            chosen_models.append(p17[i][0])
 
        #----------------------------------------
        if len(chosen_models) != 0:
            final = np.sum(chosen_models, axis=0)        
            highest_index = np.argmax(final)
            label = LABELS[highest_index]
            image_name = p[i][2]        
            ens_results.append([label, image_name, final, len(chosen_models)])
        
        if len(chosen_models) == 0:
            exc_image_name = p[i][2]
            excluded_imgs.append([exc_image_name, len(chosen_models)])
            
    print('Ensemble of Context-aware models:')
    print('--------------------------------------')
    for i in ens_results:
        print(i)
    
    if mode == 'valid':
        correct = 0
        for i in range(len(ens_results)):
            if ens_results[i][0] == LABELS[0] and LabelAbbrev[0] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[1] and LabelAbbrev[1] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[2] and LabelAbbrev[2] in ens_results[i][1]:
                correct+=1
            elif ens_results[i][0] == LABELS[3] and LabelAbbrev[3] in ens_results[i][1]:
                correct+=1
        print ('Included Images: ', len(ens_results))
        exc = 80 - len(ens_results)
        print('Excluded Images: ', exc)
        val_acc = (correct / len(ens_results)) *100
        print('Validation Accuracy = ', val_acc,'%')
    
    print('EXCLUDED IMGS')
    print('-------------------------')
    for j in excluded_imgs:
        print(j)
    print('-------------------------')


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


pw_network_N1 = networks_N1.PatchWiseNetwork(args.channels)
iw_network_N1 = networks_N1.ImageWiseNetwork(args.channels)

"""pw_network_N2 = networks_N2.PatchWiseNetwork(args.channels)
iw_network_N2 = networks_N2.ImageWiseNetwork(args.channels)

pw_network_N3 = networks_N3.PatchWiseNetwork(args.channels)
iw_network_N3 = networks_N3.ImageWiseNetwork(args.channels)

pw_network_P4 = networks_P4.PatchWiseNetwork(args.channels)
iw_network_P4 = networks_P4.ImageWiseNetwork(args.channels)
  
pw_network_P5 = networks_P5.PatchWiseNetwork(args.channels)
iw_network_P5 = networks_P5.ImageWiseNetwork(args.channels)

pw_network_P6 = networks_P6.PatchWiseNetwork(args.channels)
iw_network_P6 = networks_P6.ImageWiseNetwork(args.channels)"""

#-----------------------------------------------------------------#

pw_network1_N1 = networks1_N1.PatchWiseNetwork1(args.channels)
iw_network1_N1 = networks1_N1.ImageWiseNetwork1(args.channels)

"""pw_network1_N2 = networks1_N2.PatchWiseNetwork1(args.channels)
iw_network1_N2 = networks1_N2.ImageWiseNetwork1(args.channels)

pw_network1_P4 = networks1_P4.PatchWiseNetwork1(args.channels)
iw_network1_P4 = networks1_P4.ImageWiseNetwork1(args.channels)

pw_network1_P5 = networks1_P5.PatchWiseNetwork1(args.channels)
iw_network1_P5 = networks1_P5.ImageWiseNetwork1(args.channels)

pw_network1_P6 = networks1_P6.PatchWiseNetwork1(args.channels)
iw_network1_P6 = networks1_P6.ImageWiseNetwork1(args.channels)

pw_network1_P8 = networks1_P8.PatchWiseNetwork1(args.channels)
iw_network1_P8 = networks1_P8.ImageWiseNetwork1(args.channels)"""

#----------------------------------------------------------------

pw_network2_N1 = networks2_N1.PatchWiseNetwork2(args.channels)
iw_network2_N1 = networks2_N1.ImageWiseNetwork2(args.channels)

"""pw_network2_N2 = networks2_N2.PatchWiseNetwork2(args.channels)
iw_network2_N2 = networks2_N2.ImageWiseNetwork2(args.channels)

pw_network2_N3 = networks2_N3.PatchWiseNetwork2(args.channels)
iw_network2_N3 = networks2_N3.ImageWiseNetwork2(args.channels)

pw_network2_P4 = networks2_P4.PatchWiseNetwork2(args.channels)
iw_network2_P4 = networks2_P4.ImageWiseNetwork2(args.channels)

pw_network2_P5 = networks2_P5.PatchWiseNetwork2(args.channels)
iw_network2_P5 = networks2_P5.ImageWiseNetwork2(args.channels)

pw_network2_P6 = networks2_P6.PatchWiseNetwork2(args.channels)
iw_network2_P6 = networks2_P6.ImageWiseNetwork2(args.channels)"""

#-----------------------------



if args.testset_path is '':
    import tkinter.filedialog as fdialog

    args.testset_path = fdialog.askopenfilename(initialdir=r"./dataset/test", title="choose your file", filetypes=(("tiff files", "*.tif"), ("all files", "*.*")))

if args.network == '1':
    pw_model_N1 = models_N1.PatchWiseModel(args, pw_network_N1)
    pred = pw_model_N1.test(args.testset_path)
  
    pw_model1_N1 = models1_N1.PatchWiseModel1(args, pw_network1_N1)
    pred1 = pw_model1_N1.test(args.testset_path)
    
    pw_model2_N1 = models2_N1.PatchWiseModel2(args, pw_network2_N1)
    pred2 = pw_model2_N1.test(args.testset_path)

    
    Feature_extractor_Ensemble(pred, pred1, pred2, mode = 'valid')
    #Single_Feature_extractor(pred, mode = 'valid')
    
else:
    
    im_model_N1 = models_N1.ImageWiseModel(args, iw_network_N1, pw_network_N1)
    context_N1 = im_model_N1.test(args.testset_path, ensemble= False)
    
    im_model_N2 = models_N2.ImageWiseModel(args, iw_network_N2, pw_network_N2)
    context_N2 = im_model_N2.test(args.testset_path, ensemble= False)
        
    im_model_N3 = models_N3.ImageWiseModel(args, iw_network_N3, pw_network_N3)
    context_N3 = im_model_N3.test(args.testset_path, ensemble= False)
    
    im_model_P4 = models_P4.ImageWiseModel(args, iw_network_P4, pw_network_P4)
    context_P4 = im_model_P4.test(args.testset_path, ensemble= False)
    
    im_model_P5 = models_P5.ImageWiseModel(args, iw_network_P5, pw_network_P5)
    context_P5 = im_model_P5.test(args.testset_path, ensemble= False)
    
    im_model_P6 = models_P6.ImageWiseModel(args, iw_network_P6, pw_network_P6)
    context_P6 = im_model_P6.test(args.testset_path, ensemble= False)
    
    #-----------------------
    
    im_model1_N1 = models1_N1.ImageWiseModel1(args, iw_network1_N1, pw_network1_N1)
    context1_N1 = im_model1_N1.test(args.testset_path, ensemble= False)
    
    im_model1_N2 = models1_N2.ImageWiseModel1(args, iw_network1_N2, pw_network1_N2)
    context1_N2 = im_model1_N2.test(args.testset_path, ensemble= False)
    
    im_model1_P4 = models1_P4.ImageWiseModel1(args, iw_network1_P4, pw_network1_P4)
    context1_P4 = im_model1_P4.test(args.testset_path, ensemble= False)
    
    im_model1_P5 = models1_P5.ImageWiseModel1(args, iw_network1_P5, pw_network1_P5)
    context1_P5 = im_model1_P5.test(args.testset_path, ensemble= False)
    
    im_model1_P6 = models1_P6.ImageWiseModel1(args, iw_network1_P6, pw_network1_P6)
    context1_P6 = im_model1_P6.test(args.testset_path, ensemble= False)
    
    im_model1_P8 = models1_P8.ImageWiseModel1(args, iw_network1_P8, pw_network1_P8)
    context1_P8 = im_model1_P8.test(args.testset_path, ensemble= False)
    
    #-----------------------
    
    im_model2_N1 = models2_N1.ImageWiseModel2(args, iw_network2_N1, pw_network2_N1)
    context2_N1 = im_model2_N1.test(args.testset_path, ensemble=False)

    im_model2_N2 = models2_N2.ImageWiseModel2(args, iw_network2_N2, pw_network2_N2)
    context2_N2 = im_model2_N2.test(args.testset_path, ensemble=False)
    
    im_model2_N3 = models2_N3.ImageWiseModel2(args, iw_network2_N3, pw_network2_N3)
    context2_N3 = im_model2_N3.test(args.testset_path, ensemble=False)
    
    im_model2_P4 = models2_P4.ImageWiseModel2(args, iw_network2_P4, pw_network2_P4)
    context2_P4 = im_model2_P4.test(args.testset_path, ensemble=False)
    
    im_model2_P5 = models2_P5.ImageWiseModel2(args, iw_network2_P5, pw_network2_P5)
    context2_P5 = im_model2_P5.test(args.testset_path, ensemble=False)
    
    im_model2_P6 = models2_P6.ImageWiseModel2(args, iw_network2_P6, pw_network2_P6)
    context2_P6 = im_model2_P6.test(args.testset_path, ensemble=False)
    
    #--------------------------
    
    #import numpy as np
    t = np.array([0.001, 0.002, 0.003,
                  0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                  0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
    
    #t = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
    
    for val in t:
        print('THRESHOLD = ', val)
        ContextAware_Ensemble(context_N1, context_N2, context_N3, context_P4, context_P5, 
                          context_P6, context1_N1, context1_N2, context1_P4,
                          context1_P5, context1_P6,  context1_P8, context2_N1,
                          context2_N2, context2_N3, context2_P4, context2_P5,
                          context2_P6, threshold = val, mode = 'valid')

    
    
                          
    #Single_ContextAware(context_N1, mode = 'valid')