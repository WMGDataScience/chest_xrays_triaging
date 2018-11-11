# -*- coding: utf-8 -*-
from __future__ import print_function,division

import argparse
from pymongo import MongoClient
import torchfile
from sklearn.metrics import confusion_matrix,auc
import numpy as np
import re
import pickle
import csv
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt

def definePriorityMapping(all_classes):
    
    priority_list = {}
    priority_list["normal"],priority_list["non_urgent_finding"],priority_list["urgent"],priority_list["critical"] = 4,3,2,1
    per_priority_mapping = {}
    for cl in all_classes:
        if cl == 'intraabdominal_pathology' or cl == "pneumomediastinum" or cl == "pneumothorax" or cl == "subcutaneous_emphysema" \
           or cl == "mediastinum_widened" or cl == "pneumoperitoneum" or cl == 'Pneumothorax':
            per_priority_mapping[cl] = "critical"
        elif cl == 'bone_abnormality' or cl=='collapse' or cl == "consolidation" or cl == "pleural_abnormality" or cl == "parenchymal_lesion"\
            or cl == "right_upper_lobe_collapse" or cl == "cavitating_lung_lesion" or cl == "rib_fracture" or cl == "interstitial_shadowing"\
            or cl == "left_lower_lobe_collapse" or cl == "clavicle_fracture" or cl == "paratracheal_hilar_enlargement" \
            or cl == "dilated_bowel" or cl == "ground_glass_opacification" or cl == "left_upper_lobe_collapse" or cl == "rib_lesion"\
            or cl == "right_lower_lobe_collapse" or cl == "mediastinum_displaced" or cl == "right_middle_lobe_collapse" \
            or cl == "widened_paratracheal_stripe" or cl == "enlarged_hilum" or cl == "pleural_thickening" or cl == "pleural_lesion"\
            or cl == "airspace_opacifation" or cl == "Mass" or cl == "pleural_effusion" or cl == 'Effusion' or cl == 'Nodule'\
            or cl == 'Fibrosis' or cl == 'Pneumonia' or cl == 'Consolidation' or cl == 'Pleural_Thickening' or cl == 'Edema'\
            or cl == 'Infiltration':
            per_priority_mapping[cl] = "urgent"
        elif cl == 'hiatus_hernia' or cl == 'abnormal_other' or cl == "object" or cl == "cardiomegaly" or cl == "emphysema" \
               or cl == "scoliosis" or cl == "hernia" or cl == "hyperexpanded_lungs" or cl == "unfolded_aorta" \
               or cl == "dextrocardia" or cl == "bulla" or cl == "aortic_calcification" or cl == "hemidiaphragm_elevated"\
               or cl == "bronchial_wall_thickening" or cl == 'Atelectasis' or cl == 'Cardiomegaly' or cl == 'Hernia'\
               or cl == 'Emphysema' or cl == "atelectasis" :
            per_priority_mapping[cl] = "non_urgent_finding"  
        elif cl == "normal":
            per_priority_mapping[cl] = "normal"
            
    return priority_list,per_priority_mapping                  


# Define our softmax function
def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex
 

def getCM_Measures(cm):
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[0][1]    
    FP = cm[1][0]
    
    Sensitivity_Recall, Specificity, Precision, F1_Score, avg_accuracy = 0,0,0,0,0

    if (TP+FN) != 0:   
        Sensitivity_Recall = TP/float(TP+FN)
    if (TN+FP) != 0:
        Specificity = TN/float(TN+FP)
    if (TP+FP) != 0:
        Precision = TP/float(TP+FP)
    if (Precision + Sensitivity_Recall) != 0:
        F1_Score = 2*((Precision * Sensitivity_Recall)/float(Precision + Sensitivity_Recall))
    avg_accuracy = 1/2.0*(Sensitivity_Recall+Specificity)  

    
    return Sensitivity_Recall, Specificity, Precision, F1_Score, avg_accuracy
    
def printConfMatrix(cm,cl):
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[0][1]    
    FP = cm[1][0]    
    Sensitivity_Recall, Specificity, Precision, F1_Score, avg_accuracy =  getCM_Measures(cm)
    print(cl,TP,FN,FP,TN)
    
    return Sensitivity_Recall, Specificity, Precision, F1_Score, avg_accuracy

def fromV10toV12(classes):
    result = []

    collapse = ['left_lower_lobe_collapse','left_upper_lobe_collapse','right_middle_lobe_collapse',
                'right_lower_lobe_collapse','right_upper_lobe_collapse']                
     
    for cl in classes:
        if cl in collapse:
            if not 'collapse' in result:
                result.append('collapse')
        else:
            result.append(cl)
   
    return result

def fromV10toV11(classes):
    result = []
    V10_V11_map = {
                    'abnormal_other':['dextrocardia','unfolded_aorta','hemidiaphragm_elevated', 
                                      'atelectasis','bronchial_wall_thickening','bulla','emphysema', 
                                      'hyperexpanded_lungs','mediastinum_displaced','mediastinum_widened', 'scoliosis'],
                    'collapse':['left_lower_lobe_collapse','left_upper_lobe_collapse','right_middle_lobe_collapse',
                                'right_lower_lobe_collapse','right_upper_lobe_collapse'],                  
                    'intraabdominal_pathology':['dilated_bowel','pneumoperitoneum'],
                    'bone_abnormality' : ['rib_lesion' ,'rib_fracture' ,'clavicle_fracture'],
                    'airspace_opacifation' : ['ground_glass_opacification','consolidation'],
                    'cardiomegaly':['cardiomegaly'],
                    'parenchymal_lesion':['parenchymal_lesion','cavitating_lung_lesion'],
                    'interstitial_shadowing':['interstitial_shadowing'],
                    'paratracheal_hilar_enlargement':['paratracheal_hilar_enlargement'],
                    'pneumomediastinum':['pneumomediastinum'],
                    'object':['object'],
                    'normal':['normal'],
                    'pleural_abnormality':['pleural_abnormality','pleural_effusion'],
                    'pneumothorax':['pneumothorax'],
                    'subcutaneous_emphysema':['subcutaneous_emphysema'],
                    'hiatus_hernia':['hernia']
                   }       
    for cl in classes:
        for sup_cl in V10_V11_map.keys():
            if cl in V10_V11_map[sup_cl] and not sup_cl in result:
                result.append(sup_cl)
                break

    if len(result) == 0 and len(classes) > 0:
        result.append('normal')
    
    return result

def getPosition(d):
    if not "SeriesDescription" in d.keys():
        if "position" in d.keys():
            return d["position"]
        else:
            return "unknown" 
    elif re.search("(^| )[aA][\. ]?[pP]( |\.|$)",str(d["SeriesDescription"])) != None:
        return "AP"
    elif re.search("(^| )[pP][\. ]?[aA]( |\.|$)",str(d["SeriesDescription"])) != None: 
        return "PA"
    else:
        return "unknown"   


class SearchResults(object):
    def __init__(self, el=None):
        self.result = []
        if el != None:
            self.result.append(el)
    
    def count(self):
        return len(self.result)

    def __getitem__(self, key):
        return self.result[key]
    
class DataInfos(object):
    """Contains the infos on our data

    Attributes:
        collection: A dictionary with all our data.

    """

    def __init__(self, csv_file_path):
        self.collection = {}
        with open(csv_file_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            reader.next()
            i = 1
            for row in reader:
                el = {}
                el['Annotations'] = {}
                el['Annotations']['Report'] = {}
                el['Annotations']['Report']["labels"] = row[1].split("|")
                if 'No Finding' in el['Annotations']['Report']["labels"]:
                    el['Annotations']['Report']["labels"].remove('No Finding')
                    el['Annotations']['Report']["labels"].append('normal')                
                el["position"] = row[6]
                self.collection[str(i)] = el
                i = i + 1
                
                
    def find(self, input_obj):
        to_find_id =  input_obj['_id']
        if to_find_id in self.collection.keys():
            return SearchResults(self.collection[to_find_id])
        else:
            print(type(self.collection))
            print(to_find_id,self.collection[to_find_id])
            return SearchResults()

def fromLabelsToPriorityClass(labels,args):
    priority_class = 'normal'
    for cl in labels:
       if args.priority_list[priority_class] > args.priority_list[args.per_priority_mapping[cl]]:
           priority_class = args.per_priority_mapping[cl]      
    return priority_class

def updateAveragePreds(y_pred,y_pred_pc,args,current_id,AP_PA_flag):

    for th in args.ALL_THRESHOLDS + ['best_ths']:    
        predicted_priority_class = 'normal'
        
        if args.model_type == 'ordinal_regression_priority_classifier':
           
           if args.ensamble_type ==  'avg_probs':
               for cl in ["non_urgent_finding","urgent","critical"]:                     
                   pred = softmax(args.predictions_vec[0][current_id][cl])[1]
                   for i in range(1,len(args.predictions_vec)):
                       pred = pred + softmax(args.predictions_vec[i][current_id][cl])[1]
                   pred = pred / float(len(args.predictions_vec))
                   to_use_th = th
                   if th == 'best_ths':
                       if cl in args.thresholds.keys():
                           to_use_th = args.thresholds[cl]
                       else:
                           to_use_th = args.threshold 
                   if pred > to_use_th:
                       predicted_priority_class = cl
                   else:
                       break
           elif args.ensamble_type ==  'avg_classification_res':
                ensamble_predictions = {}
                for i in range(len(args.predictions_vec)): 
                    ensamble_predictions[i] = 'normal'
                    for cl in ["non_urgent_finding","urgent","critical"]: 
                        to_use_th = th
                        if th == 'best_ths':
                             to_use_th = args.thresholds_vec[i][cl]                        
                        if softmax(args.predictions_vec[i][current_id][cl])[1] > to_use_th:
                             ensamble_predictions[i] = cl   
                    else:
                        break 
    
                for cl in ["non_urgent_finding","urgent","critical"]: 
                    if ensamble_predictions.values().count(predicted_priority_class) <=  ensamble_predictions.values().count(cl):
                        predicted_priority_class = cl
                        
        elif args.model_type in ['binary_multilabel_classifier','LSTM_classifier','metric_learning']:
           predicted_classes = None
           if args.model_type in ['binary_multilabel_classifier','metric_learning']:
               predicted_classes = []
               if args.ensamble_type ==  'avg_probs':
                   for cl in args.net_output_classes:
                       if cl == 'normal':
                           continue
                       pred = softmax(args.predictions_vec[0][current_id][cl])[0]
                       for i in range(1,len(args.predictions_vec)):
                           pred = pred + softmax(args.predictions_vec[i][current_id][cl])[0]
                       pred = pred / float(len(args.predictions_vec))
                       to_use_th = th
                       if th == 'best_ths':
                           to_use_th = args.thresholds[cl]                         
                       if pred > to_use_th:                   
                           predicted_classes.append(cl)
               elif args.ensamble_type ==  'avg_classification_res':
                   for cl in args.net_output_classes:
                       if cl != 'normal':
                           ensamble_predictions = {} 
                           for i in range(len(args.predictions_vec)): 
                               pred = softmax(args.predictions_vec[i][current_id][cl])[0]
                               to_use_th = th
                               if th == 'best_ths':
                                   to_use_th = args.thresholds_vec[i][cl]                               
                               if pred > to_use_th:
                                   ensamble_predictions[i] = 1
                               else: 
                                   ensamble_predictions[i] = 0

                           to_use_th = th
                           if th == 'best_ths':
                               to_use_th = args.thresholds[cl]
                           if ensamble_predictions.values().count(1) / float(len(ensamble_predictions.values())) >= to_use_th:
                               predicted_classes.append(cl)
    
               if len(predicted_classes) == 0:
                   predicted_classes.append('normal') 
           else:    
               predicted_classes = args.predictions_vec[0][current_id]
               
           if args.labels_version == 'v11' and args.net_output_version == 'v10':
               predicted_classes = fromV10toV11(predicted_classes) 
    
           
#           if ('cardiomegaly' in predicted_classes or 'Cardiomegaly') in predicted_classes and AP_PA_flag == 'AP':
#               predicted_classes.remove('cardiomegaly')
                                         
           for cl in args.result_classes:
               if cl in predicted_classes:
                   y_pred['avg_model'][th][cl].append(cl)
               else:
                   y_pred['avg_model'][th][cl].append("non_"+cl)

           predicted_priority_class = fromLabelsToPriorityClass(predicted_classes,args)              
               
        y_pred_pc['avg_model'][th].append(predicted_priority_class) 
       
def updateSingleModelPreds(y_pred, y_pred_pc,args,mdl_idx,current_id,AP_PA_flag):          
    
    for th in args.ALL_THRESHOLDS + ['best_ths']:
        predicted_priority_class = 'normal'
        if args.model_type == 'ordinal_regression_priority_classifier':
            for cl in ["non_urgent_finding","urgent","critical"]:                        
               pred = softmax(args.predictions_vec[mdl_idx][current_id][cl])[1]
               to_use_th = th
               if th == 'best_ths':
                   if len(args.thresholds_vec) > mdl_idx:
                       to_use_th = args.thresholds_vec[mdl_idx][cl] 
                   else:
                       to_use_th = args.threshold    
               if pred > to_use_th:
                   predicted_priority_class = cl
               else:
                   break  
        elif args.model_type in ['binary_multilabel_classifier','LSTM_classifier','metric_learning']:
           predicted_classes = None
           if args.model_type in ['binary_multilabel_classifier','metric_learning']:
               predicted_classes = []
               for cl in args.net_output_classes:
                   if cl == 'normal':
                       continue
                   pred = softmax(args.predictions_vec[mdl_idx][current_id][cl])[0]
                   to_use_th = th
                   if th == 'best_ths' :
                       if len(args.thresholds_vec) > mdl_idx:
                           to_use_th = args.thresholds_vec[mdl_idx][cl]
                       else:
                           to_use_th = args.threshold
                   if pred > to_use_th:                   
                       predicted_classes.append(cl)                           
               if len(predicted_classes) == 0:
                   predicted_classes.append('normal') 
           else:    
               predicted_classes = args.predictions_vec[mdl_idx][current_id]  
               
           if args.labels_version == 'v11' and args.net_output_version == 'v10':
               predicted_classes = fromV10toV11(predicted_classes) 

       
#           if ('cardiomegaly' in predicted_classes or 'Cardiomegaly') in predicted_classes and AP_PA_flag == 'AP':
#               predicted_classes.remove('cardiomegaly')
               
           for cl in args.result_classes:
               if cl in predicted_classes:
                   y_pred[th][cl].append(cl)
               else:
                   y_pred[th][cl].append("non_"+cl)
               
           predicted_priority_class = fromLabelsToPriorityClass(predicted_classes,args)                              
       
        y_pred_pc[th].append(predicted_priority_class)
                           
def initPredVars(args):
    y_true,y_pred ={},{}
    for cl in args.result_classes:
        y_true[cl] = []
        y_pred['avg_model'] = {} 
        y_pred['avg_model']['best_ths'] = {}
        for cl in args.result_classes:
            y_pred['avg_model']['best_ths'][cl] = []        
        for th in args.ALL_THRESHOLDS:
            y_pred['avg_model'][th] = {}
            for cl in args.result_classes: 
                y_pred['avg_model'][th][cl] = [] 
        for i in range(len(args.predictions_vec)):
            y_pred[i] = {}
            y_pred[i]['best_ths'] = {}
            for cl in args.result_classes:
                y_pred[i]['best_ths'][cl] = []
            for th in args.ALL_THRESHOLDS:
                y_pred[i][th] = {}
                for cl in args.result_classes:
                    y_pred[i][th][cl] = []
    
    return y_true,y_pred
 
def initPredPcVars(args):    
    y_true_pc, y_pred_pc =  [], {}
    y_pred_pc['avg_model'] = {}  
    y_pred_pc['avg_model']['best_ths'] = []
    for th in args.ALL_THRESHOLDS:      
        y_pred_pc['avg_model'][th] = []     
    for i in range(len(args.predictions_vec)):
        y_pred_pc[i] = {}
        y_pred_pc[i]['best_ths'] = []
        for th in args.ALL_THRESHOLDS:      
            y_pred_pc[i][th] = []  
    return y_true_pc, y_pred_pc

def plotROC(fpr, tpr,roc_auc,class_name):
    
    matplotlib.rc('font', size=20)
#    plt.title('Receiver Operating Characteristic curve for '+class_name +" classification",fontsize='xx-large')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right',fontsize='large')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Sensitivity',fontsize='large')
    plt.xlabel('1 - Specificity',fontsize='large')
    plt.xticks(np.arange(0.1,1.1,0.1),["{}%".format(el) for el in range(10,110,10)],fontsize="small")
    plt.yticks(np.arange(0.1,1.1,0.1),["{}%".format(el) for el in range(10,110,10)],fontsize="small") 
    plt.show()


def fromPriorityClassesToPositiveVSNegative(labels, positive_class):
    #mapping = {"normal":"normal","urgent":"non_normal","critical":"non_normal","non_urgent_finding":"non_normal"}
    return ["positive" if l==positive_class else "negative" for l in labels]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_results_filepaths', required=True)
    parser.add_argument('--db',default='exams2016_2017',choices=['exams2016_2017', 'kcl_test','CXR8'])
    parser.add_argument('--threshold',default=0.5,type=float)
    parser.add_argument('--model_type',default='binary_multilabel_classifier',choices=['binary_multilabel_classifier', 'ordinal_regression_priority_classifier','metric_learning','LSTM_classifier'])
    parser.add_argument('--best_thresholds_files',default='')    
    parser.add_argument('--ensamble_type',default='avg_probs',choices=['avg_probs','avg_classification_res'])
    parser.add_argument('--labels_version',default='v10',choices=['v10','v11','v12','CXR8'])
    parser.add_argument('--net_output_version',default='v10',choices=['v10','v11','v12','CXR8'])
    parser.add_argument('--set',default='test',choices=['test','gl','batch3','CXR8'])
    parser.add_argument('--base_dir',default='/home/ma16/x-rays_chest/')
    parser.add_argument('--dateStart', default='')
    parser.add_argument('--dateEnd', default='')
    parser.add_argument('--patientType', default="")
    
    args = parser.parse_args()  
    BASEDIR = args.base_dir
    if args.model_type == 'metric_learning': 
        BASEDIR = BASEDIR + 'metric_learning/results/' 
        NET_RESULTS_FILE = '/net_predictions_' + args.set + '.pkl'
    else:
        BASEDIR = BASEDIR + 'multi-class_classification/results/'
        NET_RESULTS_FILE = '/test_best_F1score_model/' + args.set + '_set/net_predictions.t7'    
        BEST_F1_SCORE_FILE = '/best_F1Score_thresholds.t7'
    

    v10_CLASSES = ['object','normal','pleural_effusion','cardiomegaly', 'hernia','pneumothorax','consolidation', 'scoliosis', 'pleural_abnormality','atelectasis','parenchymal_lesion', 'interstitial_shadowing', 'dextrocardia', 'hyperexpanded_lungs', 'cavitating_lung_lesion','left_upper_lobe_collapse', 'rib_fracture','pneumoperitoneum','dilated_bowel', 'pneumomediastinum','subcutaneous_emphysema','bulla','emphysema', 'rib_lesion','right_upper_lobe_collapse', 'hemidiaphragm_elevated',  'right_lower_lobe_collapse', 'unfolded_aorta',  'bronchial_wall_thickening', 'clavicle_fracture', 'left_lower_lobe_collapse',  'mediastinum_widened', 'paratracheal_hilar_enlargement',  'ground_glass_opacification', 'right_middle_lobe_collapse', 'aortic_calcification', 'mediastinum_displaced']    
    v11_CLASSES = ['abnormal_other','airspace_opacifation','cardiomegaly','collapse', 'hiatus_hernia','interstitial_shadowing','intraabdominal_pathology','object','normal','pneumothorax', 'bone_abnormality', 'paratracheal_hilar_enlargement','pleural_abnormality','parenchymal_lesion','pneumomediastinum','subcutaneous_emphysema']    
    v12_CLASSES = ['object','normal','pleural_effusion','cardiomegaly', 'hernia','pneumothorax','consolidation', 'scoliosis', 'pleural_abnormality','atelectasis','parenchymal_lesion', 'interstitial_shadowing', 'dextrocardia', 'hyperexpanded_lungs', 'cavitating_lung_lesion','collapse', 'rib_fracture','pneumoperitoneum','dilated_bowel', 'pneumomediastinum','subcutaneous_emphysema','bulla','emphysema', 'rib_lesion', 'hemidiaphragm_elevated', 'unfolded_aorta',  'bronchial_wall_thickening', 'clavicle_fracture',  'mediastinum_widened', 'paratracheal_hilar_enlargement',  'ground_glass_opacification', 'aortic_calcification', 'mediastinum_displaced']    
    CXR8_CLASSES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia','normal']
#    args.ALL_THRESHOLDS = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    args.ALL_THRESHOLDS = []
    for i in np.arange(0.0,1.01,0.01):
        args.ALL_THRESHOLDS.append(i)
        
    if args.db == 'exams2016_2017':
        collection = MongoClient("mongodb://localhost:27017").exams2016_2017.records
    elif args.db == 'kcl_test':
        collection = MongoClient("mongodb://localhost:27017").kcl_test.records
    elif args.db == 'CXR8':
        collection = DataInfos('/montana-storage/shared/data/chest_xrays/CXR8/Data_Entry_2017.csv')
    else:
        raise Exception('Unknown DB name '+ args.db)
    
    mongoDB_filter = {}
    if args.patientType != "":
        mongoDB_filter["Patient Type Des"] =  {'$in': args.patientType.split(',')} 
    
    if args.dateStart != "":
        splitted_start_date_string = args.dateStart.split("-")
        start_date = dt.datetime(int(splitted_start_date_string[2]), int(splitted_start_date_string[1]), int(splitted_start_date_string[0]))
        mongoDB_filter['Date reported'] = {'$gte':start_date}
        mongoDB_filter['Event Date'] = {'$gte':start_date}
    if args.dateEnd != "":
        splitted_end_date_string = args.dateEnd.split("-")
        end_date = dt.datetime(int(splitted_end_date_string[2]), int(splitted_end_date_string[1]), int(splitted_end_date_string[0]))
        if not 'Event Date' in mongoDB_filter.keys():
             mongoDB_filter['Event Date'] = {}
        mongoDB_filter['Event Date']['$lt'] = end_date
    
    
    if args.labels_version == 'v10':
        args.result_classes = v10_CLASSES
    elif args.labels_version == 'v11':
        args.result_classes = v11_CLASSES
    elif args.labels_version == 'v12':
        args.result_classes = v12_CLASSES
    elif args.labels_version == 'CXR8':
        args.result_classes = CXR8_CLASSES
    args.result_classes.sort()
    
    if args.net_output_version == 'v10':  
        args.net_output_classes = v10_CLASSES
    elif args.net_output_version == 'v11':
        args.net_output_classes = v11_CLASSES 
    elif args.net_output_version == 'v12':
        args.net_output_classes = v12_CLASSES 
    elif args.labels_version == 'CXR8':
        args.net_output_classes = CXR8_CLASSES
        
        
    
    args.predictions_vec = []
    torch_results_filepaths = args.torch_results_filepaths.split(',')
    for torch_results_filepath in torch_results_filepaths: 
        filepath = BASEDIR + torch_results_filepath + NET_RESULTS_FILE
        if args.model_type == 'metric_learning': 
            args.predictions_vec.append(pickle.load(open( filepath,"rb")))
        else:
            args.predictions_vec.append(torchfile.load(filepath))

    args.thresholds,args.thresholds_vec = {},[]
        
    if args.best_thresholds_files != "":
        best_thresholds_files = args.best_thresholds_files.split(',')
        if len(best_thresholds_files) != len(args.predictions_vec):
            raise Exception('error')
                
        for best_thresholds_file in best_thresholds_files:
            filepath = BASEDIR + best_thresholds_file + BEST_F1_SCORE_FILE
            curr_thresholds = torchfile.load(filepath)
            if len(args.thresholds.keys()) == 0:
                args.thresholds = curr_thresholds
            else:
                for cl in curr_thresholds.keys():
                    args.thresholds[cl] = args.thresholds[cl] + curr_thresholds[cl]  
                                           
        for cl in args.thresholds.keys():
            args.thresholds[cl] = args.thresholds[cl] / len(best_thresholds_files)

        for best_thresholds_file in best_thresholds_files:
            filepath = BASEDIR + best_thresholds_file + BEST_F1_SCORE_FILE
            args.thresholds_vec.append(torchfile.load(filepath))
    
#        print(args.thresholds,args.thresholds_vec)
     
    y_true,y_pred = None,None                      
    if args.model_type in ['binary_multilabel_classifier','LSTM_classifier','metric_learning']:
        y_true,y_pred = initPredVars(args)

             
    args.priority_list,args.per_priority_mapping = definePriorityMapping(v10_CLASSES+v11_CLASSES+CXR8_CLASSES)#result_classes)    
    
    y_true_pc, y_pred_pc = initPredPcVars(args)        
#    out_file = open("output.txt","w") 

    for _id in args.predictions_vec[0].keys():
       mongoDB_filter['_id'] = _id
       docs = collection.find(mongoDB_filter)

       if docs.count() == 1:
           d = docs[0]
           groundTruthClasses = d['Annotations']['Report']['labels']
           if 'golden_labels_v10' in d['Annotations']:
               groundTruthClasses = d['Annotations']['golden_labels_v10']
           
           if 'unclassifiable' in groundTruthClasses: 
               continue
              
           NLP_priority_class = fromLabelsToPriorityClass(groundTruthClasses,args) 
           y_true_pc.append(NLP_priority_class)

           if args.labels_version == 'v11':
              groundTruthClasses = fromV10toV11(groundTruthClasses)
           elif args.labels_version == 'v12':
              groundTruthClasses = fromV10toV12(groundTruthClasses)

           if args.model_type != 'ordinal_regression_priority_classifier':
               for cl in args.result_classes:
                   if cl in groundTruthClasses:
                       y_true[cl].append(cl)
                   else:
                       y_true[cl].append("non_"+cl)  
               
           if len(args.predictions_vec) > 1:
               updateAveragePreds(y_pred,y_pred_pc,args,_id,getPosition(d))
#               gt_lbls_str = "|".join([str(el) for el in d['Annotations']['golden_labels_v10']])
#               pred_priority = y_pred_pc['avg_model']["best_ths"][len(y_pred_pc['avg_model']["best_ths"])-1]
#               if y_true_pc[len(y_true_pc)-1] in ['critical'] and y_pred_pc['avg_model']["best_ths"][len(y_pred_pc['avg_model']["best_ths"])-1] in ['normal']:
#                   print(_id, y_true_pc[len(y_true_pc)-1], y_pred_pc['avg_model']["best_ths"][len(y_pred_pc['avg_model']["best_ths"])-1])
#               if y_true_pc[len(y_true_pc)-1] in ['urgent'] and y_pred_pc['avg_model']["best_ths"][len(y_pred_pc['avg_model']["best_ths"])-1] in ['normal']:
#                   if y_true_pc[len(y_true_pc)-1]  != "normal":
#                       out_file.write("{},{},{},{},\"{}\"\n".format(_id,y_true_pc[len(y_true_pc)-1],gt_lbls_str,pred_priority,d['Rep Text with Spacing']))
#                   

           else:  
               updateSingleModelPreds(y_pred[0], y_pred_pc[0],args,0,_id,getPosition(d))
#    out_file.close()
 
    print("completed")       

#       else:
#           raise Exception(_id +' not included in db '+ args.db)
    
    for idx in list(range(len(args.predictions_vec))) + ['avg_model']:
        if len(args.predictions_vec) == 1 and idx == 'avg_model':
            continue
        elif len(args.predictions_vec) > 1 and idx != 'avg_model':
            continue
        

        priority_class_cf = confusion_matrix(y_true_pc, y_pred_pc[idx]["best_ths"], labels=["normal","non_urgent_finding","urgent","critical"])  
        print(priority_class_cf)   
        
        for pr_cl in args.priority_list.keys():
            TPR_vec,FPR_vec = [],[]
            for th in args.ALL_THRESHOLDS:
                cl_cf = confusion_matrix(fromPriorityClassesToPositiveVSNegative(y_true_pc, pr_cl), fromPriorityClassesToPositiveVSNegative(y_pred_pc[idx][th], pr_cl), labels=["positive","negative"])
                Sensitivity_Recall, Specificity, _, _, _ = getCM_Measures(cl_cf)
    
                TPR_vec.append(Sensitivity_Recall)
                FPR_vec.append(1-Specificity)
                         
            roc_auc_score = auc(FPR_vec,TPR_vec,reorder=True)
            plotROC(FPR_vec, TPR_vec,roc_auc_score,pr_cl)
            print(pr_cl,'=',roc_auc_score)    
    
        if args.model_type in ['binary_multilabel_classifier','LSTM_classifier','metric_learning']:
            Sensitivity_Recall_sum, Specificity_sum, Precision_sum, F1_Score_sum, avg_accuracy_sum = 0,0,0,0,0
            for cl in args.result_classes:
                cl_cf = confusion_matrix(y_true[cl], y_pred[idx]["best_ths"][cl], labels=[cl,"non_"+cl]) 
                Sensitivity_Recall, Specificity, Precision, F1_Score, avg_accuracy = printConfMatrix(cl_cf,cl)
                Sensitivity_Recall_sum, Specificity_sum, Precision_sum, F1_Score_sum, avg_accuracy_sum = Sensitivity_Recall_sum + Sensitivity_Recall, Specificity_sum + Specificity, Precision_sum + Precision, F1_Score_sum + F1_Score, avg_accuracy_sum + avg_accuracy
            
            print('\nAvg. Recall',round(Sensitivity_Recall_sum / float(len(args.result_classes)),4))
            print('Avg. Specificity',round(Specificity_sum / float(len(args.result_classes)),4))
            print('Avg. Precision',round(Precision_sum / float(len(args.result_classes)),4))
            print('Avg. F1 Score',round(F1_Score_sum / float(len(args.result_classes)),4))
            print('Avg. avg accuracy',round(avg_accuracy_sum / float(len(args.result_classes)),4),"\n")
        
            print('ROC AUC scores:')
            
            
            for cl in args.result_classes:
                TPR_vec,FPR_vec = [],[]
                for th in args.ALL_THRESHOLDS:
                    cl_cf = confusion_matrix(y_true[cl], y_pred[idx][th][cl], labels=[cl,"non_"+cl])
                    Sensitivity_Recall, Specificity, _, _, _ = getCM_Measures(cl_cf)
    
                    TPR_vec.append(Sensitivity_Recall)
                    FPR_vec.append(1-Specificity)
                    
                
                roc_auc_score = auc(FPR_vec,TPR_vec)
                plotROC(FPR_vec, TPR_vec,roc_auc_score,cl)
                print(cl,'=',roc_auc_score)
#                if cl == 'cardiomegaly':
#                    plt.plot(FPR_vec,TPR_vec)   
#                    plt.ylabel('TPR')
#                    plt.xlabel('FPR')
#                    plt.title(cl + ' ROC')
#                    plt.show()  

if __name__ == "__main__":
    main()
