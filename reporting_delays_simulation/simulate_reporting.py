# -*- coding: utf-8 -*-

import argparse
import pymongo
import datetime as dt
import Queue
import numpy as np 
import torchfile
import random
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats 

def definePriorityMapping():
    
    priority_list = {}
    priority_list["normal"],priority_list["non_urgent_finding"],priority_list["urgent"],priority_list["critical"] = 16,12,8,4
    all_classes_v10 = ['object','normal','pleural_effusion','cardiomegaly', 'hernia','pneumothorax','consolidation', 'scoliosis', 'pleural_abnormality','atelectasis','parenchymal_lesion', 'interstitial_shadowing', 'dextrocardia', 'hyperexpanded_lungs', 'cavitating_lung_lesion','left_upper_lobe_collapse', 'rib_fracture','pneumoperitoneum','dilated_bowel', 'pneumomediastinum','subcutaneous_emphysema','bulla','emphysema', 'rib_lesion','right_upper_lobe_collapse', 'hemidiaphragm_elevated',  'right_lower_lobe_collapse', 'unfolded_aorta',  'bronchial_wall_thickening', 'clavicle_fracture', 'left_lower_lobe_collapse',  'mediastinum_widened', 'paratracheal_hilar_enlargement',  'ground_glass_opacification', 'right_middle_lobe_collapse', 'aortic_calcification', 'mediastinum_displaced']    
    per_priority_mapping = {}
    for cl in all_classes_v10:
        if cl == "pneumomediastinum" or cl == "pneumothorax" or cl == "subcutaneous_emphysema" \
           or cl == "mediastinum_widened" or cl == "pneumoperitoneum":
            per_priority_mapping[cl] = "critical"
        elif cl == "consolidation" or cl == "pleural_abnormality" or cl == "parenchymal_lesion" or cl == "pleural_effusion" \
            or cl == "right_upper_lobe_collapse" or cl == "cavitating_lung_lesion" or cl == "rib_fracture" or cl == "interstitial_shadowing"\
            or cl == "left_lower_lobe_collapse" or cl == "clavicle_fracture" or cl == "paratracheal_hilar_enlargement" \
            or cl == "dilated_bowel" or cl == "ground_glass_opacification" or cl == "left_upper_lobe_collapse" or cl == "rib_lesion"\
            or cl == "right_lower_lobe_collapse" or cl == "mediastinum_displaced" or cl == "right_middle_lobe_collapse" \
            or cl == "widened_paratracheal_stripe" or cl == "enlarged_hilum" or cl == "pleural_thickening" or cl == "pleural_lesion":
            per_priority_mapping[cl] = "urgent"
        elif cl == "object" or cl == "cardiomegaly" or cl == "emphysema" or cl == "atelectasis" or cl == "bronchial_wall_thickening" \
               or cl == "scoliosis" or cl == "hernia" or cl == "hyperexpanded_lungs" or cl == "unfolded_aorta" \
               or cl == "dextrocardia" or cl == "bulla" or cl == "aortic_calcification" or cl == "hemidiaphragm_elevated":
            per_priority_mapping[cl] = "non_urgent_finding"  
        elif cl == "normal":
            per_priority_mapping[cl] = "normal"
            
    return priority_list,per_priority_mapping

def getPriorityClass(classes,priority_list,per_priority_mapping):
    priority_class = "normal"
    
    for cl in classes:
        if priority_list[priority_class] > priority_list[per_priority_mapping[cl]]:
            priority_class = per_priority_mapping[cl] 
    
    return priority_class

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def getPriorityClassFromDoc(d,priority_list,per_priority_mapping):
    return getPriorityClass(d['Annotations']['Report']['labels'],priority_list,per_priority_mapping)

def getPriorityClassFromOrdinalRegressionNetPredictions(preds,threshold):
    predicted_priority_class = "normal"
    confidence = preds["non_urgent_finding"][0]      
    for cl in ["non_urgent_finding","urgent","critical"]: 
        if preds[cl][1] > threshold: #preds[cl][0] < preds[cl][1]:
            predicted_priority_class = cl
            confidence = preds[cl][1]
        else:
            break
    return predicted_priority_class,confidence
                    
def avgPredictions(predictions_vec):
    final_predictions = {}
    if len(predictions_vec) == 1:
        for acc_numb in predictions_vec[0].keys():
            for cl in ["non_urgent_finding","urgent","critical"]:
                sf_max = softmax(predictions_vec[0][acc_numb][cl])
                for i in range(2):
                    final_predictions[acc_numb][cl][i] = sf_max[i]              
          
        return final_predictions
    elif len(predictions_vec) > 1:
 
        for acc_numb in predictions_vec[0].keys():
            final_predictions[acc_numb] = {"non_urgent_finding":[0,0],"urgent":[0,0],"critical":[0,0]}           
            for cl in ["non_urgent_finding","urgent","critical"]:
                for i in range(2):
                    for predictions in predictions_vec:
                        final_predictions[acc_numb][cl][i] =  final_predictions[acc_numb][cl][i] + softmax(predictions[acc_numb][cl])[i]              
                    final_predictions[acc_numb][cl][i] =  final_predictions[acc_numb][cl][i] / len(predictions_vec)
        return final_predictions
    else:
        raise Exception("Error, predictions_vec empty.")

def getInfosFromDoc(d):

    result = {
        '_id':str(d['_id']),
        'Event Date':d['Event Date'],
        'Date reported':d['Date reported'],
        'classes':d['Annotations']['Report']['labels']   
    }
    return result

def toPlotTag(tag):
    if tag == 'normal':
        return 'Normal'
    elif tag == 'urgent':
        return 'Urgent'
    elif tag == 'non_urgent_finding':
        return 'Non urgent'
    elif tag == 'critical':
        return 'Critical'

def printDistributionMeasures(in_dict):
    np_array = np.array(in_dict)
    mean = np_array.mean() / 24.0
    percentile_99 = np.percentile(np_array,99)
    mean_99 = np_array[np_array<percentile_99].mean() / 24.0 
    median = np.median(np_array) / 24.0 
    median_99 = np.median(np_array[np_array<percentile_99]) / 24.0
    std = np_array.std() / 24.0
    std_99 = np_array[np_array<percentile_99].std() / 24.0
    iqr = scipy.stats.iqr(np_array) 
    
    print "{:.2f} mean delay [days] - {:.2f} median delay [days] - {:.2f} std delay [days]".format(mean,median,std)
    print "{:.2f} 99 percentile, {:.2f} iqr".format(percentile_99/24.0,iqr/24.0)
    print "{:.2f} mean delay 99 percentile [days] - {:.2f} median delay 99 percentile [days] - {:.2f} std delay 99 percentile [days]".format(mean_99,median_99,std_99)
 
def plotResults(result_set,priorities):
#    import seaborn as sns
#    import pandas as pd  
    
#    sns.set(color_codes=True)
    bins = np.arange(0.0,80.0,1.00) #range(0,125,0.1) 
    
#    my_data = pd.DataFrame()
#    priority_class_vec,delay_vec,triaging_system_vec = [],[],[]
    for cl in priorities:
        print cl,len(result_set["Observed reporting"][cl].keys()),len(result_set["Priority queues simulation"][cl].keys()),len(result_set["FIFO queue simulation"][cl].keys())
        if len(result_set["Observed reporting"][cl].values()) > 0: 
            print "{} real:".format(cl)
            printDistributionMeasures(result_set["Observed reporting"][cl].values())
            #print cl,"real",round(np.mean(result_set["Observed reporting"][cl].values()) / 24.0,1), "avg. delay [days] ", "(std="+str(round(np.std(result_set["Observed reporting"][cl].values()) / 24.0,2))+")"    
        if len(result_set["Priority queues simulation"][cl].values()) > 0:
            print "{} simulated priority:".format(cl)
            printDistributionMeasures(result_set["Priority queues simulation"][cl].values())
            #print cl,"simulated priority",round(np.mean(result_set["Priority queues simulation"][cl].values()) / 24.0,1), "avg. delay [days]", "(std="+str(round(np.std(result_set["Priority queues simulation"][cl].values()) / 24.0,2))+")"
#        if len(result_set["FIFO queue simulation"][cl].values()) > 0:
#            print "{} simulated sequential:".format(cl)
#            printDistributionMeasures(result_set["FIFO queue simulation"][cl].values())
            #print cl,"simulated sequential",round(np.mean(result_set["FIFO queue simulation"][cl].values()) / 24.0,1), "avg. delay [days]", "(std="+str(round(np.std(result_set["FIFO queue simulation"][cl].values()) / 24.0,2))+")"                 

        #1) dividi i dati in bins
        obs_h = np.histogram([el / 24.0 for el in result_set["Observed reporting"][cl].values()],bins)[0]
        obs_h = obs_h/float(obs_h.sum())
        sim_pq_h = np.histogram([el / 24.0 for el in result_set["Priority queues simulation"][cl].values()],bins)[0]
        sim_pq_h = sim_pq_h/float(sim_pq_h.sum())
#        sim_fifo_h = np.histogram([el / 24.0 for el in result_set["FIFO queue simulation"][cl].values()],bins)[0]
#        sim_fifo_h = sim_fifo_h/float(sim_fifo_h.sum())
        
        #2) plotta i bins
        #plt.title(toPlotTag(cl))
        matplotlib.rc('font', size=24)
        plt.xlabel("Reporting delay [in days]",fontsize='large')
        plt.ylabel("Proportion of total",fontsize='large')        
#        plt.plot(bins[:-1]+0.5, obs_h, "o-",label='Observed reporting', clip_on=False)
#        plt.plot(bins[:-1]+0.5, sim_pq_h, "o-",label='Priority queues simulation', clip_on=False)
#        plt.plot(bins[:-1]+0.5, sim_fifo_h, "o-",label='FIFO queue simulation', clip_on=False)
        plt.bar(bins[:-1]+0.2, obs_h,width=0.3,color='red',align='center',label='Observed', clip_on=False)
        plt.bar(bins[:-1]+0.5, sim_pq_h,width=0.3,color='lime',align='center',label='AI prioritisation', clip_on=False)
#        plt.bar(bins[:-1]+0.8, sim_fifo_h,width=0.3,color='indigo',align='center',label='FIFO', clip_on=False)
        plt.xticks(fontsize='large')
        plt.yticks(np.arange(0.0,1.0,0.1),fontsize='large')          
        plt.xlim([0,80])
        plt.ylim([0,0.92])#bottom=0)
        plt.grid(True)
        plt.legend(fontsize='large')
        plt.show()
#        for k in result_set["Priority queues simulation"][cl].keys():
#            if not k in result_set["FIFO queue simulation"][cl]:
#                continue
#            for triaging_system in ["Priority queues simulation","FIFO queue simulation","Observed reporting"]:
#                priority_class_vec.append(toPlotTag(cl))
#                delay_vec.append(result_set[triaging_system][cl][k]/ 24.0)
#                triaging_system_vec.append(triaging_system)
#    
#    my_data["Priority class"] = priority_class_vec
#    my_data["delays"] = delay_vec
#    my_data["triaging_system"] = triaging_system_vec
#    
#    sns.boxplot(x="delays", y="Priority class", hue="triaging_system", data=my_data)#, split=True);
#    sns.plt.legend(fontsize='xx-large')
#    sns.plt.xlabel('Reporting delay [in days]',fontsize='x-large')
#    sns.plt.ylabel('',fontsize='x-large')
#    sns.plt.xticks(fontsize='large')
#    sns.plt.yticks(fontsize='large')  
#    sns.plt.show()

   
#    for cl in ['Critical','Urgent','Non urgent','Normal'] :
#        
#        target_observed = my_data.delays[(my_data['triaging_system'] == "Observed reporting") & (my_data['Priority class'] == cl)]
#        target_prior_queues = my_data.delays[(my_data['triaging_system'] == "Priority queues simulation") & (my_data['Priority class'] == cl)] 
#        target_FIFO_queue = my_data.delays[(my_data['triaging_system'] == "FIFO queue simulation") & (my_data['Priority class'] == cl)]        
#        sns.distplot(target_observed, hist=True, kde=False,bins=bins,label="Observed reporting",norm_hist=True)
#        sns.distplot(target_prior_queues, hist=True, kde=False,bins=bins,label="Priority queues simulation",color="green",norm_hist=True)
#        sns.distplot(target_FIFO_queue, hist=True, kde=False,bins=bins,label="FIFO queue simulation",color="red",norm_hist=True)
#        sns.plt.xlabel('Reporting delay [in days]',fontsize='xx-large')
#        sns.plt.ylabel('Proportion of total',fontsize='xx-large')
#        sns.plt.legend(fontsize='xx-large')
#        sns.plt.xlim([0,80])
#        sns.plt.xticks(fontsize='xx-large')
#        sns.plt.yticks(fontsize='xx-large')        
#        sns.plt.show()
#        sns.plt.close()    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dateStart')
    parser.add_argument('--dateEnd')
    parser.add_argument('--torch_results_filepaths')
    parser.add_argument('--patientType', default="")
    parser.add_argument('--discrete_time_bins', default=24,type=float)
    parser.add_argument('--base_dir',default='/home/ma16/x-rays_chest/multi-class_classification/results/')
    parser.add_argument('--noise_level', default=0.0,type=float)
    parser.add_argument('--threshold', default=0.5,type=float)
    parser.add_argument('--plotTempFiles',action='store_true')
    
    args = parser.parse_args()
    
    collection = pymongo.MongoClient("mongodb://localhost:27017").exams2016_2017.records#.kcl_test.records
    mongoDB_filter = {}  
    if args.patientType != "":
        mongoDB_filter["Patient Type Des"] =  {'$in': args.patientType.split(',')} 

    mongoDB_filter['Annotations.Report.labels'] = {'$exists':True,'$not': {'$size': 0}}

    priority_list,per_priority_mapping = definePriorityMapping()
    priorities = ['critical','urgent','non_urgent_finding','normal']   
 
    if args.plotTempFiles:
        result_set = {}   
        result_set["Priority queues simulation"] = pickle.load(open('report_delay_ourPrioritization.pkl',"rb"))
        result_set["FIFO queue simulation"] = pickle.load(open('report_delay_seqPrioritization.pkl',"rb"))
        result_set["Observed reporting"] = pickle.load(open('report_delay_real.pkl',"rb"))
        plotResults(result_set,priorities)
    else:
        assert not args.dateStart is None and len(args.dateStart.split("-")) == 3,"dateStart mandatory input. Format must be dd-mm-aaaa"
        assert not args.dateEnd is None and len(args.dateEnd.split("-")) == 3,"dateEnd mandatory input. Format must be dd-mm-aaaa"
                
        predictions_vec = []
        BASEDIR = args.base_dir
        NET_RESULTS_FILE = '/test_best_F1score_model/batch3_set/net_predictions.t7'    
        assert not args.torch_results_filepaths is None, "torch_results_filepaths mandatory arguments. Write the directories names separed by commas."
        torch_results_filepaths = args.torch_results_filepaths.split(',')
        for torch_results_filepath in torch_results_filepaths: 
            filepath = BASEDIR + torch_results_filepath + NET_RESULTS_FILE
            assert os.path.isfile(filepath),"file {} does not exists".format(filepath)
            predictions_vec.append(torchfile.load(filepath))   
        predictions = avgPredictions(predictions_vec) #torchfile.load(args.torch_results_filepath)
        
        print 'CNN predictions loaded.'
    
            
        splitted_start_date_string = args.dateStart.split("-")
        start_date = dt.datetime(int(splitted_start_date_string[2]), int(splitted_start_date_string[1]), int(splitted_start_date_string[0]))
        stats_start_date = start_date
        start_date = start_date - dt.timedelta(days=30)
        
        splitted_end_date_string = args.dateEnd.split("-")
        end_date = dt.datetime(int(splitted_end_date_string[2]), int(splitted_end_date_string[1]), int(splitted_end_date_string[0]))
        
    
        q_priority,q_sequential = Queue.PriorityQueue(),Queue.Queue()
        report_delay_real, report_delay_ourPrioritization,report_delay_seqPrioritization = {'normal':{},'non_urgent_finding':{},'urgent':{},'critical':{}},{'normal':{},'non_urgent_finding':{},'urgent':{},'critical':{}},{'normal':{},'non_urgent_finding':{},'urgent':{},'critical':{}}  
        
        exams_dict = {}
        
#        #fill the queue for the first time bin
#        #insert all the exams with start_date - 7 days < 'Event Date' < start_date and 'Date reported' > start_date         
#        mongoDB_filter['Event Date'] = {'$lt':start_date, '$gte':start_date - dt.timedelta(days=7)}        
#        mongoDB_filter['Date reported'] = {'$gte':start_date}    
#        not_yet_reported = collection.find(mongoDB_filter).sort([("Event Date",1)]) 
#         
#        for d in not_yet_reported:  
#            real_reporting_time = (d['Date reported'] - d['Event Date']).total_seconds()
#            if d['_id'] in predictions.keys() and real_reporting_time >= 0:
#                exam_infos = getInfosFromDoc(d)
#                exams_dict[exam_infos['_id']] = exam_infos            
#                
#                NLP_priority_class = getPriorityClassFromDoc(d,priority_list,per_priority_mapping)
#                predicted_priority_class = getPriorityClassFromOrdinalRegressionNetPredictions(predictions[d['_id']],args.threshold)
#    
#                q_priority.put((priority_list[predicted_priority_class],d['_id'])) 
#                q_sequential.put(d['_id'])
#                report_delay_real[NLP_priority_class][d['_id']] = real_reporting_time / 60.0 / 60.0
#    
#      
#        mongoDB_filter.pop('Event Date', None)
#        mongoDB_filter.pop('Date reported', None)    
        
        mongoDB_filter['Event Date'] = {'$lt':end_date, '$gte':start_date} 
        mongoDB_filter['Date reported'] = {'$gte':start_date} 
        
    #    collection.ensure_index([("Event Date", pymongo.ASCENDING)])  
        exams = collection.find(mongoDB_filter)#.sort([("Event Date",1)])
    #    exams_date_reported_ordering = collection.find(mongoDB_filter).sort([("Event Date",1)])
    
        reports_time_bins, events_time_bins = {}, {} 
        
        
        for exam in exams:
            real_reporting_time = (exam['Date reported'] - exam['Event Date']).total_seconds()
            if exam['_id'] in predictions.keys() and real_reporting_time >= 0:
                exam_infos = getInfosFromDoc(exam)
                exams_dict[exam_infos['_id']] = exam_infos
            
                reporting_time_bin_index = int((exam_infos['Date reported']-dt.datetime.min).total_seconds()) // int(60*60*args.discrete_time_bins)
                if not reporting_time_bin_index in reports_time_bins.keys():
                    reports_time_bins[reporting_time_bin_index] = []
                reports_time_bins[reporting_time_bin_index].append(exam_infos['_id'])
                
                event_time_bin_index = int((exam_infos['Event Date']-dt.datetime.min).total_seconds()) // int(60*60*args.discrete_time_bins)
                if not event_time_bin_index in events_time_bins.keys():
                    events_time_bins[event_time_bin_index] = []
                events_time_bins[event_time_bin_index].append(exam_infos['_id'])    
        
        counter,end_loop = 0,int((end_date - start_date).total_seconds()) // int(60*60*args.discrete_time_bins)
        while start_date < end_date:
            current_time_bin_index = int((start_date-dt.datetime.min).total_seconds()) // int(60*60*args.discrete_time_bins)
             
            #Evaluate current bin:
        
            #1) how many exams was reported in this time bin?
            if current_time_bin_index in reports_time_bins:
                reported_count = len(reports_time_bins[current_time_bin_index])
            else:
                reported_count = 0
                
            #print start_date.strftime("%Y-%m-%d %A %H:%M"),(start_date + dt.timedelta(seconds=60*60*args.discrete_time_bins)).strftime("%Y-%m-%d %A %H:%M"),"priority queue size = " + str(q_priority.qsize()),"FIFO queue size = " + str(q_sequential.qsize()),"Num. exams to add current bin = " + str(reported_count) 
            #2) pop reported_count exams from the priority queue                
            for i in range(reported_count):  
                if not q_priority.empty():
                    class_id_tuple = q_priority.get(timeout=0)
                    d = exams_dict[class_id_tuple[1]]#collection.find({'_id':class_id_tuple[1]})[0]               
                    NLP_priority_class = getPriorityClass(d['classes'],priority_list,per_priority_mapping)                       
                    if d['Event Date'] >= stats_start_date:
                        report_delay_ourPrioritization[NLP_priority_class][class_id_tuple[1]] = (start_date + dt.timedelta(seconds=60*60*args.discrete_time_bins) - d['Event Date']).total_seconds()  / 60.0 / 60.0
                else:
                    break   
                
            #3) pop reported_count exams from the sequential queue                
            for i in range(reported_count):  
                if not q_sequential.empty():
                    acc_numb = q_sequential.get(timeout=0)
                    d = exams_dict[acc_numb] #collection.find({'_id':acc_numb})[0]               
                    NLP_priority_class = getPriorityClass(d['classes'],priority_list,per_priority_mapping)                       
                    if d['Event Date'] >= stats_start_date:
                        report_delay_seqPrioritization[NLP_priority_class][acc_numb] = (start_date + dt.timedelta(seconds=60*60*args.discrete_time_bins) - d['Event Date']).total_seconds()  / 60.0 / 60.0
                else:
                    break
                
            #4) fill the queue for the next time bin
            if current_time_bin_index in events_time_bins.keys(): 
                for acc_numb in events_time_bins[current_time_bin_index]:
                    d = exams_dict[acc_numb]
                    NLP_priority_class = getPriorityClass(d['classes'],priority_list,per_priority_mapping)
                    predicted_priority_class,confidence = getPriorityClassFromOrdinalRegressionNetPredictions(predictions[d['_id']],args.threshold)             
                    priority_level = priority_list[predicted_priority_class]
                    if confidence > 0.8:
                        priority_level = priority_level - 3
                    elif confidence > 0.6:
                        priority_level = priority_level - 2
                    elif confidence > 0.4:
                        priority_level = priority_level - 1                     
                    #print predicted_priority_class,NLP_priority_class
                    
                    #4.a) randomly use the real delay based on the noise level
                    if random.uniform(0, 1) > args.noise_level:
                        q_priority.put((priority_level,d['_id']))
                        q_sequential.put(d['_id'])
                    else:
                        if d['Event Date'] >= stats_start_date:
                            report_delay_seqPrioritization[NLP_priority_class][d['_id']] = (d['Date reported'] - d['Event Date']).total_seconds() / 60.0 / 60.0 
                            report_delay_ourPrioritization[NLP_priority_class][d['_id']] = (d['Date reported'] - d['Event Date']).total_seconds() / 60.0 / 60.0 
                        this_exam_reporting_time_bin_index = int((d['Date reported']-dt.datetime.min).total_seconds()) // int(60*60*args.discrete_time_bins)
                        reports_time_bins[this_exam_reporting_time_bin_index].remove(reports_time_bins[this_exam_reporting_time_bin_index][0])
                        
                    if d['Event Date'] >= stats_start_date:    
                        report_delay_real[NLP_priority_class][d['_id']] = (d['Date reported'] - d['Event Date']).total_seconds() / 60.0 / 60.0            
                
            #Move to next bin:
            start_date = start_date + dt.timedelta(seconds=60*60*args.discrete_time_bins)
            counter = counter + 1                       
            if counter % 100000 == 0:
                print counter,"/",end_loop  
    
    
        result_set = {}   
        result_set["Priority queues simulation"] = report_delay_ourPrioritization
        result_set["FIFO queue simulation"] = report_delay_seqPrioritization
        result_set["Observed reporting"] = report_delay_real
        plotResults(result_set,priorities)  
        
     
        #save results in temp files, in this way you can restart the script and avoid the simulation
        with open('report_delay_real.pkl', 'wb') as f:
            pickle.dump(report_delay_real, f, pickle.HIGHEST_PROTOCOL)     
        with open('report_delay_ourPrioritization.pkl', 'wb') as f:
            pickle.dump(report_delay_ourPrioritization, f, pickle.HIGHEST_PROTOCOL)    
        with open('report_delay_seqPrioritization.pkl', 'wb') as f:
            pickle.dump(report_delay_seqPrioritization, f, pickle.HIGHEST_PROTOCOL)           
            
    #    splitted_start_date_string = args.dateStart.split("-")
    #    start_date = dt.datetime(int(splitted_start_date_string[2]), int(splitted_start_date_string[1]), int(splitted_start_date_string[0]))
    #    mongoDB_filter['Event Date'] = {'$lt':end_date, '$gte':start_date - dt.timedelta(days=7)} 
    #    mongoDB_filter['Date reported'] = {'$gte':start_date}        
    #    docs = collection.find(mongoDB_filter)
    #    priority_class_counter = {'normal':0,'non_urgent_finding':0,'urgent':0,'critical':0}
    #    for d in docs:
    #        if d['_id'] in predictions.keys(): 
    #            NLP_priority_class = getPriorityClassFromDoc(d,priority_list,per_priority_mapping)            
    #            priority_class_counter[NLP_priority_class] = priority_class_counter[NLP_priority_class] + 1
    #        
    #    print priority_class_counter

        
if __name__ == "__main__":
    main()