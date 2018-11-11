# -*- coding: utf-8 -*-

import argparse
import pymongo
import datetime as dt
import Queue
import random
import pickle
import numpy as np
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import main as m

def plotDistributions(distribution,values,label):
    matplotlib.rc('font', size=20)
    bins = np.arange(0,30,0.03125)#0.125) #range(0,125,0.1) 
    targets = ["AI prioritisation","Observed","FIFO"]
    tcolors = {"AI prioritisation":"lime","Observed":"indigo","FIFO":"red"}
    sns.distplot(distribution, hist=True, kde=False,label=label,bins=bins,norm_hist=False)
    for i in range(2):
        p_value = (np.asarray(distribution) < values[i]).sum()/float(len(distribution))
        p_value_string = (" (p-value = {:.2f})".format(p_value) if p_value >= 0.001 else " (p-value < 0.001)")
        plt.axvline(values[i],color=tcolors[targets[i]],label=targets[i]+p_value_string)
        plt.text(values[i],22500,"x = {:.02f}".format(values[i]),rotation=90, verticalalignment='center', horizontalalignment="right", fontsize="x-large",color=tcolors[targets[i]])

    plt.xlabel('Mean reporting time from acquisition [in days]',fontsize='x-large')
    plt.ylabel('Bin dimension',fontsize='x-large')
    plt.legend(fontsize='x-large',frameon=True,framealpha=1.0,loc='upper left')
    #min_value,max_value = int(min(values + [np.min(distribution)]))-1,int(max(values + [np.mean(distribution)]))+1
    #plt.xlim([min_value,max_value])
    plt.xlim([0,13])
            
    #plt.xticks(values+oth_values,fontsize='xx-large')
    plt.xticks(np.array(range(0,14,1)),fontsize="large")
    plt.yticks(np.arange(0,45000,5000),fontsize="large")        
    plt.show()
    plt.close() 
        
def plotResults(result_set,priorities):

    sns.set(color_codes=True)

    mean_targets,std_targets = {"critical":[2.7,12.1,10.5],"urgent":[4.1,8.4,10.2],"non_urgent_finding":[4.4,12.7,10.3],"normal":[13.0,8.2,10.1]},{"critical":[11.88,19.57,2.88],"urgent":[15.76,16.71,3.00],"non_urgent_finding":[18.81,20.27,2.98],"normal":[24.18,15.91,3.00]}
    targets = ["ours","observed","FIFO"]
    for cl in priorities:
        print len(result_set["means"][cl]),np.mean(result_set["means"][cl]),np.std(result_set["means"][cl])
        print len(result_set["stds"][cl]),np.mean(result_set["stds"][cl]),np.std(result_set["stds"][cl])
        for i in range(3):
            print cl,"mean p-value "+targets[i],(np.asarray(result_set["means"][cl]) < mean_targets[cl][i]).sum()/float(len(result_set["means"][cl]))
            print cl,"std p-value "+targets[i],(np.asarray(result_set["stds"][cl]) < std_targets[cl][i]).sum()/float(len(result_set["stds"][cl])) 

        plotDistributions(result_set["means"][cl],mean_targets[cl],"Randomly assigned priority levels\n("+" ".join(cl.split("_"))+" exams)")
        plotDistributions(result_set["stds"][cl],std_targets[cl],"stds distribution for "+" ".join(cl.split("_"))+" exams")

              

    
    


def simulateRandomPredictions(args):
    with open(args.acc_nums_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    acc_numbs = [x.strip() for x in content] 

    collection = pymongo.MongoClient("mongodb://localhost:27017").exams2016_2017.records#.kcl_test.records
    mongoDB_filter = {}  
    if args.patientType != "":
        mongoDB_filter["Patient Type Des"] =  {'$in': args.patientType.split(',')} 

    mongoDB_filter['Annotations.Report.labels'] = {'$exists':True,'$not': {'$size': 0}}
    
    priority_list,per_priority_mapping = m.definePriorityMapping()
    priorities = ['critical','urgent','non_urgent_finding','normal']   

    assert not args.dateStart is None and len(args.dateStart.split("-")) == 3,"dateStart mandatory input. Format must be dd-mm-aaaa"
    assert not args.dateEnd is None and len(args.dateEnd.split("-")) == 3,"dateEnd mandatory input. Format must be dd-mm-aaaa"

    mongoDB_filter['_id'] = {'$in':acc_numbs} 

    exams_dict = {}
    means,stds = {},{}
    for p in priorities:
        means[p],stds[p] = [],[]

    last_cp_file = None
    for h in range(args.B/5000):
        if os.path.isfile('temp_mean_and_std.checkpoint_' + str(h) +'.pkl'):
            last_cp_file= 'temp_mean_and_std.checkpoint_' + str(h) +'.pkl'
            continue
        
        if not last_cp_file is None:
            print "Restart simulatation from ",h-1,"/",args.B/5000
            pk_dict = pickle.load(open(last_cp_file,"rb"))
            means,stds = pk_dict["means_list"], pk_dict["stds_list"]
            
        exams = collection.find(mongoDB_filter)#.sort([("Event Date",1)])
        
        splitted_start_date_string = args.dateStart.split("-")
        start_date = dt.datetime(int(splitted_start_date_string[2]), int(splitted_start_date_string[1]), int(splitted_start_date_string[0]))
        stats_start_date = start_date
        start_date = start_date - dt.timedelta(days=30)
        
        splitted_end_date_string = args.dateEnd.split("-")
        end_date = dt.datetime(int(splitted_end_date_string[2]), int(splitted_end_date_string[1]), int(splitted_end_date_string[0]))
        
        mongoDB_filter['Event Date'] = {'$lt':end_date, '$gte':start_date} 
        mongoDB_filter['Date reported'] = {'$gte':start_date}         
        
        reports_time_bins, events_time_bins = {}, {} 
        
        
        for exam in exams:
            real_reporting_time = (exam['Date reported'] - exam['Event Date']).total_seconds()
            if real_reporting_time >= 0:
                exam_infos = m.getInfosFromDoc(exam)
                exams_dict[exam_infos['_id']] = exam_infos
            
                reporting_time_bin_index = int((exam_infos['Date reported']-dt.datetime.min).total_seconds()) // int(60*60*args.discrete_time_bins)
                if not reporting_time_bin_index in reports_time_bins.keys():
                    reports_time_bins[reporting_time_bin_index] = []
                reports_time_bins[reporting_time_bin_index].append(exam_infos['_id'])
                
                event_time_bin_index = int((exam_infos['Event Date']-dt.datetime.min).total_seconds()) // int(60*60*args.discrete_time_bins)
                if not event_time_bin_index in events_time_bins.keys():
                    events_time_bins[event_time_bin_index] = []
                events_time_bins[event_time_bin_index].append(exam_infos['_id'])    
        
        
        q_priority_list = [Queue.PriorityQueue() for i in range(5000)]  
        rep_del_rand_prior_list = [{'normal':{},'non_urgent_finding':{},'urgent':{},'critical':{}} for i in range(5000)]          
        
        counter,end_loop = 0,int((end_date - start_date).total_seconds()) // int(60*60*args.discrete_time_bins)
        while start_date < end_date:
            current_time_bin_index = int((start_date-dt.datetime.min).total_seconds()) // int(60*60*args.discrete_time_bins)
    
            #1) how many exams were reported in this time bin?
            if current_time_bin_index in reports_time_bins:
                reported_count = len(reports_time_bins[current_time_bin_index])
            else:
                reported_count = 0
    
            #2) pop reported_count exams from the priority queues
            for j in range(5000):                
                for i in range(reported_count):
                    if not q_priority_list[j].empty():
                        class_id_tuple = q_priority_list[j].get(timeout=0)
                        d = exams_dict[class_id_tuple[1]]#collection.find({'_id':class_id_tuple[1]})[0]               
                        NLP_priority_class = m.getPriorityClass(d['classes'],priority_list,per_priority_mapping)                       
                        if d['Event Date'] >= stats_start_date:
                            rep_del_rand_prior_list[j][NLP_priority_class][class_id_tuple[1]] = (start_date + dt.timedelta(seconds=60*60*args.discrete_time_bins) - d['Event Date']).total_seconds()  / 60.0 / 60.0
                    else:
                        break   
                    
                
            #3) fill the queues for the next time bin
            if current_time_bin_index in events_time_bins.keys():           
                for acc_numb in events_time_bins[current_time_bin_index]:
                    d = exams_dict[acc_numb]
                    #NLP_priority_class = m.getPriorityClass(d['classes'],priority_list,per_priority_mapping)
                    for j in range(5000):
                        random_priority_level = random.choice([1,2,3,4])                  
                        q_priority_list[j].put((random_priority_level,d['_id']))
              
                
            #Move to next bin:
            start_date = start_date + dt.timedelta(seconds=60*60*args.discrete_time_bins)
            counter = counter + 1                       
            if counter % 25000 == 0:
                print counter,"/",end_loop    
        
        print(h,args.B/5000)         
    
        for cl in priorities:
            for j in range(5000):
                if len(rep_del_rand_prior_list[j][cl].values()) > 0: 
                    means[cl].append(np.mean(rep_del_rand_prior_list[j][cl].values()) / 24.0)
                    stds[cl].append(np.std(rep_del_rand_prior_list[j][cl].values()) / 24.0)
        with open('temp_mean_and_std.checkpoint_' + str(h) +'.pkl', 'wb') as f:
            pickle.dump({"means_list":means,"stds_list":stds}, f, pickle.HIGHEST_PROTOCOL)          
    return means,stds
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dateStart')
    parser.add_argument('--dateEnd')
    parser.add_argument('--patientType', default="")
    parser.add_argument('--discrete_time_bins', default=24,type=float)
    parser.add_argument('--B', default=5000,type=int)
    parser.add_argument('--acc_nums_file')

    args = parser.parse_args()
    assert args.B >= 5000 and args.B%5000==0,"--B must be a multiple of 5000 and equal or greater than 5000"
    if not os.path.isfile('temp_mean_and_std.pkl'):
        means,stds = simulateRandomPredictions(args)
        with open('temp_mean_and_std.pkl', 'wb') as f:
            pickle.dump({"means_list":means,"stds_list":stds}, f, pickle.HIGHEST_PROTOCOL)        
    else:
        pk_dict = pickle.load(open('temp_mean_and_std.pkl',"rb"))
        means,stds = pk_dict["means_list"],pk_dict["stds_list"]
    
    plotResults({"means":means,"stds":stds},['critical','urgent','non_urgent_finding','normal'])              
        
if __name__ == "__main__":
    main()
