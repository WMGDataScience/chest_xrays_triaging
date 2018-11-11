----------------------------------------------------------------------
-- This script define the testing procedure
-- 
-- multi-thread data loading version
----------------------------------------------------------------------

require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'utils'

----------------------------------------------------------------------

local data_loading_threads = opt.dataLoadingThreads


----------------------------
-- Init confusion matrices
----------------------------

if opt.predict_age then 
    predict_age_error = 0 
end

if opt.multi_task then
    -- This matrix records the current confusion across classes   
    test_confusions = {}
    test_confusions['labels'] = {}   
    for i=1,#opt.classes do
        test_confusions['labels'][opt.classes[i]] = optim.ConfusionMatrix({opt.classes[i],"not_" .. opt.classes[i]})    
    end
 
    test_confusions['priority'] = optim.ConfusionMatrix({"normal","non_urgent_finding","urgent","critical"})

elseif opt.classifier_type == "multi" and opt.ordinal_regression == 0 then
    test_confusions = {}

    if opt.classify_per_priority == 1 or opt.test_per_priority == 1 then
        test_confusions = optim.ConfusionMatrix({"normal","non_urgent_finding","urgent","critical"})       
    else
        for i=1,#opt.classes do
            test_confusions[opt.classes[i]] = optim.ConfusionMatrix({opt.classes[i],"not_" .. opt.classes[i]})   
        end 
    end     

elseif opt.classifier_type == "single" or opt.ordinal_regression == 1 then
    local cm_input = {}
    
    if opt.ordinal_regression == 1 then
        for i=1,#opt.classes do
            table.insert(cm_input,opt.classes[i])
        end        
    else
        for i=1,#classes do
            table.insert(cm_input,classes[i])
        end
    end
    
    confusion = optim.ConfusionMatrix(cm_input)
end

----------------------------
-- Declare test variables
----------------------------

best_F1score_so_far = 0
local batchNumber,total_batches,total_images = 0,nil,nil 

-- GPU inputs (preallocate)
local inputs = createCudaTensor()

local timer = torch.Timer()

if covariate_variables_dim > 0 then
    covariate_inputs = createCudaTensor(opt.batchSize,covariate_variables_dim)
end
if opt.path_CH_embeddings ~= "" then
    covariate_inputs = createCudaTensor(torch.LongStorage({opt.batchSize,2,max_CH_len,CH_emb_size}))
end

local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local function updateConfMatrices(net_predictions,imgs_infos,classes)
    if best_thresholds == nil then
        local_confusions = {}
        for ii=1,#classes do
            local_confusions[classes[ii]] = {}
            for jj=1,#thresholds do
                local_confusions[classes[ii]][thresholds[jj]] =  optim.ConfusionMatrix({classes[ii],"not_" .. classes[ii]}) 
            end
        end
    end

    for acc_numb,preds in pairs(net_predictions) do
        if opt.multi_task then      
            for ii=1,#classes do   
                        
                local sftmax_res = nn.SoftMax():forward(preds[classes[ii]])                    
                local label = 2
                if table.contains(imgs_infos[acc_numb].classes,classes[ii]) then
                    label = 1
                end  
                                                             
                if best_thresholds ~= nil then
                    --we have best_thresholds, we do not have to calculate it
                    local current_prediction = 2 
                    if sftmax_res[1] > best_thresholds[classes[ii]] then
                        current_prediction = 1                     
                    end 
                        
                    test_confusions['labels'][classes[ii]]:add(current_prediction, label)               
                else 
                    
                    for jj=1,#thresholds do                   
                        local current_prediction = 2 
                        if sftmax_res[1] > thresholds[jj] then
                            current_prediction = 1
                        end
                        local_confusions[classes[ii]][thresholds[jj]]:add(current_prediction, label)
                    end

                end                            
            end   
            
            if best_thresholds ~= nil then
                local current_predicted_class, priority_classes = 'normal',{'non_urgent_finding','urgent','critical'}
                for i=1,#priority_classes do
                    if preds[priority_classes[i]][1] < preds[priority_classes[i]][2] then
                        current_predicted_class = priority_classes[i]
                    else
                        break
                    end
                end 
                local label_class = fromClassesToPriorityClass(imgs_infos[acc_numb].classes,priority_list,per_priority_mapping)               
                test_confusions['priority']:add(priority_list[current_predicted_class],priority_list[label_class])  

                local label,current_prediction = 2,2 
                if table.contains(imgs_infos[acc_numb].classes,'normal') then
                    label = 1
                end  
                if preds['normal'][1] > preds['normal'][2] then
                    current_prediction = 1    
                end
                test_confusions['labels']['normal']:add(current_prediction, label)
            end              
        elseif opt.classifier_type == "multi" then 
            
            if opt.ordinal_regression == 1 then

                local current_prediction_class = 'normal'
                local priority_classes = {'non_urgent_finding','urgent','critical'}

                for i=1,#priority_classes do
                    if preds[priority_classes[i]][1] < preds[priority_classes[i]][2] then
                        current_prediction_class = priority_classes[i]
                    else
                        break
                    end
                end
                local label_class = fromClassesToPriorityClass(imgs_infos[acc_numb].classes,priority_list,per_priority_mapping)               
                confusion:add(priority_list[current_prediction_class],priority_list[label_class])       
            else     
                local normal_flag, current_predicted_class = 1, 'normal'  
        
                for ii=1,#classes do   
                            
                    local sftmax_res = nn.SoftMax():forward(preds[classes[ii]])                    
                    local label = 2
                    if table.contains(imgs_infos[acc_numb].classes,classes[ii]) then
                        label = 1
                    end  
                                                                   
                    if best_thresholds ~= nil then
                        --we have best_thresholds, we do not have to calculate it
                        local current_prediction = 2 
                        if sftmax_res[1] > best_thresholds[classes[ii]] then
                            current_prediction = 1 
                            normal_flag = 2                      
                        end 
                            
                        if opt.classify_per_priority == 1 or opt.test_per_priority == 1  then

                            if current_prediction == 1 and priority_list[per_priority_mapping[classes[ii]]] > priority_list[current_predicted_class] then
                                current_predicted_class = per_priority_mapping[classes[ii]]   
                            end    
                        else  
                            test_confusions[classes[ii]]:add(current_prediction, label) 
                        end                
                       
                    else 
                        
                        for jj=1,#thresholds do                   
                            local current_prediction = 2 
                            if sftmax_res[1] > thresholds[jj] then
                                current_prediction = 1
                            end
                            local_confusions[classes[ii]][thresholds[jj]]:add(current_prediction, label)
                        end

                    end
                                        
                end
                
                if opt.classify_per_priority == 1 or opt.test_per_priority == 1  then
                    test_confusions:add(priority_list[current_predicted_class],priority_list[fromClassesToPriorityClass(imgs_infos[acc_numb].classes,priority_list,per_priority_mapping)])                
                elseif best_thresholds ~= nil then
                    local label = 2
                    if table.contains(imgs_infos[acc_numb].classes,"normal") and #imgs_infos[acc_numb].classes == 1  then
                        label = 1
                    end                 
                    test_confusions["normal"]:add(normal_flag, label)
                end    
                    
            end                    
            
        elseif opt.classifier_type == "single" then
            --TODO: fix here
            local label = nil        
            confusion:add(pred,label)
        end
    end
    
    if best_thresholds == nil then            
        best_thresholds = tds.Hash()

        for ii =1,#classes do

            local best_F1,best_threshold = -1,-1
            for jj=1,#thresholds do
                local acc, f1score, matthew, aacc,f_measure  = getMeasures(local_confusions[classes[ii]][thresholds[jj]])  
                if f1score > best_F1 then
                    best_F1 = f1score
                    best_threshold = thresholds[jj]
                end   
                if opt.verbose >= 2 then print(thresholds[jj],f1score,classes[ii]) end  
            end
            if best_threshold == -1 or best_F1 == 0 then
                best_threshold = 0.5    
            end
            best_thresholds[classes[ii]] = best_threshold
            if opt.verbose >= 2 then print(best_threshold,classes[ii],"\n-----------\n") end

        end
        local_confusions = nil
        return updateConfMatrices(net_predictions,imgs_infos,classes)      
    end    
end



-- used to store the results of the network
local net_predictions = tds.Hash()


local function testBatch(inputsCPU, labelsCPU,acc_numbs)
    cutorch.synchronize()
    collectgarbage()
    timer:reset()

    local pred = nil
    if opt.fixed_size_input == 1 then

        if covariate_variables_dim > 0 or opt.path_CH_embeddings ~= "" then
            inputs:resize(opt.batchSize,inputsCPU[1]:size(2),inputsCPU[1]:size(3),inputsCPU[1]:size(4))        
            inputs[{{1,inputsCPU[1]:size(1)}}]:copy(inputsCPU[1])

            covariate_inputs[{{1,inputsCPU[1]:size(1)}}]:copy(inputsCPU[2])

            pred = model:forward({inputs,covariate_inputs})[{{1,inputsCPU[1]:size(1)}}]:float()
        else
            inputs:resize(opt.batchSize,inputsCPU:size(2),inputsCPU:size(3),inputsCPU:size(4))  
            inputs[{{1,inputsCPU:size(1)}}]:copy(inputsCPU)
            pred = model:forward(inputs)[{{1,inputsCPU:size(1)}}]:float()
        end    
        
    else
        if opt.classifier_type == "multi" then
            pred = torch.FloatTensor(#inputsCPU,2*#classes)
        else
            pred = torch.FloatTensor(#inputsCPU,#classes)
        end
        for i=1,#inputsCPU do
            inputs:resize(1,inputsCPU[i]:size(1),inputsCPU[i]:size(2),inputsCPU[i]:size(3))
            inputs:copy(inputsCPU[i])
            pred[i]:copy(model:forward(inputs):max(3):max(4)) --sum(3):sum(4))
        end
        
    end 
    

    for i =1,#acc_numbs do
        if opt.multi_task then
            net_predictions[acc_numbs[i]] = splitPredictions(pred[i],classes,"multi_task")
        elseif opt.ordinal_regression == 1 then
            net_predictions[acc_numbs[i]] = splitPredictions(pred[i],classes,"ordinal_regression")--pred[i]:clone()   
        else
            net_predictions[acc_numbs[i]] = splitPredictions(pred[i],classes)
        end
    end

       
    cutorch.synchronize()
    batchNumber = batchNumber + 1
    if opt.verbose >= 0 then 
        print((">Time to forward/backward data through the network = %.2f seconds [%d/%d]<"):format(timer:time().real,batchNumber,total_batches))
    end 
     
end



function test(imgs_paths_and_infos)
    --define current classes
    local all_classes
    if opt.classifier_type == "multi" and opt.ordinal_regression == 0 then
    
        all_classes = table.copy(classes)
        table.insert(all_classes,'normal')

    elseif opt.classifier_type == "single" or opt.ordinal_regression == 1 then
           
        if opt.ordinal_regression == 1 then
            all_classes = opt.classes        
        else
            all_classes = classes
        end
        
    end
    net_predictions = tds.Hash()
    local avg_F1score = 0

    print('==> testing on test set:')

    cutorch.synchronize()
    model:evaluate()

    batchNumber = 0
    
    do -- start K datathreads (donkeys)
            local thread_info = {
                opt = opt,
                params = params,
                imgs_paths_and_infos = imgs_paths_and_infos,
                classes = classes,
                aug_flag = false,
                covariate_variables_dim = covariate_variables_dim,
                priority_list = priority_list,
                per_priority_mapping = per_priority_mapping,
                max_CH_len = max_CH_len,
                CH_emb_size =  CH_emb_size                          
            }
            data_load_thread = Threads(
                data_loading_threads,
                function()
                    require 'torch'
                    require 'utils'
                    require 'preprocess'
                    require 'data_loading_utils'
                    require 'batchLoading'
                end,
                function(idx)
                    info = thread_info -- pass to all donkeys via upvalue
                    local tid = idx
                    local seed = thread_info.opt.seed + idx
                    torch.setnumthreads(1)
                    if thread_info.opt.verbose >= 0 then print(string.format('Starting thread with id: %d seed: %d', tid, seed)) end
                end
          );

    end
    
    if opt.testMode ~= '' then
        opt.test_epoch_size = #imgs_paths_and_infos
    end
    total_images = math.min(#imgs_paths_and_infos,opt.test_epoch_size)
    total_batches = (total_images-(total_images%opt.batchSize))/opt.batchSize + 1
    
    print("Number of input images: " .. #imgs_paths_and_infos)
    print("Number of images really used: " .. total_images)
        
    for jj=1,total_images,opt.batchSize do
        --for ii=1,data_loading_jobs do
            data_load_thread:addjob(
                 -- the job callback (runs in data-worker thread)
                 function()
                    local inputs, labels, ids = loadBatch_test(jj)
                    return inputs, labels, ids
                 end,
                 -- the end callback (runs in the main thread)
                 testBatch
              )     
    end 
        
    data_load_thread:synchronize()
    cutorch.synchronize() 
    data_load_thread:terminate()
    
 
    updateConfMatrices(net_predictions,test_data_infos,classes)

    local measures = {}
    
    if opt.multi_task then
        if opt.testMode == '' then

            measures = getAvgMeasures(test_confusions['labels'],all_classes)           
            for jj=1,#all_classes do
                local info_to_log
                info_to_log = logInfos(test_confusions['labels'][all_classes[jj]])                
                testLogger[all_classes[jj]]:add(info_to_log)
                test_confusions['labels'][all_classes[jj]]:zero()
            end   
            local priority_measures = {}
            priority_measures["acc"],priority_measures["f1score"],priority_measures["matthew"],priority_measures["aacc"],priority_measures["f_measure"] = getMeasures_SingleClassifier(test_confusions['priority'])
            testLogger['priority']:add(priority_measures) 
            print(test_confusions['priority'])
            test_confusions['priority']:zero()      
        else
            
--            measures["acc"],measures["f1score"],measures["matthew"],measures["aacc"],measures["f_measure"] = getMeasures_SingleClassifier(test_confusions['priority'])
            print(test_confusions['priority'])
            measures = getAvgMeasures(test_confusions['labels'],all_classes)
     
        end
    elseif opt.classifier_type == "multi" and opt.ordinal_regression == 0 then

        print("\n>CNN test performance:")
        
        if opt.testMode == '' then 
            if #all_classes == 2 and table.contains(all_classes,'normal') then 
                measures = getAvgMeasures(test_confusions,table.find_and_remove(all_classes,'normal'))
            else
                measures = getAvgMeasures(test_confusions,all_classes)
            end           
            for jj=1,#all_classes do
                local info_to_log

                info_to_log = logInfos(test_confusions[all_classes[jj]]) 
                test_confusions[all_classes[jj]]:zero()
                                
                testLogger[all_classes[jj]]:add(info_to_log)
            end 
        else
            if opt.classify_per_priority == 1 or opt.test_per_priority == 1 then
                measures["acc"],measures["f1score"],measures["matthew"],measures["aacc"],measures["f_measure"] = getMeasures_SingleClassifier(test_confusions)
                print(test_confusions)
            else
                if #all_classes == 2 and table.contains(all_classes,'normal') then 
                    measures = getAvgMeasures(test_confusions,table.find_and_remove(all_classes,'normal'))
                else
                    measures = getAvgMeasures(test_confusions,all_classes)
                end              
            end
        end
        
        if opt.predict_age then
            print("Age error: " .. predict_age_error/num_testing_samples)
            if #all_classes == 0 then 
--                testLogger:add{['% Age error: (test set)'] = predict_age_error/num_testing_samples * 120}
                totalValid = 1 - (predict_age_error/num_testing_samples)
            end
        end
        
    elseif opt.classifier_type == "single" or opt.ordinal_regression == 1 then
        measures["acc"],measures["f1score"],measures["matthew"],measures["aacc"],measures["f_measure"] = getMeasures_SingleClassifier(confusion)
        print("Accuracy = " .. measures["acc"] .. "\nF1 score = " .. measures["f1score"] .. "\nmatthew correlation = " .. measures["matthew"] .. "\navg accuracy = " .. measures["aacc"] .. "\nf measure = " .. measures["f_measure"])
        print(confusion)
        
        if opt.testMode == '' then
            testLogger:add(measures) 
        end       
    end 

    if opt.testMode == '' then
        avg_F1score = measures["f1score"]
        if avg_F1score > best_F1score_so_far then
            local filename = paths.concat(opt.path_save, 'best_F1score_model.net')
            os.execute('mkdir -p ' .. sys.dirname(filename))
            print('==> saving model to '..filename)
            saveDataParallel(filename, model)
           
            best_F1score_so_far = avg_F1score
        
            -- Save perfomances on file
            
            if opt.multi_task or opt.classifier_type == "multi" and opt.ordinal_regression == 0 then
                for jj=1,#all_classes do   
                    if opt.multi_task then
                        save_perfomances(opt.path_save .. '/perfomances_' .. all_classes[jj] .. '.txt', test_confusions['labels'][all_classes[jj]],all_classes[jj])
                    else  
                        save_perfomances(opt.path_save .. '/perfomances_' .. all_classes[jj] .. '.txt', test_confusions[all_classes[jj]],all_classes[jj])
                    end
                end 
                torch.save(opt.path_save .. "/best_F1Score_thresholds.t7",best_thresholds)
            end                         
        end
        
        --reset inputs in order to do not waste any CUDA memory during the next train epoch
        inputs = createCudaTensor()
    else  
        -- Save perfomances on file
        if opt.multi_task or (opt.classifier_type == "multi" and opt.ordinal_regression == 0 
            and opt.classify_per_priority == 0 and opt.test_per_priority == 0) then
                for jj=1,#opt.classes do
                    if opt.multi_task then
                        save_perfomances(opt.path_test_results .. '/perfomances_' .. opt.classes[jj] .. '.txt', test_confusions['labels'][opt.classes[jj]],opt.classes[jj])
                    else
                        save_perfomances(opt.path_test_results .. '/perfomances_' .. opt.classes[jj] .. '.txt', test_confusions[opt.classes[jj]],opt.classes[jj])
                    end
                end   
        end 
       
        torch.save(opt.path_test_results .."net_predictions.t7",net_predictions)
    end
    
    if opt.classifier_type == "multi" and opt.ordinal_regression == 0 then
        best_thresholds = nil
    elseif opt.classifier_type == "single" or opt.ordinal_regression == 1 then
        confusion:zero()
    end
    if opt.predict_age then
        predict_age_error = 0
    end
    return avg_F1score
end
