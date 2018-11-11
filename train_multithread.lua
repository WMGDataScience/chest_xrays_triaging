----------------------------------------------------------------------
-- This script define the training procedure
-- 
-- multi-thread for data loading version
----------------------------------------------------------------------

require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'utils'
require 'batchLoading'
tds = require 'tds'
----------------------------------------------------------------------

local data_loading_threads = opt.dataLoadingThreads


----------------------------
-- Init loggers and confusion matrices
----------------------------
if opt.verbose >= 2 then print('Init loggers and confusion matrices') end

if opt.multi_task then
    -- This matrix records the current confusion across classes   
    trainLogger,testLogger = {},{}
    confusions = {}
    confusions['labels'] = {}   
    for i=1,#opt.classes do
        confusions['labels'][opt.classes[i]] = optim.ConfusionMatrix({opt.classes[i],"not_" .. opt.classes[i]})    
        trainLogger[opt.classes[i]] = optim.Logger(paths.concat(opt.path_save, opt.classes[i] .. '_train.log'))
        testLogger[opt.classes[i]] = optim.Logger(paths.concat(opt.path_save,opt.classes[i] .. '_val.log'))
    end
 
    confusions['priority'] = optim.ConfusionMatrix({"normal","non_urgent_finding","urgent","critical"})
    trainLogger['priority'] = optim.Logger(paths.concat(opt.path_save, 'priority_train.log'))
    testLogger['priority'] = optim.Logger(paths.concat(opt.path_save, 'priority_val.log'))

elseif opt.classifier_type == "multi" and opt.ordinal_regression == 0 then
    -- This matrix records the current confusion across classes   
    trainLogger,testLogger = {},{}
    confusions = {}
       
    for i=1,#opt.classes do
        confusions[opt.classes[i]] = optim.ConfusionMatrix({opt.classes[i],"not_" .. opt.classes[i]})    
        trainLogger[opt.classes[i]] = optim.Logger(paths.concat(opt.path_save, opt.classes[i] .. '_train.log'))
        testLogger[opt.classes[i]] = optim.Logger(paths.concat(opt.path_save,opt.classes[i] .. '_val.log'))
    end
    
elseif opt.classifier_type == "single" or opt.ordinal_regression == 1 then
    
    trainLogger = optim.Logger(paths.concat(opt.path_save, 'train.log'))
    testLogger = optim.Logger(paths.concat(opt.path_save, 'val.log'))
    
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
-- Declare train variables
----------------------------
if opt.verbose >= 2 then print('Declaring train variables...') end 

-- GPU inputs (preallocate)
local inputs = createCudaTensor() 
local labels = createCudaTensor()

local timer = torch.Timer()
local dataLoadingTimer = torch.Timer()

local parameters, gradParameters
local batchNumber, total_batches = 0,nil


if covariate_variables_dim > 0 then
    covariate_inputs = createCudaTensor(torch.LongStorage({opt.batchSize,covariate_variables_dim}))
end
if opt.path_CH_embeddings ~= "" then
    covariate_inputs = createCudaTensor(torch.LongStorage({opt.batchSize,2,max_CH_len,CH_emb_size}))
end

if opt.verbose >= 2 then print('Import threading library...') end

local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')


-----------------------------------------------------------------------------------
-- trainBatch - Used by train() to train a single batch after the data is loaded.
-----------------------------------------------------------------------------------
local function trainBatch(inputsCPU, labelsCPU, gl_labelsCPU)
    
    local data_loading_time = 0
    cutorch.synchronize()
    collectgarbage()
    timer:reset()
    
    if opt.verbose >= 1 then dataLoadingTimer:reset() end

    -- load new sample
    if covariate_variables_dim > 0 or opt.path_CH_embeddings ~= "" then
        inputs:resize(opt.batchSize,inputsCPU[1]:size(2),inputsCPU[1]:size(3),inputsCPU[1]:size(4))
        inputs:copy(inputsCPU[1])
        covariate_inputs:copy(inputsCPU[2])
    else
        inputs:resize(opt.batchSize,inputsCPU:size(2),inputsCPU:size(3),inputsCPU:size(4))  
        inputs:copy(inputsCPU)
    end
    
    if opt.classifier_type == "multi" then 
        labels:resize(opt.batchSize,labelsCPU:size(2))
    elseif opt.classifier_type == "single" then 
        labels:resize(opt.batchSize)    
    end  
            
    labels:copy(labelsCPU)

    if opt.verbose >= 1 then data_loading_time = dataLoadingTimer:time().real end
        
    local err = 0
    feval = function(x)
        -- TODO: init cleaning_network in the proper way
        model:zeroGradParameters()

        local outputs = nil
        if covariate_variables_dim > 0 or opt.path_CH_embeddings ~= "" then
            outputs = model:forward({inputs,covariate_inputs})
        else
            outputs = model:forward(inputs)
        end
        
        if opt.multi_task then
            local noutputs = 2 if opt.loss == "smc" then noutputs = 1  end
            local gradOutputs = createCudaTensor(outputs:size())
            local labels_err = 0
            for i=1,#classes do--#criterions do
                labels_err = labels_err + criterions['labels'][i]:forward(outputs[{{},{(i-1)*noutputs+1,i*noutputs}}],labels[{{},i}]) 
                gradOutputs[{{},{(i-1)*noutputs+1,i*noutputs}}]:copy(criterions['labels'][i]:backward(outputs[{{},{(i-1)*noutputs+1,i*noutputs}}], labels[{{},i}]))                                      
            end           
            local priorities, priorities_err = {"non_urgent_finding","urgent","critical"}, 0
            for j=1,#priorities do--#criterions do
                local i = j + #classes
                priorities_err = priorities_err + criterions['priority'][j]:forward(outputs[{{},{(i-1)*noutputs+1,i*noutputs}}],labels[{{},i}]) 
                gradOutputs[{{},{(i-1)*noutputs+1,i*noutputs}}]:copy(criterions['priority'][j]:backward(outputs[{{},{(i-1)*noutputs+1,i*noutputs}}], labels[{{},i}]))                                      
            end            
            local abnormal_err = criterions['normal']:forward(outputs[{{},{(#classes+#priorities)*noutputs+1,(#classes+#priorities+1)*noutputs}}],labels[{{},#classes+#priorities}]) 
            gradOutputs[{{},{(#classes+#priorities)*noutputs+1,(#classes+#priorities+1)*noutputs}}]:copy(criterions['normal']:backward(outputs[{{},{(#classes+#priorities)*noutputs+1,(#classes+#priorities+1)*noutputs}}], labels[{{},#classes+#priorities+1}]))                                      
            
            --err = (abnormal_err + labels_err + priorities_err) / (#classes+#priorities+1)
            err = (abnormal_err + (labels_err / #classes) + (priorities_err/#priorities)) / 3
            
            if covariate_variables_dim > 0 or opt.path_CH_embeddings ~= "" then
                model:backward({inputs,covariate_inputs}, gradOutputs)
            else
                model:backward(inputs, gradOutputs)
            end

            for jj=1,#classes do  
                confusions['labels'][classes[jj]]:batchAdd(outputs[{{},{(jj-1)*noutputs+1,jj*noutputs}}],labelsCPU[{{},jj}])
            end

            local pred_class = fromVecToClass(outputs[{{},{(#classes)*noutputs+1,(#classes+#priorities)*noutputs}}],priority_list)
            local label_class = fromVecToClass(labelsCPU[{{},{#classes+1,#classes+#priorities}}],priority_list)
            confusions['priority']:batchAdd(pred_class,label_class)                
            confusions['labels']["normal"]:batchAdd(outputs[{{},{(#classes+#priorities)*noutputs+1,(#classes+#priorities+1)*noutputs}}],labelsCPU[{{},#classes+#priorities+1}])
                     
        elseif opt.classifier_type == "multi" then 
            local noutputs = 2 if opt.loss == "smc" then noutputs = 1  end
            local gradOutputs = createCudaTensor(outputs:size())
            for i=1,#classes do--#criterions do
                err = err + criterions[i]:forward(outputs[{{},{(i-1)*noutputs+1,i*noutputs}}],labels[{{},i}]) 
                gradOutputs[{{},{(i-1)*noutputs+1,i*noutputs}}]:copy(criterions[i]:backward(outputs[{{},{(i-1)*noutputs+1,i*noutputs}}], labels[{{},i}]))                                      
            end
            err = err / #classes
            
            if opt.predict_age then
                err = err + age_criterion:forward(outputs[{{},{outputs:size(2)}}],labels[{{},labels:size(2)}]) 
                gradOutputs[{{},{gradOutputs:size(2)}}]:copy(age_criterion:backward(outputs[{{},{outputs:size(2)}}],labels[{{},labels:size(2)}]) )
            end  
            
            if covariate_variables_dim > 0 or opt.path_CH_embeddings ~= "" then
                model:backward({inputs,covariate_inputs}, gradOutputs)
            else
                model:backward(inputs, gradOutputs)
            end       
            
            --Update training confusion matrix(es)    
            if opt.ordinal_regression == 1 then
                local pred_class = fromVecToClass(outputs,priority_list)
                local label_class = fromVecToClass(labelsCPU,priority_list)
                confusion:batchAdd(pred_class,label_class)    
            else
                for jj=1,#classes do  
                    confusions[classes[jj]]:batchAdd(outputs[{{},{(jj-1)*noutputs+1,jj*noutputs}}],labelsCPU[{{},jj}])
                end
    
                local pred_normal = torch.FloatTensor(labelsCPU:size(1)):fill(1)
                local label_normal = torch.FloatTensor(labelsCPU:size(1)):fill(1)            
                for i =1,labels:size(1) do
    
                    for j=1,labelsCPU:size(2) do 
                        local max,idx = torch.max(outputs[{i,{(j-1)*noutputs+1,j*noutputs}}],1)
        --                print(labels[i][j] .. " - pred: " .. idx[1])
                        if idx[1] ~= 2 then
                            pred_normal[i] = 2   
                        end
                        if labelsCPU[i][j] ~= 2 then
                            label_normal[i] = 2
                        end
                        --TODO: change labels to -1 +1
                    end
                    
                end
                confusions["normal"]:batchAdd(pred_normal,label_normal)
              end
        elseif opt.classifier_type == "single" then
            err = criterion:forward(outputs,labels)
            local gradOutputs = criterion:backward(outputs,labels)
            model:backward(inputs, gradOutputs)        
            confusion:batchAdd(outputs,labels)     
        end     
           
--        -- clip gradient element-wise
--        gradParameters:clamp(-5, 5)
        
        return err, gradParameters
    end
    -- optimize on current mini-batch
    if optimMethod == optim.asgd then
        _,_,average = optimMethod(feval, parameters, optimState)
    else
        optimMethod(feval, parameters, optimState)
    end 

--[[    
    if model.needsSync then
        model:syncParameters()
    end
]]-- Useful? 
    cutorch.synchronize()
    batchNumber = batchNumber + 1
    if opt.verbose >= 0 then 
        print((">[%d/%d] fwd/bwd current mini-batch time = %.2f seconds. Err = %.4f<"):format(batchNumber,total_batches,timer:time().real,err))
    end
    if opt.verbose >= 1 then
        print((">Time for batch copy from CPU to GPU = %.4f seconds [%d/%d]<"):format(data_loading_time,batchNumber,total_batches))
    end    
end



function train(imgs_paths_and_infos)
    epoch = epoch or 1    
    collectgarbage()
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)
    
    --define current classes
    local all_classes
    if (opt.classifier_type == "multi" and opt.ordinal_regression == 0) or opt.multi_task then
    
        all_classes = table.copy(classes)
        if table.contains(all_classes,'normal') then
            classes = table.find_and_remove(classes,'normal')
        else
            table.insert(all_classes,'normal')
        end
    elseif opt.classifier_type == "single" or opt.ordinal_regression == 1 then
           
        if opt.ordinal_regression == 1 then
            all_classes = opt.classes        
        else
            all_classes = classes
        end
        
    end

    parameters, gradParameters = model:getParameters()
    local avg_F1score = 0
    batchNumber = 0
    
    cutorch.synchronize()
    model:training()
    
    if opt.optimization ~= "adadelta" then
        local decay = math.floor((epoch - 1) / 30)
        optimState.learningRate = opt.learningRate * math.pow(0.1, decay)
    end
    
    
    local num_images_sum,shuffle,loadBatch = 0,nil,nil

    if opt.balance_classes == 1 then
        shuffle = {}
        for k,v in pairs(imgs_paths_and_infos) do
            print(k .. " : " .. #v)   
            if #v == 0 then
                imgs_paths_and_infos[k] = nil
            else         
                shuffle[k] = torch.randperm(#v) --torch.range(1,#v) -- 
                num_images_sum = num_images_sum + #v
            end 
        end
        loadBatch = loadUniformlyOverClasses
    else
        num_images_sum = #imgs_paths_and_infos
        loadBatch = loadRandomlyOverClasses
        shuffle = torch.randperm(#imgs_paths_and_infos)
    end
    
    local total_images = math.min(num_images_sum,opt.epoch_size)
    local num_training_samples = total_images - (total_images % opt.batchSize)
    total_batches = (total_images-(total_images%opt.batchSize))/opt.batchSize

    
    print("Number of input images: " .. num_images_sum)
    print("Number of images really used: " .. num_training_samples)

    local augment = false
    if opt.augment_train_data == 1 then
        augment = true
    end
            
    do -- start K datathreads (donkeys)

            local thread_info = {
                opt = opt,
                params = params,
                imgs_paths_and_infos = imgs_paths_and_infos,
                shuffle = shuffle,             
                classes = classes,
                aug_flag = augment,
                gl_infos = gl_train_infos,
                covariate_variables_dim = covariate_variables_dim,
                to_not_flip_classes = to_not_flip_classes,
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
                    tds = require 'tds'
                    require 'batchLoading'
                end,
                function(idx)
                    info = tds.Hash(thread_info) -- pass to all threads via upvalue
                    local tid = idx
                    local seed = thread_info.opt.seed + idx
                    torch.setnumthreads(1)
                    if thread_info.opt.verbose >= 0 then print(string.format('Starting thread with id: %d seed: %d', tid, seed)) end
                end
          );

    end    
    

    for jj=1,num_training_samples,opt.batchSize do
        
            data_load_thread:addjob(
                 -- the job callback (runs in data-worker thread)
                 function()
                    local inputs, labels = loadBatch(jj)
                    return inputs, labels
                 end,
                 -- the end callback (runs in the main thread)
                 trainBatch
              )        

    end        
        
    data_load_thread:synchronize()
    cutorch.synchronize() 
 
    data_load_thread:terminate()
    
    if opt.multi_task then
        local avg_measures = getAvgMeasures(confusions['labels'],all_classes)
        local info_to_log = logInfos(confusions['labels'])

        for i=1,#all_classes do
            trainLogger[all_classes[i]]:add(info_to_log[all_classes[i]])
            confusions['labels'][all_classes[i]]:zero()           
        end
        avg_F1score = avg_measures["f1score"]

        local measures = {}
        measures["acc"],measures["f1score"],measures["matthew"],measures["aacc"],measures["f_measure"] = getMeasures_SingleClassifier(confusions['priority'])
        print(confusions['priority'])
        trainLogger['priority']:add(measures)
        print("Priority prediction Measures: Accuracy = " .. measures["acc"] .. "\nF1 score = " .. measures["f1score"] .. "\nmatthew correlation = " .. measures["matthew"] .. "\navg accuracy = " .. measures["aacc"] .. "\nf measure = " .. measures["f_measure"])
        confusions['priority']:zero()                 
    elseif opt.classifier_type == "multi" and opt.ordinal_regression == 0 then        
        -- print confusion matrix
        print("\n>CNN train performance:")
        local avg_measures = getAvgMeasures(confusions,all_classes)

        -- update logger
        local info_to_log = logInfos(confusions)
        
        for i=1,#all_classes do
            trainLogger[all_classes[i]]:add(info_to_log[all_classes[i]])
            confusions[all_classes[i]]:zero()           
        end
        avg_F1score = avg_measures["f1score"]
    elseif opt.classifier_type == "single" or opt.ordinal_regression == 1 then
        -- print confusion matrix
        local measures = {}
        measures["acc"],measures["f1score"],measures["matthew"],measures["aacc"],measures["f_measure"] = getMeasures_SingleClassifier(confusion)
        print(confusion)
        
        -- update logger
        trainLogger:add(measures)
--        trainLogger:add{['% mean class f1 score (train set)'] = f1score} 
        print("Accuracy = " .. measures["acc"] .. "\nF1 score = " .. measures["f1score"] .. "\nmatthew correlation = " .. measures["matthew"] .. "\navg accuracy = " .. measures["aacc"] .. "\nf measure = " .. measures["f_measure"])
        confusion:zero() 
        avg_F1score = measures["f1score"]     
    end    
        
    collectgarbage()    

    torch.save(opt.path_save .. "/optimState.t7", optimState)
    saveDataParallel(opt.path_save .. '/last_epoch_model.net', model) 
    print('==> saving model to '..opt.path_save .. '/last_epoch_model.net')        
    -- next epoch
    epoch = epoch + 1
    
    return avg_F1score   
end
