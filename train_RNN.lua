require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'utils'
tds = require 'tds'

if opt.verbose >= 2 then print('Import threading library...') end

local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

if opt.verbose >= 1 then print('Declaring train variables...') end 

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataLoadingTimer = torch.Timer()

if opt.verbose > 2 then print('Call model:getParameters()') end
local parameters, gradParameters = model:getParameters()

local batchNumber = 0
local total_batches

if opt.verbose >= 2 then print('Init loggers and confusion matrices') end

confusions = {}
trainLogger = {}
testLogger = {}

for i=1,#opt.classes do
    confusions[opt.classes[i]] = optim.ConfusionMatrix({opt.classes[i],"not_" .. opt.classes[i]})
    trainLogger[opt.classes[i]] = optim.Logger(paths.concat(opt.path_save, opt.classes[i] .. '_train.log'))
    testLogger[opt.classes[i]] = optim.Logger(paths.concat(opt.path_save, opt.classes[i] .. '_val.log'))
end

local function loadBatchBalanced(index)
    local dataTimer = torch.Timer()
    dataTimer:reset()

    local imgs_batch = torch.FloatTensor(info.opt.batchSize,info.opt.input_size[1],info.opt.input_size[2],info.opt.input_size[3])
    local labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes)
    local input_classes = {}
    local class_i = 1
    for k,v in pairs(info.imgs_paths_and_infos) do
        input_classes[class_i] = k
        class_i = class_i + 1
    end

    local class_shuffle = torch.randperm(#input_classes) 
    
    local i = 1
    local k = 1
    while i <= info.opt.batchSize do
        for j=1,#input_classes do
            if i<=info.opt.batchSize then
                local random_class = input_classes[class_shuffle[j]]
                local current_elem = info.imgs_paths_and_infos[random_class][info.shuffle[random_class][(index+k-1)%#info.imgs_paths_and_infos[random_class] + 1]]

                imgs_batch[i]:copy(getImage(current_elem,info))
--                    image.save("test_" .. i .. ".jpg",imgs_batch[i])             
                labels_batch[i] = getLabelsSequence(current_elem.classes,info.classes)
--                    print(labels_batch[i])
                i=i+1
            end
        end
        k = k + 1
    end

    if info.opt.verbose >= 2 then print((">imgs loading [no preprocess] = %.2f seconds<"):format(dataTimer:time().real)) end     
    
    if info.opt.doPreprocess == true then
        preprocess(imgs_batch, info.params['mean'], info.params['var'])
    end

    if info.opt.verbose >= 1 then print((">Time to load data from disk = %.2f seconds<"):format(dataTimer:time().real)) end 

    return imgs_batch, labels_batch
end

local function loadGLBatch(index)
    local dataTimer = torch.Timer()
    dataTimer:reset()
    
    collectgarbage()
    local labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes)
    local noisy_labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes)
    local imgs_batch = torch.FloatTensor(info.opt.batchSize,info.opt.input_size[1],info.opt.input_size[2],info.opt.input_size[3])

    for i=1,info.opt.batchSize do

        local current_img_info = info.gl_infos[info.gl_infos_shuffle[(index+i-1)%#info.gl_infos + 1]]

        imgs_batch[i]:copy(getImage(current_img_info,info))
        labels_batch[i] = getLabelsSequence(current_img_info.classes,info.classes)
        noisy_labels_batch[i] = getLabelsSequence(current_img_info.noisy_classes,info.classes)
    end

    if info.opt.verbose >= 2 then print((">imgs loading [no preprocess] = %.2f seconds<"):format(dataTimer:time().real)) end     
    
    if info.opt.doPreprocess == true then
        preprocess(imgs_batch, info.params['mean'], info.params['var'])
    end

    if info.opt.verbose >= 1 then print((">Time to load data from disk = %.2f seconds<"):format(dataTimer:time().real)) end 

    return imgs_batch,noisy_labels_batch, labels_batch          

    
end

local function loadBatch(index)
    local dataTimer = torch.Timer()
    dataTimer:reset()
    
    local imgs_batch = torch.FloatTensor(info.opt.batchSize,info.opt.input_size[1],info.opt.input_size[2],info.opt.input_size[3])
    local labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes)

    for i = 1,info.opt.batchSize do   
        local current_elem = info.imgs_paths_and_infos[info.shuffle[index + i - 1]]
        imgs_batch[i]:copy(getImage(current_elem,info))
        labels_batch[i] = getLabelsSequence(current_elem.classes,info.classes)
        
--        local label_string = "" .. i
--        for j=1,#current_elem.classes do
--            label_string = label_string .. "|" .. current_elem.classes[j]
--        end
--        image.save(label_string .. ".jpg",imgs_batch[i])
--        print(labels_batch[i])

    end

    if info.opt.verbose >= 2 then print((">imgs loading [no preprocess] = %.2f seconds<"):format(dataTimer:time().real)) end     
    
    if info.opt.doPreprocess == true then
        preprocess(imgs_batch, info.params['mean'], info.params['var'])
    end

    if info.opt.verbose >= 1 then print((">Time to load data from disk = %.2f seconds<"):format(dataTimer:time().real)) end 

    return imgs_batch, labels_batch
end

local total_correct = 0

local function testBatch(inputsCPU, labelsCPU)
    local data_loading_time = 0
    cutorch.synchronize()
    collectgarbage()
    timer:reset()
    
    inputs:resize(opt.batchSize,inputsCPU:size()[2],inputsCPU:size()[3],inputsCPU:size()[4])
    labels:resize(opt.batchSize,#classes) 
    
    if opt.verbose >= 1 then dataLoadingTimer:reset() end 
    
    -- load new sample
    inputs:copy(inputsCPU)
    labels:copy(labelsCPU)
    

    local outputs = model:forward(inputs)
    
   
    for i = 1,labels:size(1) do
        local error_flag = 0
        for j=1,labels:size(2) do
            local max,idx = torch.max(outputs[i][j],1)
            if idx[1] ~= labels[i][j] then
                error_flag = 1
                break    
            end
            
        end

        updateConfusions(outputs[i],labels[i],confusions,classes)
        if error_flag == 0 then
            total_correct = total_correct + 1    
        end
    end 
    
    
    cutorch.synchronize()
    batchNumber = batchNumber + 1
    if opt.verbose >= 0 then 
        print((">Time to forward/backward data through the network = %.2f seconds [%d/%d]<"):format(timer:time().real,batchNumber,total_batches))
    end
    if opt.verbose >= 1 then
        print((">Time for batch copy from CPU to GPU = %.4f seconds [%d/%d]<"):format(data_loading_time,batchNumber,total_batches))
    end  
           
end

local function labelContains(label_seq,pred)
    for i=1,label_seq:size(1) do
        if label_seq[i] == pred then
            return i
        end
    end
    return -1
end

local function getLabelsFromOutput(label,output)
    local error_flag = 0
    local label_index = 1
    local out_index = 1
    local new_label_index = 1
    local new_label = torch.FloatTensor(output:size(1)):fill(label:size(1) + 1)
    
    local tensor_stop_class = torch.FloatTensor(label:size(1) + 1):fill(0)
--    local first_wrong_label = -1
    tensor_stop_class[label:size(1) + 1] = 1
    torch.log(tensor_stop_class,tensor_stop_class)    
    
    local flag_stop_signal = 0 
    for i =1,output:size(1) do
        local max,idx = torch.max(output[i],1)

        if idx[1] ~= label:size(1) + 1  then
            local label_index = labelContains(label,idx[1])
            if error_flag == 0 and label_index == -1 then
                error_flag = 1
    --            first_wrong_label = label[i]
                new_label[i] = label[i]
            elseif label_index ~= -1 then
                new_label[i] = label[label_index]
                label[label_index] = label[i]            
            elseif error_flag == 1 then
                new_label[i] = label[i]                    
            end
        else
            new_label[i] = label[i]        
        end
        
        if label[i] == label:size(1) + 1 then 
            if flag_stop_signal == 1 then
                output[i]:copy(tensor_stop_class)
            elseif flag_stop_signal == 0 then 
                flag_stop_signal = 1
            end   
        end     
    end
    
    
    return new_label,output,error_flag
end

if opt.label_cleaning_network == 1 then
    cl_net_parameters, cl_net_gradParameters = cleaning_network:getParameters()   
end

   

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
local function trainBatch(inputsCPU, labelsCPU)
    local data_loading_time = 0
    cutorch.synchronize()
    collectgarbage()
    timer:reset()
    
    inputs:resize(opt.batchSize,inputsCPU:size()[2],inputsCPU:size()[3],inputsCPU:size()[4])
    labels:resize(opt.batchSize,#classes)    
    
    if opt.verbose >= 1 then dataLoadingTimer:reset() end

    -- load new sample
    inputs:copy(inputsCPU)
    labels:copy(labelsCPU)
    
    local tensor_stop_class = torch.FloatTensor(#classes + 1):fill(0)
    tensor_stop_class[#classes + 1] = 1
    torch.log(tensor_stop_class,tensor_stop_class)
    
    if opt.verbose >= 1 then data_loading_time = dataLoadingTimer:time().real end
        
    local err, outputs
    feval = function(x)
    
        model:zeroGradParameters()
       
--        for j=1,labels:size(1) do
--            local label_string = ""
--            for i=1,labels:size(2) do
--                label_string = label_string .. "|" .. labels[j][i]
--            end
--            image.save(label_string .. ".jpg",inputs[j])
--        end              
--        print(inputs:size())
        outputs = model:forward(inputs)

        for i =1,outputs:size(1) do


            updateConfusions(outputs[i],labels[i],confusions,classes)
            
            local error_flag = 0
--            labels[i],outputs[i],error_flag = getLabelsFromOutput(labels[i],outputs[i])          

            
            local flag_stop_signal = 0
            for j=1,outputs:size(2) do
                local max,idx = torch.max(outputs[i][j],1)
                if error_flag == 0 and idx[1] ~= labels[i][j] then
                    error_flag = 1   
                end
                if labels[i][j] == #classes + 1 and flag_stop_signal == 1 then
                    outputs[i][j]:copy(tensor_stop_class)
                elseif labels[i][j] == #classes + 1 and flag_stop_signal == 0 then 
                    flag_stop_signal = 1
                end
                if idx[1] == #classes + 1 and flag_stop_signal == 1 then
                    -- no error backpropagated
                    outputs[i][j]:copy(tensor_stop_class)
                    labels[i][j] = #classes + 1
                elseif idx[1] == #classes + 1 and flag_stop_signal == 0 then 
                    flag_stop_signal = 1
                end
            end
            if error_flag == 0 then
                total_correct = total_correct + 1    
            end
        end

        err = criterion:forward(outputs,labels)
        local gradOutputs = criterion:backward(outputs,labels)
        model:backward(inputs, gradOutputs)


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
    batchNumber = 0
    total_correct = 0
    
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    local all_classes = table.copy(classes)
    table.insert(all_classes,'normal')

    cutorch.synchronize()
    model:training()
    
    if epoch % 3 == 0 and opt.optimization ~= "adadelta" then
        optimState.learningRate = optimState.learningRate * 0.94
    end
    
    local shuffle = nil
    local num_images_sum = 0
    local load_function = loadBatch
    if opt.balance_classes == 1 then
        shuffle = {}
        for k,v in pairs(imgs_paths_and_infos) do
--            print(k .. " : " .. #v)        
            shuffle[k] = torch.randperm(#v)
            num_images_sum = num_images_sum + #v 
        end        
        load_function = loadBatchBalanced
    else
        shuffle = torch.randperm(#imgs_paths_and_infos)
        num_images_sum = #imgs_paths_and_infos          
    end
    
    local total_images = math.min(num_images_sum,opt.epoch_size)
    local num_training_samples = total_images - (total_images % opt.batchSize)
    total_batches = (total_images-(total_images%opt.batchSize))/opt.batchSize         

   
    print("Number of input images: " .. total_images)
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
                to_not_flip_classes = to_not_flip_classes        
            }
            data_load_thread = Threads(
                thread_info.opt.dataLoadingThreads,
                function()
                    require 'torch'
                    require 'utils'
                    require 'preprocess'
                    require 'data_loading_utils'
                    tds = require 'tds'
                end,
                function(idx)
                    info = tds.Hash(thread_info) -- pass to all threads via upvalue
                    local tid = idx
                    local seed = thread_info.opt.seed + idx
                    torch.setnumthreads(1)
--                    torch.manualSeed(seed)
                    if thread_info.opt.verbose >= 0 then print(string.format('Starting thread with id: %d seed: %d', tid, seed)) end
                end
          );

    end    

    for jj=1,num_training_samples,opt.batchSize do

        --for ii=1,data_loading_jobs do
            data_load_thread:addjob(
                 -- the job callback (runs in data-worker thread)
                 function()
                    local inputs, labels = load_function(jj)
                    return inputs, labels
                 end,
                 -- the end callback (runs in the main thread)
                 trainBatch
              )     
    end        
        
    data_load_thread:synchronize()
    cutorch.synchronize() 
 
    data_load_thread:terminate()   
        
    collectgarbage()    

    local avg_measures = getAvgMeasures(confusions,all_classes)
  
    print("Correct sequences = " .. total_correct .. " / " .. num_training_samples)
--    print("Avg. F1 score = " .. totalValid)
    
    torch.save(opt.path_save .. "/last_epoch_optimState.t7", optimState)
    saveDataParallel(opt.path_save .. '/last_epoch_model.net', model) 
    print('==> saving model to '..opt.path_save .. '/last_epoch_model.net')
    
    local info_to_log = logInfos(confusions)        
--    trainLogger:add{['% correct labeled imaged / total images (train set)'] = (total_correct / num_training_samples * 100)}  
--    trainLogger:add{['mean class F1 score (train set) = '] = avg_measures["f1score"] }    
  
    for i=1,#all_classes do
        info_to_log[all_classes[i]]["tot_correct_sequences"] = total_correct
        info_to_log[all_classes[i]]["tot_sequences"] = num_training_samples    
        trainLogger[all_classes[i]]:add(info_to_log[all_classes[i]]) 
        confusions[all_classes[i]]:zero()
    end    
    -- next epoch
    epoch = epoch + 1 
    
    return avg_measures["f1score"]
end

best_F1score_so_far = -1

function test(imgs_paths_and_infos)

    batchNumber = 0
    total_correct = 0
    
    print('==> test:')

    cutorch.synchronize()
    model:evaluate()
    
    local total_images = math.min(#imgs_paths_and_infos,opt.test_epoch_size)
    local num_samples = total_images - (total_images % opt.batchSize)
    total_batches = (total_images-(total_images%opt.batchSize))/opt.batchSize
 
    local all_classes = table.copy(classes)
    table.insert(all_classes,'normal')    
    
    local shuffle = torch.randperm(#imgs_paths_and_infos) --torch.range(1,#imgs_paths_and_infos)
   
    print("Number of input images: " .. #imgs_paths_and_infos)
    print("Number of images really used: " .. num_samples)
       
    do -- start K datathreads (donkeys)

            local thread_info = {
                opt = opt,
                params = params,
                imgs_paths_and_infos = imgs_paths_and_infos,
                shuffle = shuffle,             
                classes = classes,
                aug_flag = false
            }
            data_load_thread = Threads(
                thread_info.opt.dataLoadingThreads,
                function()
                    require 'torch'
                    require 'utils'
                    require 'preprocess'
                    require 'data_loading_utils'
                    tds = require 'tds'
                end,
                function(idx)
                    info = tds.Hash(thread_info) -- pass to all threads via upvalue 
                    local tid = idx
                    local seed = thread_info.opt.seed + idx
                    torch.setnumthreads(1)
--                    torch.manualSeed(seed)
                    if thread_info.opt.verbose >= 0 then print(string.format('Starting thread with id: %d seed: %d', tid, seed)) end
                end
          );

    end    

    for jj=1,num_samples,opt.batchSize do

        --for ii=1,data_loading_jobs do
            data_load_thread:addjob(
                 -- the job callback (runs in data-worker thread)
                 function()
                    local inputs, labels = loadBatch(jj)
                    return inputs, labels
                 end,
                 -- the end callback (runs in the main thread)
                 testBatch
              )     
    end        
        
    data_load_thread:synchronize()
    cutorch.synchronize() 
 
    data_load_thread:terminate()   
        
    collectgarbage()  
      
--    local totalValid = 0 
--    for i=1,#all_classes do
--        print(confusions[all_classes[i]])
--        totalValid = totalValid + getF1Score(confusions[all_classes[i]])
--    end
    local avg_measures = getAvgMeasures(confusions,all_classes)
--    totalValid = totalValid / #all_classes

    if avg_measures["f1score"] > best_F1score_so_far then
        local filename = paths.concat(opt.path_save, 'best_F1score_model.net')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)      
        saveDataParallel(filename, model)

        best_F1score_so_far = avg_measures["f1score"]
            
        -- Save perfomances on file
        for jj=1,#all_classes do   
            save_perfomances(opt.path_save .. '/perfomances_' .. all_classes[jj] .. '.txt', confusions[all_classes[jj]],all_classes[jj])
        end                 
    end
    
    local info_to_log = logInfos(confusions)  
    for i=1,#all_classes do
        info_to_log[all_classes[i]]["tot_correct_sequences"] = total_correct
        info_to_log[all_classes[i]]["tot_sequences"] = num_samples
        testLogger[all_classes[i]]:add(info_to_log[all_classes[i]])    
        confusions[all_classes[i]]:zero()
    end
     
            
--    testLogger:add{['% mean class F1 score (val set)'] = avg_measures["f1score"] } 
    print("Correct sequences = " .. total_correct .. " / " .. num_samples) 
--    print("Avg. F1 score = " .. totalValid)  
    return avg_measures["f1score"] 
end
