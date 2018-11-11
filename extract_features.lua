require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'utils'

tds = require 'tds'
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data_loading_threads = opt.dataLoadingThreads

local inputs = createCudaTensor()
local timer = torch.Timer()
local batchNumber,total_batches,total_images = 0,nil,nil 
if covariate_variables_dim > 0 then
    covariate_inputs = createCudaTensor(opt.batchSize,covariate_variables_dim)
end
if opt.path_CH_embeddings ~= nil then
    covariate_inputs = createCudaTensor(torch.LongStorage({opt.batchSize,2,max_CH_len,CH_emb_size}))
end
local function evaluateBatch(inputsCPU, labelsCPU,acc_numbs)

    cutorch.synchronize()

    timer:reset()
    
    if opt.fixed_size_input == 1 then
        local outputs
        if covariate_variables_dim > 0 or opt.path_CH_embeddings ~= nil then
            inputs:resize(opt.batchSize,inputsCPU[1]:size(2),inputsCPU[1]:size(3),inputsCPU[1]:size(4))        
            inputs[{{1,inputsCPU[1]:size(1)}}]:copy(inputsCPU[1])

            covariate_inputs[{{1,inputsCPU[1]:size(1)}}]:copy(inputsCPU[2])

            outputs = model:forward({inputs,covariate_inputs})[{{1,inputsCPU[1]:size(1)}}]:float()
        else
            inputs:resize(opt.batchSize,inputsCPU:size(2),inputsCPU:size(3),inputsCPU:size(4))  
            inputs[{{1,inputsCPU:size(1)}}]:copy(inputsCPU)
            outputs = model:forward(inputs)[{{1,inputsCPU:size(1)}}]:float()
        end    
        for i =1,#acc_numbs do
            torch.save(opt.path_test_results .. acc_numbs[i]..".t7",outputs[i])
        end 
    else
        for i=1,#inputsCPU do
            inputs:resize(1,inputsCPU[i]:size(1),inputsCPU[i]:size(2),inputsCPU[i]:size(3))
            inputs:copy(inputsCPU[i])
            local output = model:forward(inputs):max(3):max(4)
            
            local filename = opt.path_test_results .. acc_numbs[i]..".t7"
            if not file_exists(filename) then
                torch.save(filename,output)
            end
        end
        
    end 
    

   
  
    if opt.verbose >= 0 then 
        print((">Time to forward/backward data through the network = %.2f seconds [%d/%d]<"):format(timer:time().real,batchNumber,total_batches))
    end    
    cutorch.synchronize()
    batchNumber = batchNumber + 1 
end

function extractFeaturesFromData(imgs_paths_and_infos)
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
    

    opt.test_epoch_size = #imgs_paths_and_infos
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
                 evaluateBatch
              )     
    end 
               
        
    data_load_thread:synchronize()
    cutorch.synchronize() 
    data_load_thread:terminate() 
    collectgarbage()
  
end