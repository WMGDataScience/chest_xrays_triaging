require 'torch'   -- torch
require 'optim'   -- an optimization package, for online and batch methods
require 'utils'

tds = require 'tds'
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

--testLogger = optim.Logger(paths.concat(opt.path_save, 'test.log'))

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataLoadingTimer = torch.Timer()

local batchNumber = 0
local total_batches

-- used to store the results of the network
local net_predictions = tds.Hash()

confusions = {}
for i=1,#opt.classes do
    confusions[opt.classes[i]] = optim.ConfusionMatrix({opt.classes[i],"not_" .. opt.classes[i]})
end

local function loadBatch(index)
    local dataTimer = torch.Timer()
    dataTimer:reset()
    
    local batch_dim = math.min(#info.imgs_paths_and_infos - (index-1),info.opt.batchSize)
    local imgs_batch = torch.FloatTensor(batch_dim,info.opt.input_size[1],info.opt.input_size[2],info.opt.input_size[3])
    local labels_batch = torch.FloatTensor(batch_dim,#info.classes)
    local accnumbs_batch = tds.Vec()

    for i = 1,batch_dim do   
        local current_elem = info.imgs_paths_and_infos[info.shuffle[index + i - 1]]
        imgs_batch[i]:copy(getImage(current_elem,info))
        labels_batch[i] = getLabelsSequence(current_elem.classes,info.classes)
        accnumbs_batch[i] = current_elem.acc_numb
    end

    if info.opt.verbose >= 2 then print((">imgs loading [no preprocess] = %.2f seconds<"):format(dataTimer:time().real)) end     
    
    if info.opt.doPreprocess == true then
        preprocess(imgs_batch, info.params['mean'], info.params['var'])
    end

    if info.opt.verbose >= 1 then print((">Time to load data from disk = %.2f seconds<"):format(dataTimer:time().real)) end 

    return imgs_batch, labels_batch, accnumbs_batch
end

local total_correct = 0

local function testBatch(inputsCPU, labelsCPU,acc_numbs)
    local data_loading_time = 0
    cutorch.synchronize()
    collectgarbage()
    timer:reset()
    
    inputs:resize(inputsCPU:size(1),inputsCPU:size(2),inputsCPU:size(3),inputsCPU:size(4))
    labels:resize(inputsCPU:size(1),#classes) 
    
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

        net_predictions[acc_numbs[i]] = updateConfusions(outputs[i],labels[i],confusions,classes)
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


function test(imgs_paths_and_infos)

    batchNumber = 0
    total_correct = 0
    
    local all_classes = table.copy(classes)
    table.insert(all_classes,'normal')  
       
    print('==> test:')

    cutorch.synchronize()
    model:evaluate()

    if opt.testMode ~= '' then
        opt.test_epoch_size = #imgs_paths_and_infos
    end
    local total_images = math.min(#imgs_paths_and_infos,opt.test_epoch_size)    
    total_batches = (total_images-(total_images%opt.batchSize))/opt.batchSize + 1
    
    
    local shuffle = torch.range(1,#imgs_paths_and_infos)
   
    print("Number of input images: " .. #imgs_paths_and_infos)
    print("Number of images really used: " .. total_images)
        
    do -- start K datathreads (donkeys)

            local thread_info = {
                opt = opt,
                params = params,
                imgs_paths_and_infos = imgs_paths_and_infos,
                shuffle = shuffle, 
                aug_flag = false,            
                classes = classes
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

    for jj=1,total_images,opt.batchSize do

        --for ii=1,data_loading_jobs do
            data_load_thread:addjob(
                 -- the job callback (runs in data-worker thread)
                 function()
                    local inputs, labels, acc_numbs = loadBatch(jj)
                    return inputs, labels, acc_numbs
                 end,
                 -- the end callback (runs in the main thread)
                 testBatch
              )     
    end        
        
    data_load_thread:synchronize()
    cutorch.synchronize() 
 
    data_load_thread:terminate()   
    
    local avg_measures = getAvgMeasures(confusions,all_classes)  
--    local totalValid = 0 
    for i=1,#all_classes do
--        print(confusions[all_classes[i]])
        -- Save perfomances on file
        if opt.testMode == '' then
            save_perfomances(opt.path_save .. '/perfomances_' .. all_classes[i] .. '.txt', confusions[all_classes[i]],all_classes[i])
        else
            save_perfomances(opt.path_test_results .. 'perfomances_' .. all_classes[i] .. '.txt', confusions[all_classes[i]],all_classes[i])
        end
    end

    print("Correct sequences = " .. total_correct .. " / " .. total_images) 
    torch.save(opt.path_test_results .."net_predictions.t7",net_predictions)
end
