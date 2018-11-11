require 'nn'      -- provides all sorts of trainable modules/layers
require 'rnn'
require 'cunn'
require 'cudnn'
require 'model_utils'

local function createRNN(dim_input,dim_output,dim_hidden)
    local RNN = nn.Sequential()
    RNN:add(cudnn.LSTM(dim_input, dim_hidden, 1,true))
    RNN:add(nn.Dropout(0.3))
    RNN:add(cudnn.LSTM(dim_hidden, dim_output, 1,true))
    RNN:add(nn.Dropout(0.3))  
    return RNN    
end

local function getCNN(net)
    if torch.type(model) == 'nn.DataParallelTable' then
        return net:get(1):get(1)
    else
        return net:get(1)
    end
end

function updateClassifier(model,num_class)
    print('Adding '.. num_class ..' to final classifier')
    
    local net = nn.Sequential()
    
    local CNN, lstm_layers
    if torch.type(model) == 'nn.DataParallelTable' then
        CNN, lstm_layers = model:get(1):get(1):clone(),model:get(1):get(2):clone() 
    elseif torch.type(model) == 'nn.Sequential' then
        CNN, lstm_layers = model:get(1),model:get(2)
    end  
    
    local lstm_input_size,output = 0,0
    local last_layer = nil
    if opt.loss == "nll" then
        last_layer = cudnn.LogSoftMax
    elseif opt.loss == "bce" then
        last_layer = cudnn.SoftMax
    end
        
    local Linear_nodes, container_nodes = lstm_layers:findModules('nn.Linear')
    for i = 1, #Linear_nodes do
      -- Search the container for the current Linear node
      for j = 1, #(container_nodes[i].modules) do
        if container_nodes[i].modules[j] == Linear_nodes[i] then
            -- Replace with a new instance
            lstm_input_size = Linear_nodes[i].weight:size(2)
            output = Linear_nodes[i].weight:size(1)+num_class
        end
      end
    end      

    local linear = nn.Sequential():add(nn.View(lstm_input_size)):add(nn.Dropout(0.2)):add(nn.Linear(lstm_input_size,output))
    if last_layer then
        linear:add(last_layer())
    end

    local Sequencer_nodes, container_nodes = lstm_layers:findModules('nn.Sequencer')
    for i = 1, #Sequencer_nodes do
      -- Search the container for the current Linear node
      for j = 1, #(container_nodes[i].modules) do
        if container_nodes[i].modules[j] == Sequencer_nodes[i] and torch.type(container_nodes[i].modules[j]:get(1):get(1)) == 'nn.Sequential' then
            -- Replace with a new instance
            container_nodes[i].modules[j] = nn.Sequencer(linear)
        end
      end
    end
    
    local Replicate_nodes, container_nodes = lstm_layers:get(3):get(3):findModules('nn.Replicate')
    for i = 1, #Replicate_nodes do
      -- Search the container for the current Linear node
      for j = 1, #(container_nodes[i].modules) do
        if container_nodes[i].modules[j] == Replicate_nodes[i] then
            -- Replace with a new instance
            container_nodes[i].modules[j] = nn.Replicate(Replicate_nodes[i].nfeatures + num_class,1,1) 
            print(container_nodes[i].modules[j].nfeatures) 
        end
      end
    end

    net:add(CNN):add(lstm_layers)
    toCuda(net)
    net = makeDataParallel(net, opt.numGPU) 
    return net
end


local function pretrain_CNN(opt)
    
    
    local pretrain_data = splitPerClass(train_infos,opt.classes)
    local pretrain_val_data = filterElementWithOverlappingLabels(val_infos,opt.classes)

    local old_classes = classes
    classes = {}
    for k,v in pairs (pretrain_data) do
        table.insert(classes,k)
    end 
    
    local ep = 1

    require "2_model"
    local to_pretrain_model = createModel()    
    model = to_pretrain_model
    print(model)

    dofile 'train_multithread.lua'
    dofile 'test_multithread.lua'
    dofile '3_loss.lua'

    while ep <= opt.preTrainingEpochs do
        print("==========> Pretrain epoch " .. ep)  
        train(pretrain_data)
        test(pretrain_val_data)       
        ep = ep + 1 
    end
    classes = old_classes
    epoch = nil
    return to_pretrain_model
end
function createModel_LSTM(opt)
    local last_layer = nil
    if opt.loss == "nll" then
        last_layer = cudnn.LogSoftMax
    elseif opt.loss == "bce" then
        last_layer = cudnn.SoftMax
    end
    
    local lstm_input_size = 512
    if opt.model == "inception_v3" then
        lstm_input_size = 2048
    end
    local CNN = nil

    if opt.preTrainedCNN == "" then
--        -- train a standard CNN a set of images with non-overlapping classes
--        opt.classifier_type = "single"       
--        local to_pretrain_model = pretrain_CNN(opt)
--        saveDataParallel(paths.concat(opt.path_save, 'pre_trained_model.net'), to_pretrain_model)
--        opt.classifier_type = "RNN"
--        CNN = getCNN(to_pretrain_model)
        require '2_model'

        CNN,lstm_input_size = createNewNetwork(2,last_layer)
                
    else
        assert(file_exists(opt.preTrainedCNN),"Error, file " .. opt.preTrainedCNN .. " does not exist")
        CNN = getCNN(torch.load(opt.preTrainedCNN))  
    end

    local LSTM_par = {lstm_input_size,256,opt.LSTM_hidden_layers} -- [ in , out, hidden]
    local output = #classes + 1 -- plus stop signal
    

--    local clone_feats = nn.Concat(2)
--    for i = 1,#classes do
--        clone_feats:add(nn.Identity())
--    end

--    local pre_RNN = nn.Sequential()  
--    pre_RNN:add(nn.Replicate(#classes,1,2))
--    pre_RNN:add(nn.Copy(nil,nil,true))
--    pre_RNN:add(nn.View(-1,#classes,LSTM_par[1]))
--    pre_RNN:add(nn.Copy(nil,nil,true))


    local zeroInputVector = nn.Sequential():add(nn.Padding(1,-1,1)):add(nn.Narrow(2,1,1)):add(nn.Replicate(#classes,1,1)):add(nn.Transpose({1,2}))--:add(nn.PrintSize("zeroInputVector"))

    local pre_RNN = nn.Sequential()  
    pre_RNN:add(nn.Replicate(2,1,1)) --batchSize x featureEmbeddingSize -->  batchSize x 2 x featureEmbeddingSize 
    pre_RNN:add(nn.SplitTable(1, 2)) --batchSize x 2 x featureEmbeddingSize --> {batchSize x featureEmbeddingSize , batchSize x featureEmbeddingSize}
    pre_RNN:add(nn.ParallelTable():add(nn.Identity()):add(zeroInputVector))
        
--    local linear = nn.Sequential():add(nn.View(LSTM_par[2])):add(nn.Linear(LSTM_par[2],output))
    local linear = nn.Sequential():add(nn.View(lstm_input_size)):add(nn.Dropout(0.2)):add(nn.Linear(lstm_input_size,output))
    if last_layer then
        linear:add(last_layer())
    end
    
    local lstm_layers = nn.Sequential()  
    --lstm_layers:add(nn.PrintSize("before LSTM"))    
    lstm_layers:add(nn.View(lstm_input_size))
    lstm_layers:add(nn.Dropout(0.2))
    lstm_layers:add(pre_RNN)
    lstm_layers:add(nn.SeqLSTM(1, lstm_input_size))
    lstm_layers:add(nn.Sequencer(nn.NormStabilizer()))
    --lstm_layers:add(nn.PrintSize("preLinearSequencer"))    
    lstm_layers:add(nn.Sequencer(linear))
    lstm_layers:add(nn.Transpose({1,2}))
    --lstm_layers:add(nn.PrintSize("after LSTM layers"))
    
--    lstm_layers:add(pre_RNN):add(createRNN(LSTM_par[1],LSTM_par[2],LSTM_par[3])):add(nn.Transpose({1,2})):add(nn.Sequencer(linear)):add(nn.Transpose({1,2}))
    local model = nn.Sequential()
--    model:add(CNN):add(clone_feats):add(nn.View(-1,#classes,LSTM_par[1])):add(nn.Copy(nil,nil,true)):add(createRNN(LSTM_par[1],LSTM_par[2],LSTM_par[3])):add(nn.Transpose({1,2})):add(nn.Sequencer(linear)):add(nn.Transpose({1,2}))
    --CNN = makeDataParallel(CNN,opt.numGPU)
    model:add(CNN):add(lstm_layers)
    toCuda(model)
    model = makeDataParallel(model,opt.numGPU)
    
    if opt.verbose >= 1 then print(model) end
    return model
end
