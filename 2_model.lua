----------------------------------------------------------------------
-- This script contains the model.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cudnn'
require 'model_utils'

cudnn.benchmark = true-- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                       -- If this is set to false, uses some in-built heuristics that might not always be fastest.
cudnn.fastest = true -- this is like the :fastest() mode for the Convolution modules,
                     -- simply picks the fastest convolution algorithm, rather than tuning for workspace size

function createCleaningNetwork(dim_last_layer)
    
    
    local classes_vec_reduction = math.ceil(#classes) --10
    local feature_vec_reduction = math.ceil(dim_last_layer/4)
    
    local input1 = nn.Identity()()
    local input2 = nn.Identity()() 
    local sub_input1 = nn.AddConstant(-1)(input1)
    
    local linear_1 = nn.Tanh()(nn.Linear(#classes,classes_vec_reduction)(sub_input1))
    local linear_2 = nn.Tanh()(nn.Linear(dim_last_layer,feature_vec_reduction)(nn.View(dim_last_layer)(input2)))

    local concat = nn.JoinTable(2)({linear_1,linear_2})
    
    local middle_linear = nn.Tanh()(nn.Linear(classes_vec_reduction + feature_vec_reduction,classes_vec_reduction + feature_vec_reduction)(concat))
    local final_linear = nn.Tanh()(nn.Linear(classes_vec_reduction + feature_vec_reduction,#classes)(middle_linear))
    
    local final_node = nn.Clamp(0,1)(nn.CAddTable()({final_linear, sub_input1}))

    
    return nn.gModule({input1, input2}, {final_node})
end

function createNewNetwork(noutputs,last_layer)
    local net = nil
    local last_fe_layer_dim = 0
    local num_classifiers = #opt.classes - 1

    if opt.progressive_train == 1 then num_classifiers = #classes_sets[1]  end
    if opt.multi_task then num_classifiers = num_classifiers + 4 end -- 3 classifiers for the priority prediction and 1 for the normal/abnormal
    
    ----------------------------------------------------------------------
    print '==> construct model'
    
    local model_string = split_string(opt.model,"_")
    
    if model_string[1] == 'resnext' then
        assert(cudnn.version >= 4000, "cuDNN v4 or higher is required")
        local resnext_options = {
            depth = tonumber(model_string[2]), --how many layers? 50-101-152
            bottleneckType = 'resnext_C',
            baseWidth = 4,
            cardinality = 32,
            shortcutType = 'B', -- B or C type, for more info see http://arxiv.org/abs/1512.03385
            nOutputs = noutputs,
            predict_age = opt.predict_age,
            classifier_type = opt.classifier_type,        
            num_classifiers = num_classifiers,
            input_size = opt.input_size,        
            numGPU = opt.numGPU,
            last_layer = last_layer
        }    


        require "models/resnext"              
        net,last_fe_layer_dim = createResNeXtModel(resnext_options)    
    elseif model_string[1] == 'resnet' then
        assert(cudnn.version >= 4000, "cuDNN v4 or higher is required")
        local resnet_options = {
            depth = tonumber(model_string[2]), --how many layers? 18-34-50-101-152
            shortcutType = 'B', -- B or C type, for more info see http://arxiv.org/abs/1512.03385
            nOutputs = noutputs,
            predict_age = opt.predict_age,
            classifier_type = opt.classifier_type,        
            num_classifiers = num_classifiers,
            input_size = opt.input_size,        
            numGPU = opt.numGPU,
            last_layer = last_layer
        }    
        if (opt.input_size[2] == 2560 and opt.input_size[3] == 2304) 
           or (opt.input_size[2] == 1280 and opt.input_size[3] == 1152 ) then
            require "models/resnet_2560x2304"
            if #model_string == 3 and model_string[3] == "multiinput" then
                resnet_options.multi_input_flag = true
            else
                resnet_options.multi_input_flag = false
            end
        else
            -- residual model, code here https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
            if #model_string>2 and model_string[3] == "pretrained" then
                require "models/resnet_imgnet_pretrain" 
            else
                require "models/resnet"
            end           
        end

        
        net,last_fe_layer_dim = createResNetModel(resnet_options)
    elseif model_string[1] == 'resnetBig' then
        
        require "models/resnet_biginput"
        
        -- residual model, code here https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
        assert(cudnn.version >= 4000, "cuDNN v4 or higher is required")
        local resnet_options = {
            depth = tonumber(model_string[2]), --how many layers? 18-34-50-101-152
            shortcutType = 'B', -- B or C type, for more info see http://arxiv.org/abs/1512.03385
            nOutputs = noutputs,
            predict_age = opt.predict_age,        
            num_classifiers = num_classifiers,
            input_size = opt.input_size,        
            numGPU = opt.numGPU,
            last_layer = last_layer
        }
        
        net,last_fe_layer_dim = createResNetBigModel(resnet_options)    
    
    elseif model_string[1] == 'inception' then
        assert(cudnn.version >= 4000, "cuDNN v4 or higher is required")
        if model_string[2] == "v3" then        
            
            local inception_options = {
                dump_files_path = "", --path to the google inception v3 dump files
                imageNetPreInit = false,
                mod_flag = false,
                nOutputs = noutputs,
                predict_age = opt.predict_age,
                classifier_type = opt.classifier_type,
                num_classifiers = num_classifiers,
                input_size = opt.input_size,
                numGPU = opt.numGPU,
                last_layer = last_layer
            }
                
            if #model_string>2 then
                require "models/inception_v3_imgnet_pretrain" 
                inception_options.dump_files_path = os.getenv("HOME") .. "/inception-v3.torch-Moodstocks/dump/"
                assert(directory_exists(inception_options.dump_files_path),"download the Moodstocks dumps of inceptionv3 network. Look here: https://github.com/Moodstocks/inception-v3.torch/blob/master/inceptionv3.lua ")                
                
                if model_string[3] == "modified" then
                    inception_options.mod_flag = true
                elseif model_string[3] == "pretrained" then
                    inception_options.imageNetPreInit = true
                end              
            elseif --(opt.input_size[2] == 2560 and opt.input_size[3] == 2304) or
                (opt.input_size[2] == 1211 and opt.input_size[3] == 1083 ) then
                require "models/inception_v3_1211x1083"
            else
                require "models/inception_v3"           
            end

            net,last_fe_layer_dim = createInceptionv3Model(inception_options)

        elseif model_string[2] == "v4" then
            --inceptionv4 network, inspiration code here https://github.com/itaicaspi/inception-v4.torch/blob/master/inceptionv4.lua
            
            require "models/inception_v4"
            
            local inception_options = {
                nOutputs = noutputs,
                predict_age = opt.predict_age,
                classifier_type = opt.classifier_type,
                num_classifiers = num_classifiers,
                input_size = opt.input_size,
                numGPU = opt.numGPU,
                last_layer = last_layer
            }
            
            net,last_fe_layer_dim = createInceptionv4Model(inception_options)           
        end
    elseif model_string[1] == 'VGG' then
    
        require "models/VGG_like_net"
        local VGG_options = {
            nOutputs = noutputs,
            predict_age = opt.predict_age,
            classifier_type = opt.classifier_type,
            num_classifiers = num_classifiers,
            input_size = opt.input_size,
            numGPU = opt.numGPU,
            last_layer = last_layer
        }
        
        net,last_fe_layer_dim = createVGGLikeModel(VGG_options)   
     
    elseif model_string[1] == 'bigInputNet' then
        require "models/VGG_like_big_net"
        
        local VGG_options = {
            nOutputs = noutputs,
            predict_age = opt.predict_age,
            classifier_type = opt.classifier_type,
            num_classifiers = num_classifiers,
            input_size = opt.input_size,
            numGPU = opt.numGPU,
            last_layer = last_layer
        }
        
        net,last_fe_layer_dim = createVGGLikeBigModel(VGG_options)  
    else
    
       error('unknown -model')
    
    end
    return net,last_fe_layer_dim
end

local function createFinalClassifiersConcat(last_fe_layer_dim,num_classifiers,last_layer,noutputs,layers)
    local classifiers = nn.Concat(2)
    for i = 1,num_classifiers do
        local classifier = nn.Sequential()
        classifier:add(nn.View(-1):setNumInputDims(3))
        if layers == 2 then
            classifier:add(nn.Dropout(0.2))
            classifier:add(nn.Linear(last_fe_layer_dim, 256))
            classifier:add(cudnn.ReLU(true))
            classifier:add(nn.BatchNormalization(256))
            classifier:add(nn.Dropout(0.3))
            classifier:add(nn.Linear(256, noutputs))
        else
            classifier:add(nn.Dropout(0.2)):add(nn.Linear(last_fe_layer_dim, noutputs))
        end
        
        if last_layer then
            classifier:add(last_layer())
        end
        classifiers:add(classifier) 
    end
    return classifiers
end

local function loadLSTMCHSequenceProcessing()
    require 'rnn'
    local preTrainedNet = torch.load(opt.preTrainedCHLSTM )
    local CHSequenceProcessing_net, last_view
    if torch.type(preTrainedNet) == 'nn.DataParallelTable' then
        CHSequenceProcessing_net,last_view = preTrainedNet:get(1):get(1):get(1):get(2),preTrainedNet:get(1):get(1):get(1):get(2):get(#preTrainedNet:get(1):get(1):get(1):get(2))
    elseif torch.type(preTrainedNet) == 'nn.Sequential' then
        CHSequenceProcessing_net,last_view = preTrainedNet:get(1):get(1):get(2),preTrainedNet:get(1):get(1):get(2):get(#preTrainedNet:get(1):get(1):get(2))
    end
    
    CHSequenceProcessing_net = replaceResetMaskedSelect(CHSequenceProcessing_net)
   
    --CHSequenceProcessing_net.accGradParameters = function () end --disable backprop on net
    return CHSequenceProcessing_net,last_view.numElements
end

local function loadPreTrainedNetwork(noutputs,last_layer)
    

    local net,last_fe_layer_dim = nn.Sequential(),nil
    local CNN, concat_layer = nil,nil
    local preTrainedNet = torch.load(opt.preTrainedCNN) --reBuildSavedModel(torch.load(opt.preTrainedCNN)) 

    if torch.type(preTrainedNet) == 'nn.DataParallelTable' then
        CNN, concat_layer = preTrainedNet:get(1):get(1),preTrainedNet:get(1):get(2)
    elseif torch.type(preTrainedNet) == 'nn.Sequential' then
        CNN, concat_layer = preTrainedNet:get(1),preTrainedNet:get(2)
    end
    --CNN.accGradParameters = function () end --disable backprop on CNN

    if torch.type(concat_layer:get(1)) == 'nn.Sequential' then
        for i=1,concat_layer:get(1):size() do
            if torch.type(concat_layer:get(1):get(i)) == 'nn.Linear' then
                last_fe_layer_dim = concat_layer:get(1):get(i).weight:size(2)
                break
            end
        end
        assert(last_fe_layer_dim ~= nil,"no Linear layers in the first classifier? It is impossible")
    else
        last_fe_layer_dim = concat_layer:get(2).weight:size(2)
    end
    if opt.tensor_precision == 'half' then
        cudnn.convert(CNN, nn, function(module)
            return not torch.type(module):find('BatchNormalization')
        end)
    end
    
    if opt.classifier_type == "multi" then
        local num_classifiers = #opt.classes - 1

        if opt.progressive_train == 1 then num_classifiers = #classes_sets[1]  end
        local classifiers = createFinalClassifiersConcat(last_fe_layer_dim,num_classifiers,last_layer,noutputs,1)     

        net:add(CNN):add(classifiers)

    elseif opt.classifier_type == "single" then
        local classifier = nn.Sequential()
        classifier:add(nn.View(-1):setNumInputDims(3)):add(nn.Linear(last_fe_layer_dim, noutputs))    
        if last_layer then
            classifier:add(last_layer())
        end
        net:add(CNN):add(classifier)     
    end    

    net = makeDataParallel(net, opt.numGPU) 

    return net,last_fe_layer_dim
end


function addClassifiers(model,num_classifiers)
    print('Adding '.. num_classifiers..' classifiers')
    local net = nn.Sequential()
    
    local CNN, concat_layer
    if torch.type(model) == 'nn.DataParallelTable' then
        CNN, concat_layer = model:get(1):get(1):clone(),model:get(1):get(2):clone() 
    elseif torch.type(model) == 'nn.Sequential' then
        CNN, concat_layer = model:get(1),model:get(2)
    end

    print(concat_layer:size())          
    for i = 1,num_classifiers do
        local classifier = concat_layer:get(1):clone()
        classifier:reset()
        if opt.multi_task then
            table.insert(concat_layer.modules,concat_layer:size()-3,classifier) 
        else
            concat_layer:add(classifier) 
        end   
    end 
    print(concat_layer:size()) 
    
    net:add(CNN):add(concat_layer)
    net = makeDataParallel(net, opt.numGPU) 
    
    if opt.multi_task then
        -- we need these instructions in order to prevent an error with the multiGPU
        net:training() 
        local _ = net:forward(createCudaTensor(torch.LongStorage({opt.batchSize,opt.input_size[1],opt.input_size[3],opt.input_size[2]})))
        net:evaluate()
        _ = net:forward(createCudaTensor(torch.LongStorage({opt.batchSize,opt.input_size[1],opt.input_size[3],opt.input_size[2]})))
    end
    return net
end

function createModel()

    local net,last_fe_layer_dim = nil,nil

    local last_layer = nil
    if opt.loss == "nll" then
        last_layer = cudnn.LogSoftMax
    elseif opt.loss == "bce" then
        last_layer = cudnn.SoftMax
    elseif opt.loss == "ce" then
        last_layer = nil 
    elseif opt.loss == "smc" then 
        last_layer = cudnn.Sigmoid    
    end
    
    local noutputs = 0
    
    -- 2-class problem
    if opt.classifier_type == "multi" then
        if opt.loss == "smc" then
            noutputs = 1
        else
            noutputs = 2
        end
    elseif opt.classifier_type == "single" then
        noutputs = #classes
    elseif opt.classifier_type == "RNN" then
        noutputs = #opt.classes  
    end
    
    if opt.preTrainedCNN == "" then
        net,last_fe_layer_dim = createNewNetwork(noutputs,last_layer)
    else
        net,last_fe_layer_dim = loadPreTrainedNetwork(noutputs,last_layer)
    end
    
    ----------------------------------------------------------------------
    -- Printing and visualization
    ----------------------------------------------------------------------
    if opt.verbose >= 2 then
        print '==> here is the model:'
        print(net)
    end  
    
    
    if opt.path_CH_embeddings ~= "" or covariate_variables_dim > 0 then
        local last_cov_var_embedding = 256
        assert((opt.path_CH_embeddings == nil and covariate_variables_dim > 0) or (opt.path_CH_embeddings ~= nil or covariate_variables_dim <= 0),"model with covariate variables and CH_embeddigns not yet implemented")
        
        local CNN, concat_layer = nil,nil
        if torch.type(net) == 'nn.DataParallelTable' then
            CNN, concat_layer = net:get(1):get(1),net:get(1):get(2)
        elseif torch.type(net) == 'nn.Sequential' then
            CNN, concat_layer = net:get(1),net:get(2)
        end
        
        
        local parallel = nn.ParallelTable()
        parallel:add(nn.Sequential():add(CNN)
--                :add(nn.Padding(2,-1,4)):add(nn.Narrow(2,1,1)):add(nn.Padding(2,last_fe_layer_dim-1,4))
                :add(nn.View(-1):setNumInputDims(3)))
        
        if opt.path_CH_embeddings == "" then
            parallel:add(nn.Sequential():add(nn.Linear(covariate_variables_dim,30)):add(nn.BatchNormalization(30)):add(cudnn.ReLU(true)):add(nn.Linear(30,10)):add(nn.BatchNormalization(10)):add(cudnn.ReLU(true)))
        else
            if opt.preTrainedCHLSTM == "" then
                if opt.text_embedding_net == "sum_linear" then
                    local sum_embeddings = nn.Sequential()
                    sum_embeddings:add(nn.Sum(2))
                    sum_embeddings:add(nn.Sum(2))              
                    sum_embeddings:add(nn.Linear(CH_emb_size,1024))
                    sum_embeddings:add(nn.BatchNormalization(1024))
                    sum_embeddings:add(cudnn.ReLU(true))
                    sum_embeddings:add(nn.Dropout(0.3))
                    sum_embeddings:add(nn.Linear(1024,last_cov_var_embedding))
                    sum_embeddings:add(nn.BatchNormalization(last_cov_var_embedding))
                    sum_embeddings:add(cudnn.ReLU(true))
                    sum_embeddings:add(nn.Dropout(0.3))                     
                    parallel:add(sum_embeddings)              
                elseif opt.text_embedding_net == "sum" then
                    local sum_embeddings = nn.Sequential()
                    sum_embeddings:add(nn.Sum(2))
                    sum_embeddings:add(nn.Sum(2))                                  
                    parallel:add(sum_embeddings) 
                    last_cov_var_embedding = CH_emb_size                 
                else
                    require 'rnn'            
                    
                    local reduce_dim_embedding = nn.Sequential()            
                    --reduce_dim_embedding:add(nn.Linear(4096,1024))
                    --reduce_dim_embedding:add(cudnn.ReLU(true))
                    --reduce_dim_embedding:add(nn.BatchNormalization(1024))
                    --reduce_dim_embedding:add(nn.Dropout(0.3))            
                    reduce_dim_embedding:add(nn.Linear(1024,last_cov_var_embedding))
                    reduce_dim_embedding:add(cudnn.ReLU(true))
                    --reduce_dim_embedding:add(nn.BatchNormalization(last_cov_var_embedding))
                    reduce_dim_embedding:add(nn.Dropout(0.3))  
        
                    local LSTM = nn.SeqLSTM(CH_emb_size, 1024) --nn.SeqLSTMP(CH_emb_size, 2048,512)            
                    LSTM.maskzero = true 
                    local process_embedding = nn.Sequential():add(nn.Transpose({1, 2})):add(LSTM):add(nn.Sequencer(reduce_dim_embedding)):add(nn.Transpose({2, 1})) 
                    local to_byte_tensor = nn.Sequential():add(nn.Copy(nil,'torch.CudaByteTensor',false,true)):add(nn.Narrow(3, 1, last_cov_var_embedding))
                    local CH_embeddings_net_parallel = nn.ParallelTable():add(process_embedding):add(to_byte_tensor)
         
                    local CH_embeddings_net = nn.Sequential()            
                    CH_embeddings_net:add(nn.SplitTable(2, 4))           
                    CH_embeddings_net:add(CH_embeddings_net_parallel)          
                    CH_embeddings_net:add(nn.MaskedSelect())
                    CH_embeddings_net:add(nn.View(-1,last_cov_var_embedding)) 
                               
                    parallel:add(CH_embeddings_net)
                end
            else
                local sub_net = nil
                sub_net,last_cov_var_embedding = loadLSTMCHSequenceProcessing()
                parallel:add(sub_net)
            end              
        end
        
        local linear_nodes, container_nodes = concat_layer:findModules('nn.Linear')
        local final_features_dim
        for i = 1, #linear_nodes do
          -- Search the container for the current Linear node
          for j = 1, #(container_nodes[i].modules) do
            if container_nodes[i].modules[j] == linear_nodes[i] and linear_nodes[i].weight:size(2) == last_fe_layer_dim then
                -- Replace with a new instance
                container_nodes[i].modules[j] = nn.Linear(linear_nodes[i].weight:size(2)+last_cov_var_embedding,linear_nodes[i].weight:size(1))
                final_features_dim = linear_nodes[i].weight:size(2)+last_cov_var_embedding
            end
          end
        end

        local view_nodes, container_nodes = concat_layer:findModules('nn.View')
        for i = 1, #view_nodes do
          -- Search the container for the current Linear node
          for j = 1, #(container_nodes[i].modules) do
            if container_nodes[i].modules[j] == view_nodes[i] then
              -- Replace with a new instance
              container_nodes[i].modules[j] = nn.View(-1):setNumInputDims(1)
            end
          end
        end        

        local final_net = nn.Sequential()
        if opt.feature_extraction then
            final_net:add(nn.Sequential():add(parallel):add(nn.JoinTable(2))):add(nn.View(-1,final_features_dim))
        else
            final_net:add(nn.Sequential():add(parallel):add(nn.JoinTable(2))):add(concat_layer)
            
            for k,v in pairs(final_net:findModules('nn.Linear')) do
                v.bias:zero()
            end
    
            for k,v in pairs(final_net:findModules('nn.BatchNormalization')) do
                v.weight:fill(1)
                v.bias:zero()
            end
        end
      
        
        toCuda(final_net)
        return makeDataParallel(final_net,opt.numGPU)     
    else
--        if opt.label_cleaning_network == 1 then
--            require 'nngraph'
--            local cn = toCuda(createCleaningNetwork(last_fe_layer_dim))
--            return net,cn
--        else 
            return net
--        end   
    end


 
end
