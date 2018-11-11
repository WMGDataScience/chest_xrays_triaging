function createResNetModel(resnet_options)
    local cfg = {
--        [18]  = {{2, 2, 2, 2}, {64,128,256,512,512}},
--        [34]  = {{3, 4, 6, 3}, {64,128,256,512,512}},
        [50]  = {{3, 4, 6, 3}, {64,128,256,512,2048}},
--        [101] = {{3, 4, 23, 3}, {64,128,256,512,2048}},
--        [152] = {{3, 8, 36, 3}, {64,128,256,512,2048}},
    }    
    assert(cfg[resnet_options.depth], 'Invalid depth: ' .. tostring(resnet_options.depth))
    local def, nFeatures = table.unpack(cfg[resnet_options.depth])
    
    print(' | ResNet-' .. resnet_options.depth)
    local fe_CNN = torch.load('models/pretrained_models/resnet-'.. resnet_options.depth ..'.t7')
    
    --remove last two layers (view and linear)
    fe_CNN:remove(#fe_CNN)
    fe_CNN:remove(#fe_CNN)
    local c1_weight = fe_CNN:get(1).weight
    local new_conv_1 = cudnn.SpatialConvolution(resnet_options.input_size[1],c1_weight:size(1),c1_weight:size(3),c1_weight:size(4),fe_CNN:get(1).dW,fe_CNN:get(1).dH,fe_CNN:get(1).padW,fe_CNN:get(1).padH)
    if new_conv_1.weight:size(2) == c1_weight:size(2) then
       new_conv_1.weight:copy(c1_weight) 
    else
       new_conv_1.weight:copy(c1_weight:narrow(2,1,1))
    end
    fe_CNN:remove(1)
    fe_CNN:insert(new_conv_1,1)
    
    if resnet_options.input_size[2] ~= 224 or resnet_options.input_size[3] ~= 224  then
        local final_pool_size_w = resnet_options.input_size[2] / math.pow(2,5)
        local final_pool_size_h = resnet_options.input_size[3] / math.pow(2,5)
        local new_avg_pool = cudnn.SpatialAveragePooling(final_pool_size_w,final_pool_size_h,1,1) 
        fe_CNN:remove(#fe_CNN)
        fe_CNN:add(new_avg_pool)
    end
    
    
    local net = nn.Sequential()
    if resnet_options.classifier_type == "multi"  then
        local classifiers = nn.Concat(2)
        for i = 1,resnet_options.num_classifiers do
            local classifier = nn.Sequential()
            classifier:add(nn.View(nFeatures[5]):setNumInputDims(3)):add(nn.Dropout(0.2)):add(nn.Linear(nFeatures[5],resnet_options.nOutputs))
            if resnet_options.last_layer then
                classifier:add(resnet_options.last_layer())
            end
            classifiers:add(classifier) 
        end
        if resnet_options.predict_age then     
            local classifier = nn.Sequential() 
            classifier:add(nn.View(nFeatures[5])):add(nn.Dropout(0.2)):add(nn.Linear(nFeatures[5],512))classifier:add(cudnn.ReLU(true))    
            classifier:add(nn.Dropout(0.2)):add(nn.Linear(512, 1))  
            classifiers:add(classifier)         
        end        
        toCuda(classifiers) 
        net:add(fe_CNN):add(classifiers)
    elseif resnet_options.classifier_type == "single" then
        local classifier = nn.Sequential()
        classifier:add(nn.View(nFeatures[5]):setNumInputDims(3))
        classifier:add(nn.Linear(nFeatures[5], resnet_options.nOutputs))
        toCuda(classifier)
        net:add(fe_CNN):add(classifier)
    else
        error("classifier type do not supported")   
    end     
    
    local function ConvInit(name)
        for k,v in pairs(net:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            if cudnn.version >= 4000 then
                v.bias = nil
                v.gradBias = nil
            else
                v.bias:zero()
            end
        end
    end
    local function BNInit(name)
        for k,v in pairs(net:findModules(name)) do
            v.weight:fill(1)
            v.bias:zero()
        end
    end

    ConvInit('cudnn.SpatialConvolution')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')
    for k,v in pairs(net:findModules('nn.Linear')) do
        v.bias:zero()
    end
    
    return makeDataParallel(net, resnet_options.numGPU),nFeatures[5]    
end
