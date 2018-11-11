function createResNetModel(resnet_options)
    local Convolution = cudnn.SpatialConvolution
    local Avg = cudnn.SpatialAveragePooling
    local ReLU = cudnn.ReLU
    local Max = cudnn.SpatialMaxPooling
    local SBatchNorm = cudnn.SpatialBatchNormalization
    if opt.tensor_precision == 'half' then
        SBatchNorm = nn.SpatialBatchNormalization
    end
        
    local depth = resnet_options.depth
    local shortcutType = resnet_options.shortcutType or 'B'
    local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
    local function shortcut(nInputPlane, nOutputPlane, stride)
        local useConv = shortcutType == 'C' or 
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
        if useConv then
            -- 1x1 convolution
            return nn.Sequential()
             :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
             :add(SBatchNorm(nOutputPlane))
             :add(nn.SpatialDropout(0.2))
        elseif nInputPlane ~= nOutputPlane then
            -- Strided, zero-padded identity shortcut
            return nn.Sequential()
             :add(nn.SpatialAveragePooling(1, 1, stride, stride))
             :add(nn.Concat(2)
                :add(nn.Identity())
                :add(nn.MulConstant(0)))
        else
            return nn.Identity()
        end
    end

    -- The basic residual layer block for 18 and 34 layer network
    local function basicblock(n, stride)
        local nInputPlane = iChannels
        iChannels = n

        local s = nn.Sequential()
        s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
        s:add(SBatchNorm(n))
        s:add(nn.SpatialDropout(0.2))
        s:add(ReLU(true))
        s:add(Convolution(n,n,3,3,1,1,1,1))
        s:add(SBatchNorm(n))
        s:add(nn.SpatialDropout(0.2))
        
        return nn.Sequential()
         :add(nn.ConcatTable()
         :add(s)
         :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

    -- The bottleneck residual layer for 50, 101, and 152 layer networks
    local function bottleneck(n, stride)
        local nInputPlane = iChannels
        iChannels = n * 4

        local s = nn.Sequential()
        s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
        s:add(SBatchNorm(n))
        s:add(nn.SpatialDropout(0.2))
        s:add(ReLU(true))
        s:add(Convolution(n,n,3,3,stride,stride,1,1))
        s:add(SBatchNorm(n))
        s:add(nn.SpatialDropout(0.2))
        s:add(ReLU(true))
        s:add(Convolution(n,n*4,1,1,1,1,0,0))
        s:add(SBatchNorm(n * 4))
        s:add(nn.SpatialDropout(0.2))

        return nn.Sequential()
         :add(nn.ConcatTable()
         :add(s)
         :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
    end

    -- Creates count residual blocks with specified number of features
    local function layer(block, features, count, stride)
        local s = nn.Sequential()
        for i=1,count do
            s:add(block(features, i == 1 and stride or 1))
        end
        return s
    end

    local cfg = {
        [18]  = {{2, 2, 2, 2}, {64,128,256,512,512}, basicblock},
        [34]  = {{3, 4, 6, 3}, {64,128,256,512,512}, basicblock},
        [50]  = {{3, 4, 6, 3}, {64,128,256,512,2048}, bottleneck},
        [101] = {{3, 4, 23, 3}, {64,128,256,512,2048}, bottleneck},
        [152] = {{3, 8, 36, 3}, {64,128,256,512,2048}, bottleneck},
    }

    assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
    local def, nFeatures, block = table.unpack(cfg[depth])
    
    print(' | ResNet-' .. depth)
    
    local final_pool_size = 7
    if (resnet_options.input_size[2] == 299 and resnet_options.input_size[3] == 299)
        or (resnet_options.input_size[2] == 320 and resnet_options.input_size[3] == 320) then
        final_pool_size = 10
    elseif resnet_options.input_size[2] == 256 and resnet_options.input_size[3] == 256 then
        final_pool_size = 8      
    end
    
    local first_conv_stride = 2
    if resnet_options.input_size[2] == 448 and resnet_options.input_size[3] == 448 then
        first_conv_stride = 4
    end
    
    iChannels = nFeatures[1]
    local fe_CNN = nn.Sequential()
    
    fe_CNN:add(Convolution(resnet_options.input_size[1],nFeatures[1],7,7,first_conv_stride,first_conv_stride,3,3))
    fe_CNN:add(SBatchNorm(nFeatures[1]))
    fe_CNN:add(ReLU(true))
    fe_CNN:add(Max(3,3,2,2,1,1))
    fe_CNN:add(layer(block, nFeatures[1], def[1]))
    fe_CNN:add(layer(block, nFeatures[2], def[2], 2))
    fe_CNN:add(layer(block, nFeatures[3], def[3], 2))
    fe_CNN:add(layer(block, nFeatures[4], def[4], 2))
    if def[5] then fe_CNN:add(layer(block, nFeatures[5], def[5], 2)) end
    fe_CNN:add(Avg(final_pool_size, final_pool_size, 1, 1))

    fe_CNN:get(1).gradInput = nil -- The gradInput of the first module is not used for training since nothing comes before it. 

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
    

--   if opt.cudnn == 'deterministic' then
--      model:apply(function(m)
--         if m.setMode then m:setMode(1,1,1) end
--      end)
--   end

    
    return makeDataParallel(net, resnet_options.numGPU),nFeatures[5]
end