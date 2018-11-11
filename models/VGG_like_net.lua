



function createVGGLikeModel(VGG_options) 
    
    local function createClassifier(nfeat,nout)
        local classifier = nn.Sequential()
        classifier:add(nn.View(nfeat))   
        classifier:add(nn.Dropout(0.2))
        classifier:add(nn.Linear(nfeat, nout))
        classifier:add(VGG_options.last_layer())
        return classifier
    end 
        
    local nfeaturemaps = {64,128,256,512}
    local nhiddenl = {1024,1024}
    -- input dimensions
    local nfeats = VGG_options.input_size[1]

    -- a typical modern convolution network (conv+relu+pool) in VGG style
    local net = nn.Sequential()
    
    local fe_CNN = nn.Sequential()
    
    -- stage 1 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(cudnn.SpatialConvolution(nfeats,16,7,7,2,2,3,3)) 
    fe_CNN:add(cudnn.SpatialBatchNormalization(16, 0.0010000000475, nil, true))  
    fe_CNN:add(cudnn.ReLU())
    fe_CNN:add(cudnn.SpatialConvolution(16, nfeaturemaps[1], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[1], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[1], nfeaturemaps[1], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[1], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())      
    fe_CNN:add(cudnn.SpatialMaxPooling(2,2,2,2))
    
    -- stage 2 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[1], nfeaturemaps[2], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[2], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[2], nfeaturemaps[2], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[2], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())         
    fe_CNN:add(cudnn.SpatialMaxPooling(2,2,2,2))
    
    -- stage 3 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[2], nfeaturemaps[3], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[3], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[3], nfeaturemaps[3], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[3], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())   
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[3], nfeaturemaps[3], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[3], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())              
    fe_CNN:add(cudnn.SpatialMaxPooling(2,2,2,2))
    
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[3], nfeaturemaps[4], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[4], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[4], nfeaturemaps[4], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[4], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())   
    fe_CNN:add(cudnn.SpatialConvolution(nfeaturemaps[4], nfeaturemaps[4], 3, 3,1,1,1,1))
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[4], 0.0010000000475, nil, true)) 
    fe_CNN:add(cudnn.ReLU())              
    fe_CNN:add(cudnn.SpatialMaxPooling(2,2,2,2))    
    fe_CNN:cuda()
    
    fe_CNN = makeDataParallel(fe_CNN, VGG_options.numGPU) -- defined in util.lua
    
    -- stage 4 : standard 3-layer neural network
    local dim_after_CNN = math.floor(VGG_options.input_size[2]/32) * math.floor(VGG_options.input_size[3]/32) -- 3 max pool then 2³ reduction
    
    if VGG_options.classifier_type == "multi" then
        net:add(fe_CNN)
        local classifiers = nn.Concat(2)
        for i = 1,VGG_options.num_classifiers do
            classifiers:add(createClassifier(nfeaturemaps[4]*dim_after_CNN,2)) 
        end
        net:add(fe_CNN):add(classifiers:cuda())
    elseif VGG_options.classifier_type == "single" then 
        local classifier = nn.Sequential()
        classifier:add(nn.View(nfeaturemaps[3]*dim_after_CNN)) 
        classifier:add(nn.Dropout(0.5))
        classifier:add(nn.Linear(nfeaturemaps[3]*dim_after_CNN, nhiddenl[1]))
        classifier:add(nn.ReLU())
        classifier:add(nn.Linear(nhiddenl[1], nhiddenl[2]))
        classifier:add(nn.ReLU())     
        classifier:add(nn.Linear(nhiddenl[2], VGG_options.nOutputs))
        classifier:add(nn.LogSoftMax())
        net:add(fe_CNN):add(classifier:cuda())
    end
    

    return net,nfeaturemaps[3]*dim_after_CNN
end    