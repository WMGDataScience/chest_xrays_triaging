  function createVGGLikeBigModel(VGG_options)   
  
    local function createConv_BatchNorm_ReLu(in_feat,out_feat)
        local result = nn.Sequential()
        local std_epsilon = 0.0010000000475
        result:add(cudnn.SpatialConvolution(in_feat, out_feat, 3, 3,1,1))
        result:add(cudnn.SpatialBatchNormalization(out_feat, std_epsilon, nil, true))
        result:add(cudnn.ReLU(true))
        return result
    end

    local function createClassifier(nfeat,nout)
        local classifier = nn.Sequential()
        classifier:add(nn.View(nfeat))   
        classifier:add(nn.Dropout(0.2))
        classifier:add(nn.Linear(nfeat, nout))
        classifier:add(VGG_options.last_layer())
        return classifier
    end
     
    local nfeaturemaps = {16,32,64,128,256,512}
    local nhiddenl = {1024,1024}
    -- input dimensions
    local nfeats = VGG_options.input_size[1]


    local net = nn.Sequential()
    
    local fe_CNN = nn.Sequential()
    
    -- stage 1 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(cudnn.SpatialConvolution(nfeats, nfeaturemaps[1], 3, 3,2,2)) -- 3x407x407 -> 16x201x201
    fe_CNN:add(cudnn.SpatialBatchNormalization(nfeaturemaps[1], 0.0010000000475, nil, true))    
    fe_CNN:add(cudnn.ReLU())
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[1], nfeaturemaps[1]))-- 16x201x201 -> 16x199x199
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[1], nfeaturemaps[1]))-- 16x199x199 -> 16x197x197        
    fe_CNN:add(cudnn.SpatialMaxPooling(3,3,2,2)) -- 16x197x197 -> 16x96x96
    
    -- stage 2 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[1], nfeaturemaps[2])) -- 16x96x96 -> 32x94x94
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[2], nfeaturemaps[2])) -- 32x94x94 -> 32x92x92    
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[2], nfeaturemaps[2])) -- 32x92x92 -> 32x90x90      
    fe_CNN:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- 32x90x90 -> 32x45x45
    
    -- stage 3 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[2], nfeaturemaps[3]))
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[3], nfeaturemaps[3]))              
    fe_CNN:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- 64x41x41 -> 64x20x20
    
    -- stage 4 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[3], nfeaturemaps[4]))
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[4], nfeaturemaps[4]))               
    fe_CNN:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- 128x16x16 -> 128x8x8    

    -- stage 4 : filter bank -> squashing -> L2 pooling
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[4], nfeaturemaps[5]))
    fe_CNN:add(createConv_BatchNorm_ReLu(nfeaturemaps[5], nfeaturemaps[5]))              
    fe_CNN:add(cudnn.SpatialAveragePooling(4,4,1,1)) -- 256x4x4 -> 256x1x1
--    
    fe_CNN:cuda()
    
    fe_CNN = makeDataParallel(fe_CNN, opt.numGPU) -- defined in util.lua
    
    net:add(fe_CNN)
    local classifiers = nn.Concat(2)
    for i = 1,VGG_options.num_classifiers do
        classifiers:add(createClassifier(nfeaturemaps[5],2)) 
    end
    if VGG_options.predict_age then     
        local classifier = nn.Sequential() 
        classifier:add(nn.View(nfeaturemaps[5]))   
        classifier:add(nn.Dropout(0.2))
        classifier:add(nn.Linear(nfeaturemaps[5],nhiddenl[1]))  
        classifier:add(cudnn.ReLU(true))    
        classifier:add(nn.Dropout(0.2))
        classifier:add(nn.Linear(nhiddenl[1], 1))  
        classifiers:add(classifier)         
    end    
    net:add(classifiers:cuda())
    return net,nfeaturemaps[5]
 end     