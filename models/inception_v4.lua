function createInceptionv4Model(inception_options)

    local SpatialConvolution = cudnn.SpatialConvolution
    local SpatialMaxPooling = cudnn.SpatialMaxPooling
    local SpatialAveragePooling = cudnn.SpatialAveragePooling
    local ReLU = cudnn.ReLU
    local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
    if opt.tensor_precision == 'half' then
        SpatialBatchNormalization = nn.SpatialBatchNormalization
    end
        
    local std_epsilon = 0.0010000000475
    
    local function Tower(layers)
      local tower = nn.Sequential()
      for i=1,#layers do
        tower:add(layers[i])
      end
      return tower
    end
    
    local function FilterConcat(towers)
      local concat = nn.DepthConcat(2)
      for i=1,#towers do
        concat:add(towers[i])
      end
      return concat
    end
    
    local function Stem()
      local stem = nn.Sequential()

      stem:add(SpatialConvolution(inception_options.input_size[1], 32, 3, 3, 2, 2)) -- 32x149x149
      stem:add(SpatialBatchNormalization(32, std_epsilon, nil, true))
      stem:add(ReLU(true))
      stem:add(SpatialConvolution(32, 32, 3, 3, 1, 1)) -- 32x147x147
      stem:add(SpatialBatchNormalization(32, std_epsilon, nil, true))
      stem:add(ReLU(true))      
      stem:add(SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) -- 64x147x147
      stem:add(SpatialBatchNormalization(64, std_epsilon, nil, true))
      stem:add(ReLU(true))      
      stem:add(FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2), -- 64x73x73
          Tower(
           {
             SpatialConvolution(64, 96, 3, 3, 2, 2),-- 96x73x73 
             SpatialBatchNormalization(96, std_epsilon, nil, true),
             ReLU(true) 
           }
          )      
        }
      )) -- 160x73x73
      stem:add(FilterConcat(
        {
          Tower(
            {
              SpatialConvolution(160, 64, 1, 1, 1, 1), -- 64x73x73
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(64, 96, 3, 3, 1, 1), -- 96x71x71
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true)                
            }
          ),
          Tower(
            {
              SpatialConvolution(160, 64, 1, 1, 1, 1), -- 64x73x73
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(64, 64, 1, 7, 1, 1, 0, 3), -- 64x73x73
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(64, 64, 7, 1, 1, 1, 3, 0), -- 64x73x73
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(64, 96, 3, 3, 1, 1), -- 96x71x71
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true)              
            }
          )
        }
      )) -- 192x71x71
      stem:add(FilterConcat(
        {
          Tower(
           {
             SpatialConvolution(192, 192, 3, 3, 2, 2),-- 192x35x35 
             SpatialBatchNormalization(192, std_epsilon, nil, true),
             ReLU(true),  
           }
          ),   
          SpatialMaxPooling(2, 2, 2, 2) -- 192x35x35
        }
      )) -- 384x35x35
      return stem
    end

    
    local function Reduction_A()
      local inception = FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2), -- 384x17x17
          Tower(
           {
             SpatialConvolution(384, 384, 3, 3, 2, 2), -- 384x17x17
             SpatialBatchNormalization(384, std_epsilon, nil, true),
             ReLU(true)
           }
          ),      
          Tower(
            {
              SpatialConvolution(384, 192, 1, 1, 1, 1), -- 192x35x35
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(192, 224, 3, 3, 1, 1, 1, 1), -- 224x35x35
              SpatialBatchNormalization(224, std_epsilon, nil, true),
              ReLU(true),                 
              SpatialConvolution(224, 256, 3, 3, 2, 2), -- 256x17x17
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true)               
            }
          )
        }
      ) -- 1024x17x17
      -- 384 ifms, 1024 ofms
      return inception
    end
    
    local function Inception_A()
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3), -- 384x35x35
              SpatialConvolution(384, 96, 1, 1, 1, 1), -- 96x35x35
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true)                
            }
          ),
          Tower(
           {
             SpatialConvolution(384, 96, 1, 1, 1, 1),-- 96x35x35
             SpatialBatchNormalization(96, std_epsilon, nil, true),
             ReLU(true) 
           }
          ),   
          Tower(
            {
              SpatialConvolution(384, 64, 1, 1, 1, 1), -- 64x35x35
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1), -- 96x35x35
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true)                
            }
          ),
          Tower(
            {
              SpatialConvolution(384, 64, 1, 1, 1, 1), -- 64x35x35
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1), -- 96x35x35
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1), -- 96x35x35
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true)                 
            }
          )
        }
      ) -- 384x35x35
      -- 384 ifms / ofms
      return inception
    end
        

    
    local function Reduction_B()
      local inception = FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2), -- 1024x8x8
          Tower(
            {
              SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),             
              SpatialConvolution(192, 192, 3, 3, 2, 2), -- 192x8x8
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true)              
            }
          ),
          Tower(
            {
              SpatialConvolution(1024, 256, 1, 1, 1, 1), -- 256x17x17
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(256, 256, 7, 1, 1, 1, 3, 0), -- 256x17x17
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true),                  
              SpatialConvolution(256, 320, 1, 7, 1, 1, 0, 3), -- 320x17x17
              SpatialBatchNormalization(320, std_epsilon, nil, true),
              ReLU(true),                  
              SpatialConvolution(320, 320, 3, 3, 2, 2), -- 320x8x8
              SpatialBatchNormalization(320, std_epsilon, nil, true),
              ReLU(true)                 
            }
          )
        }
      ) -- 1536x8x8
      -- 1024 ifms, 1536 ofms
      return inception
    end

    local function Inception_B()
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1024x17x17
              SpatialConvolution(1024, 128, 1, 1, 1, 1), -- 128x17x17
              SpatialBatchNormalization(128, std_epsilon, nil, true),
              ReLU(true)                 
            }
          ),
          Tower(
            {
              SpatialConvolution(1024, 384, 1, 1, 1, 1), -- 384x17x17   
              SpatialBatchNormalization(384, std_epsilon, nil, true),
              ReLU(true)
            }
          ),
          Tower(
            {
              SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(192, 224, 7, 1, 1, 1, 3, 0), -- 224x17x17
              SpatialBatchNormalization(224, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(224, 256, 1, 7, 1, 1, 0, 3), -- 256x17x17
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true)                 
            }
          ),
          Tower(
            {
              SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(192, 192, 7, 1, 1, 1, 3, 0), -- 192x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(192, 224, 1, 7, 1, 1, 0, 3), -- 224x17x17
              SpatialBatchNormalization(224, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(224, 224, 7, 1, 1, 1, 3, 0), -- 224x17x17
              SpatialBatchNormalization(224, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(224, 256, 1, 7, 1, 1, 0, 3), -- 256x17x17
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true)              
            }
          )
        }
      ) -- 1024x17x17
      -- 1024 ifms / ofms
      return inception
    end
    
    local function Inception_C()
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1536x8x8
              SpatialConvolution(1536, 256, 1, 1, 1, 1), -- 256x8x8
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true)               
            }
          ),
          Tower(
            {
              SpatialConvolution(1536, 256, 1, 1, 1, 1),-- 256x8x8     
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true)
            }
          ),    
          Tower(
            {
              SpatialConvolution(1536, 384, 1, 1, 1, 1), -- 384x8x8
              SpatialBatchNormalization(384, std_epsilon, nil, true),
              ReLU(true),               
              FilterConcat(
                {
                  Tower(
                    {
                      SpatialConvolution(384, 256, 3, 1, 1, 1, 1, 0),-- 256x8x8 
                      SpatialBatchNormalization(256, std_epsilon, nil, true),
                      ReLU(true)
                    }
                  ),
                  Tower(
                    {                                  
                      SpatialConvolution(384, 256, 1, 3, 1, 1, 0, 1),-- 256x8x8 
                      SpatialBatchNormalization(256, std_epsilon, nil, true),
                      ReLU(true)
                    }
                  )          
                }
              ) -- 512x8x8
            }
          ),
          Tower(
            {
              SpatialConvolution(1536, 384, 1, 1, 1, 1), -- 384x8x8#
              SpatialBatchNormalization(384, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(384, 448, 3, 1, 1, 1, 1, 0), -- 448x8x8
              SpatialBatchNormalization(448, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(448, 512, 1, 3, 1, 1, 0, 1), -- 512x8x8
              SpatialBatchNormalization(512, std_epsilon, nil, true),
              ReLU(true),               
              FilterConcat(
                {
                  Tower(
                    {
                      SpatialConvolution(512, 256, 3, 1, 1, 1, 1, 0),-- 256x8x8
                      SpatialBatchNormalization(256, std_epsilon, nil, true),
                      ReLU(true)
                    }
                  ), 
                  Tower(
                    {                              
                      SpatialConvolution(512, 256, 1, 3, 1, 1, 0, 1),-- 256x8x8
                      SpatialBatchNormalization(256, std_epsilon, nil, true),
                      ReLU(true) 
                    }
                  )
                }
              ) -- 512x8x8
            }
          )
        }
      ) -- 1536x8x8
      -- 1536 ifms / ofms
      return inception
    end
    
    -- Overall schema of the Inception-v4 network
    local fe_CNN = nn.Sequential()
    
    print("-- Stem")
    fe_CNN:add(Stem())           -- 3x299x299 ==> 384x35x35
    print("-- Inception-A x 4")
    for i=1,4 do
      fe_CNN:add(Inception_A())  -- 384x35x35 ==> 384x35x35
    end
    print("-- Reduction-A")
    fe_CNN:add(Reduction_A())    -- 384x35x35 ==> 1024x17x17
    print("-- Inception-B x 7")
    for i=1,7 do
      fe_CNN:add(Inception_B())  -- 1024x17x17 ==> 1024x17x17
    end
    print("-- Reduction-B")
    fe_CNN:add(Reduction_B())    -- 1024x17x17 ==> 1536x8x8
    print("-- Inception-C x 3")
    for i=1,3 do
      fe_CNN:add(Inception_C())  -- 1536x8x8 ==> 1536x8x8
    end
    
    local final_pool_size = 0
    if inception_options.input_size[2] == 299 and inception_options.input_size[3] == 299 then
        final_pool_size = 8
    elseif inception_options.input_size[2] == 256 and inception_options.input_size[3] == 256 then
        final_pool_size = 6
    else
        error('unknown input size: ' .. inception_options.input_size[2] .."x"..inception_options.input_size[3])    
    end
    
    print("-- Average Pooling")
    fe_CNN:add(nn.SpatialAveragePooling(final_pool_size, final_pool_size)) -- 1536x8x8 ==> 1536x1x1
    fe_CNN:cuda()
    
        
    
    local net = nn.Sequential()
    if inception_options.classifier_type == "multi" then
        local classifiers = nn.Concat(2)
        for i = 1,inception_options.num_classifiers do
            local classifier = nn.Sequential()
            classifier:add(nn.View(1536))
            classifier:add(nn.Dropout(0.2))
            classifier:add(nn.Linear(1536, inception_options.nOutputs))  -- 1536 ==> nOutputs
            if inception_options.last_layer then
                classifier:add(inception_options.last_layer())
            end
            classifiers:add(classifier) 
        end
        classifiers:cuda() 
        net:add(fe_CNN):add(classifiers)
        
    elseif inception_options.classifier_type == "single" then    
        local classifier = nn.Sequential()
        classifier:add(nn.View(1536))
        print("-- Dropout")
        classifier:add(nn.Dropout(0.2))
        print("-- Fully Connected")
        classifier:add(nn.Linear(1536, inception_options.nOutputs))  -- 1536 ==> nOutputs
        print("-- SoftMax")
        classifier:add(nn.LogSoftMax())
        classifier:cuda()
        net:add(fe_CNN):add(classifier)
    end    
    
    net = makeDataParallel(net, inception_options.numGPU)
    
    return net,1536
end