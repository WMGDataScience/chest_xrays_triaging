function createInceptionv3Model(inception_options)

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
      stem:add(SpatialMaxPooling(3, 3, 2, 2)) -- 80x73x73
      stem:add(SpatialConvolution(64, 80, 1, 1, 1, 1)) -- 80x73x73
      stem:add(SpatialBatchNormalization(80, std_epsilon, nil, true))
      stem:add(ReLU(true))            
      stem:add(SpatialConvolution(80, 192, 3, 3, 1, 1)) -- 192x71x71
      stem:add(SpatialBatchNormalization(192, std_epsilon, nil, true))
      stem:add(ReLU(true))
      stem:add(SpatialMaxPooling(3, 3, 2, 2)) -- 192x35x35
      
      return stem
    end

    
    local function Reduction_A()
      local inception = FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2), -- 384x17x17
          Tower(
           {
             SpatialConvolution(288, 384, 3, 3, 2, 2), -- 384x17x17
             SpatialBatchNormalization(384, std_epsilon, nil, true),
             ReLU(true)
           }
          ),      
          Tower(
            {
              SpatialConvolution(288, 64, 1, 1, 1, 1), -- 192x35x35
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1), -- 224x35x35
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true),                 
              SpatialConvolution(96, 96, 3, 3, 2, 2), -- 256x17x17
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true)               
            }
          )
        }
      ) -- 1024x17x17
      -- 384 ifms, 1024 ofms
      return inception
    end
    
    local function Inception_A(in_dim)
        local out_dim = 32
        if in_dim == 192 then
            out_dim = 32
        elseif in_dim == 256 or in_dim == 288 then
            out_dim = 64
        end
        
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 192x34x34
              SpatialConvolution(in_dim, out_dim, 1, 1), -- 32x34x34
              SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
              ReLU(true)                
            }
          ),
          Tower(
           {
             SpatialConvolution(in_dim, 64, 1, 1, 1, 1),-- 64x34x34
             SpatialBatchNormalization(64, std_epsilon, nil, true),
             ReLU(true) 
           }
          ),   
          Tower(
            {
              SpatialConvolution(in_dim, 48, 1, 1, 1, 1), -- 48x34x34
              SpatialBatchNormalization(48, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(48, 64, 5, 5, 1, 1, 2, 2), -- 64x34x34
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true)                
            }
          ),
          Tower(
            {
              SpatialConvolution(in_dim, 64, 1, 1, 1, 1), -- 64x34x34
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1), -- 96x34x34
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1), -- 96x34x34
              SpatialBatchNormalization(96, std_epsilon, nil, true),
              ReLU(true)                 
            }
          )
        }
      ) -- 256x32x32

      return inception
    end
        

    
    local function Reduction_B()
      local inception = FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2), -- 1024x8x8
          Tower(
            {
              SpatialConvolution(768, 192, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),             
              SpatialConvolution(192, 320, 3, 3, 2, 2), -- 192x8x8
              SpatialBatchNormalization(320, std_epsilon, nil, true),
              ReLU(true)              
            }
          ),
          Tower(
            {
              SpatialConvolution(768, 192, 1, 1, 1, 1), -- 256x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(192, 192, 7, 1, 1, 1, 3, 0), -- 256x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),                  
              SpatialConvolution(192, 192, 1, 7, 1, 1, 0, 3), -- 320x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),                  
              SpatialConvolution(192, 192, 3, 3, 2, 2), -- 320x8x8
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true)                 
            }
          )
        }
      ) -- 1152x8x8
      -- 1024 ifms, 1536 ofms
      return inception
    end

    local function Inception_B(num)
        local middle_dim = 128
        if num == 1 then
            middle_dim = 128
        elseif num == 2 or num == 3 then
            middle_dim = 160           
        elseif num == 4 then
            middle_dim = 192
        end
            
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1024x17x17
              SpatialConvolution(768, 192, 1, 1), -- 128x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true)                 
            }
          ),
          Tower(
            {
              SpatialConvolution(768, 192, 1, 1, 1, 1), -- 192x17x17   
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true)
            }
          ),
          Tower(
            {
              SpatialConvolution(768, middle_dim, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(middle_dim, middle_dim, 7, 1, 1, 1, 3, 0), -- 224x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(middle_dim, 192, 1, 7, 1, 1, 0, 3), -- 256x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true)                 
            }
          ),
          Tower(
            {
              SpatialConvolution(768, middle_dim, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(middle_dim, middle_dim, 7, 1, 1, 1, 3, 0), -- 192x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(middle_dim, middle_dim, 1, 7, 1, 1, 0, 3), -- 224x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(middle_dim, middle_dim, 7, 1, 1, 1, 3, 0), -- 224x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(middle_dim, 192, 1, 7, 1, 1, 0, 3), -- 256x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true)              
            }
          )
        }
      ) 
      return inception
    end
    
    local function Inception_C(in_dim)
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1536x8x8
              SpatialConvolution(in_dim, 192, 1, 1, 1, 1), -- 256x8x8
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true)               
            }
          ),
          Tower(
            {
              SpatialConvolution(in_dim, 320, 1, 1, 1, 1),-- 256x8x8     
              SpatialBatchNormalization(320, std_epsilon, nil, true),
              ReLU(true)
            }
          ),    
          Tower(
            {
              SpatialConvolution(in_dim, 384, 1, 1, 1, 1), -- 384x8x8
              SpatialBatchNormalization(384, std_epsilon, nil, true),
              ReLU(true),               
              FilterConcat(
                {
                  Tower(
                    {
                      SpatialConvolution(384, 384, 3, 1, 1, 1, 1, 0),-- 256x8x8 
                      SpatialBatchNormalization(384, std_epsilon, nil, true),
                      ReLU(true)
                    }
                  ),
                  Tower(
                    {                                  
                      SpatialConvolution(384, 384, 1, 3, 1, 1, 0, 1),-- 256x8x8 
                      SpatialBatchNormalization(384, std_epsilon, nil, true),
                      ReLU(true)
                    }
                  )          
                }
              ) 
            }
          ),
          Tower(
            {
              SpatialConvolution(in_dim, 448, 1, 1, 1, 1), -- 384x8x8#
              SpatialBatchNormalization(448, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(448, 384, 3, 1, 1, 1, 1, 0), -- 448x8x8
              SpatialBatchNormalization(384, std_epsilon, nil, true),
              ReLU(true),                             
              FilterConcat(
                {
                  Tower(
                    {
                      SpatialConvolution(384, 384, 3, 1, 1, 1, 1, 0),-- 256x8x8
                      SpatialBatchNormalization(384, std_epsilon, nil, true),
                      ReLU(true)
                    }
                  ), 
                  Tower(
                    {                              
                      SpatialConvolution(384, 384, 1, 3, 1, 1, 0, 1),-- 256x8x8
                      SpatialBatchNormalization(384, std_epsilon, nil, true),
                      ReLU(true) 
                    }
                  )
                }
              ) 
            }
          )
        }
      ) -- 2048x8x8

      return inception
    end
    
    -- Overall schema of the Inception-v4 network
    local fe_CNN = nn.Sequential()
    
    print("-- Stem")
    fe_CNN:add(Stem())           -- 3x299x299 ==> 192x35x35
    
    print("-- Inception-A x 3")
    fe_CNN:add(Inception_A(192))  -- 192x35x35 ==> 256x35x35
    fe_CNN:add(Inception_A(256))  -- 256x35x35 ==> 288x35x35
    fe_CNN:add(Inception_A(288))  -- 288x35x35 ==> 288x35x35
    
    print("-- Reduction-A")
    fe_CNN:add(Reduction_A())    -- 288x35x35 ==> 768x17x17
    
    print("-- Inception-B x 4")
    for i=1,4 do
        fe_CNN:add(Inception_B(i))  -- 768x17x17 ==> 768x17x17
    end  
    print("-- Reduction-B")
    fe_CNN:add(Reduction_B())    -- 768x17x17 ==> 1280x8x8
    
    print("-- Inception-C x 2")
    fe_CNN:add(Inception_C(1280))  -- 1280x8x8 ==> 2048x8x8
    fe_CNN:add(Inception_C(2048)) -- 2048x8x8 ==> 2048x8x8
    

    print("-- Average Pooling")
    fe_CNN:add(nn.SpatialAveragePooling(8, 8)) -- 2048x8x8 ==> 2048x1x1
    fe_CNN:cuda()
    local last_layer_dim = 2048
        
    
    local net = nn.Sequential()
    if inception_options.classifier_type == "multi" then
        local classifiers = nn.Concat(2)
        for i = 1,inception_options.num_classifiers do
            local classifier = nn.Sequential()
            classifier:add(nn.View(last_layer_dim))
            classifier:add(nn.Dropout(0.2))
            classifier:add(nn.Linear(last_layer_dim, inception_options.nOutputs))  -- last_layer_dim ==> nOutputs
            if inception_options.last_layer then
                classifier:add(inception_options.last_layer())
            end
            classifiers:add(classifier) 
        end
        classifiers:cuda() 
        net:add(fe_CNN):add(classifiers)
        
    elseif inception_options.classifier_type == "single" then    
        local classifier = nn.Sequential()
        classifier:add(nn.View(last_layer_dim))
        print("-- Dropout")
        classifier:add(nn.Dropout(0.2))
        print("-- Fully Connected")
        classifier:add(nn.Linear(last_layer_dim, inception_options.nOutputs))  -- last_layer_dim ==> nOutputs
        print("-- SoftMax")
        classifier:add(nn.LogSoftMax())
        classifier:cuda()
        net:add(fe_CNN):add(classifier)
    elseif inception_options.classifier_type == "RNN" then
        net = fe_CNN
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
    
    if inception_options.classifier_type ~= "RNN" then    
        net = makeDataParallel(net, inception_options.numGPU)
    end
    return net,last_layer_dim
end