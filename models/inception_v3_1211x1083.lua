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

    local function shortcut(nInputPlane, nOutputPlane, stride,type2)
        if nInputPlane ~= nOutputPlane then
            if stride == 2 then
                -- Strided, zero-padded identity shortcut
                if type2 then                   
                    return nn.Sequential()
                     :add(nn.SpatialAveragePooling( 3, 3, 2, 2))
                     :add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))                
                else
                    return nn.Sequential()
                     :add(nn.SpatialAveragePooling( 1, 1, 2, 2,-2,-2))
                     :add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))
                end
            else
                return nn.Sequential()
                 :add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))            
            end
        else
            return nn.Identity()
        end
    end
    
    local function custom_block(inSize,middleSize,outSize,skip_connection)
        local block = nn.Sequential()
        block:add(SpatialConvolution(inSize, middleSize, 3, 3, 1, 1)) 
        block:add(SpatialBatchNormalization(middleSize, std_epsilon, nil, true))
        block:add(ReLU(true))      
        block:add(SpatialConvolution(middleSize, outSize, 3, 3, 1, 1, 1, 1)) 
        block:add(SpatialBatchNormalization(outSize, std_epsilon, nil, true))
        block:add(ReLU(true))
        block:add(SpatialMaxPooling(3, 3, 2, 2)) 
        
        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(block)
                     :add(shortcut(inSize, outSize, 2)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return block
        end  
    end
    
    local function Stem()
      local stem = nn.Sequential()

      stem:add(SpatialConvolution(inception_options.input_size[1], 32, 3, 3, 2, 2)) -- 32x605x541 
      stem:add(SpatialBatchNormalization(32, std_epsilon, nil, true))
      stem:add(ReLU(true))
              
      stem:add(custom_block(32,40,48,true)) -- 48x301x269 
      stem:add(custom_block(48,80,128,true)) -- 128x149x133
      stem:add(custom_block(128,192,256,true)) -- 256x73x65
      stem:add(custom_block(256,320,352,true)) -- 352x35x31         
   
      return stem
    end

    local function Pre_Inception(in_dim,middle_dim,out_dim,skip_connection)
        local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), 
              SpatialConvolution(in_dim, out_dim, 1, 1), 
              SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
              ReLU(true)                
            }
          ),
          Tower(
           {
             SpatialConvolution(in_dim, out_dim, 1, 1, 1, 1),
             SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
             ReLU(true) 
           }
          ),   
          Tower(
            {
              SpatialConvolution(in_dim, middle_dim, 1, 1, 1, 1), 
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(middle_dim, out_dim, 5, 5, 1, 1, 2, 2), 
              SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
              ReLU(true)                
            }
          ),
          Tower(
            {
              SpatialConvolution(in_dim, middle_dim, 1, 1, 1, 1), 
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(middle_dim, out_dim, 3, 3, 1, 1, 1, 1), 
              SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(out_dim, out_dim, 3, 3, 1, 1, 1, 1), 
              SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
              ReLU(true)                 
            }
          )
        }
      ) -- 256x32x32

        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(inception)
                     :add(shortcut(in_dim, out_dim*4, 1)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return inception
        end 
    end

    local function Pre_Reduction(in_dim,middle_dim1,middle_dim2,out_dim,skip_connection)
        local inception = FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2,-1,-1), 
          Tower(
           {
             SpatialConvolution(in_dim, middle_dim2, 3, 3, 2, 2),
             SpatialBatchNormalization(middle_dim2, std_epsilon, nil, true),
             ReLU(true),
             SpatialConvolution(middle_dim2, out_dim, 3, 3, 1, 1), 
             SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
             ReLU(true)
           }
          ),      
          Tower(
            {
              SpatialConvolution(in_dim, middle_dim1, 1, 1, 1, 1), 
              SpatialBatchNormalization(middle_dim1, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(middle_dim1, middle_dim2, 3, 3, 1, 1, 0, 0), 
              SpatialBatchNormalization(middle_dim2, std_epsilon, nil, true),
              ReLU(true),                 
              SpatialConvolution(middle_dim2, out_dim, 3, 3, 2, 2), 
              SpatialBatchNormalization(out_dim, std_epsilon, nil, true),
              ReLU(true)               
            }
          )
        }) 
        
        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(inception)
                     :add(shortcut(in_dim, out_dim * 2 + in_dim, 2)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return inception
        end 

    end

    local function Stem_version2(skip_connection)
      local stem = nn.Sequential()

      stem:add(SpatialConvolution(inception_options.input_size[1], 28, 3, 3, 2, 2)) -- 28x605x541 
      stem:add(SpatialBatchNormalization(28, std_epsilon, nil, true))
      stem:add(ReLU(true))
              
      stem:add(custom_block(28,32,36,skip_connection)) -- 32x301x269 
      
      stem:add(Pre_Inception(36,8,12,skip_connection)) -- 48x301x269
      stem:add(Pre_Reduction(48,12,18,24,skip_connection)) -- 96x149x133
      
      stem:add(Pre_Inception(96,16,24,skip_connection)) -- 96x149x133
      stem:add(Pre_Inception(96,20,28,skip_connection)) -- 112x149x133
      stem:add(Pre_Reduction(112,24,32,40,skip_connection)) -- 192x73x65
            
      stem:add(Pre_Inception(192,32,48,skip_connection)) -- 192x73x65
      stem:add(Pre_Inception(192,40,52,skip_connection)) -- 208x73x65 
      stem:add(Pre_Reduction(208,48,56,72,skip_connection)) -- 352x35x31      
   
      return stem
    end

    local function Reduction_A(skip_connection)
      local inception = FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2), -- 416x17x15
          Tower(
           {
             SpatialConvolution(416, 416, 3, 3, 2, 2), -- 416x17x15
             SpatialBatchNormalization(416, std_epsilon, nil, true),
             ReLU(true)
           }
          ),      
          Tower(
            {
              SpatialConvolution(416, 192, 1, 1, 1, 1), -- 192x35x31
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(192, 224, 3, 3, 1, 1, 1, 1), -- 224x35x31
              SpatialBatchNormalization(224, std_epsilon, nil, true),
              ReLU(true),                 
              SpatialConvolution(224, 256, 3, 3, 2, 2), -- 256x17x15
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true)               
            }
          )
        }
      ) -- 1088x17x15
        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(inception)
                     :add(shortcut(416, 416 * 2 + 256, 2,true)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return inception
        end 
    end
    
    local function Inception_A(in_dim,n,skip_connection)
        local out_dim = nil
        
        if in_dim == 352 then
            out_dim = 96
        elseif in_dim == 384 or in_dim == 416 then
            out_dim = 128
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
             SpatialConvolution(in_dim, 96, 1, 1, 1, 1),-- 64x34x34
             SpatialBatchNormalization(96, std_epsilon, nil, true),
             ReLU(true) 
           }
          ),   
          Tower(
            {
              SpatialConvolution(in_dim, 64, 1, 1, 1, 1), -- 48x34x34
              SpatialBatchNormalization(64, std_epsilon, nil, true),
              ReLU(true),               
              SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2), -- 64x34x34
              SpatialBatchNormalization(96, std_epsilon, nil, true),
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

        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(inception)
                     :add(shortcut(in_dim, 96 * 3 + out_dim, 1)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return inception
        end
    end
        

    
    local function Reduction_B(skip_connection)
      local inception = FilterConcat(
        {
          SpatialMaxPooling(3, 3, 2, 2), -- 1088x8x8
          Tower(
            {
              SpatialConvolution(1088, 192, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(192, std_epsilon, nil, true),
              ReLU(true),             
              SpatialConvolution(192, 224, 3, 3, 2, 2), -- 192x8x8
              SpatialBatchNormalization(224, std_epsilon, nil, true),
              ReLU(true)              
            }
          ),
          Tower(
            {
              SpatialConvolution(1088, 256, 1, 1, 1, 1), -- 256x17x17
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true),              
              SpatialConvolution(256, 256, 7, 1, 1, 1, 3, 0), -- 256x17x17
              SpatialBatchNormalization(256, std_epsilon, nil, true),
              ReLU(true),                  
              SpatialConvolution(256, 288, 1, 7, 1, 1, 0, 3), -- 320x17x17
              SpatialBatchNormalization(288, std_epsilon, nil, true),
              ReLU(true),                  
              SpatialConvolution(288, 288, 3, 3, 2, 2), -- 320x8x8
              SpatialBatchNormalization(288, std_epsilon, nil, true),
              ReLU(true)                 
            }
          )
        }
      ) -- 1600x8x7
      -- 1024 ifms, 1536 ofms
        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(inception)
                     :add(shortcut(1088,288 + 224 + 1088, 2,true)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return inception
        end
    end

    local function Inception_B(num,skip_connection)
        local middle_dim = 128
        if num == 1 then
            middle_dim = 160
        elseif num == 2 or num == 3 then
            middle_dim = 192           
        elseif num == 4 then
            middle_dim = 224
        end
            
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1024x17x15
              SpatialConvolution(1088, 128, 1, 1), -- 128x17x17
              SpatialBatchNormalization(128, std_epsilon, nil, true),
              ReLU(true)                 
            }
          ),
          Tower(
            {
              SpatialConvolution(1088, 384, 1, 1, 1, 1), -- 192x17x17   
              SpatialBatchNormalization(384, std_epsilon, nil, true),
              ReLU(true)
            }
          ),
          Tower(
            {
              SpatialConvolution(1088, middle_dim, 1, 1, 1, 1), -- 192x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(middle_dim, middle_dim, 7, 1, 1, 1, 3, 0), -- 224x17x17
              SpatialBatchNormalization(middle_dim, std_epsilon, nil, true),
              ReLU(true),                
              SpatialConvolution(middle_dim, 288, 1, 7, 1, 1, 0, 3), -- 256x17x17
              SpatialBatchNormalization(288, std_epsilon, nil, true),
              ReLU(true)                 
            }
          ),
          Tower(
            {
              SpatialConvolution(1088, middle_dim, 1, 1, 1, 1), -- 192x17x17
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
              SpatialConvolution(middle_dim, 288, 1, 7, 1, 1, 0, 3), -- 256x17x17
              SpatialBatchNormalization(288, std_epsilon, nil, true),
              ReLU(true)              
            }
          )
        }
      ) 
        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(inception)
                     :add(shortcut(1088,288 * 2 + 384 + 128, 1)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return inception
        end
    end
    
    local function Inception_C(in_dim,skip_connection)
      local inception = FilterConcat(
        {
          Tower(
            {
              SpatialAveragePooling(3, 3, 1, 1, 1, 1), -- 1536x8x7
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
      ) -- 2048x8x7

        if skip_connection then
            local result = nn.Sequential()
            result:add(nn.ConcatTable()
                     :add(inception)
                     :add(shortcut(in_dim,384 * 4 + 320 + 192, 1)))
                     :add(nn.CAddTable(true))
                     :add(ReLU(true))  
                             
            return result
        else
            return inception
        end
    end
    
    -- Overall schema of the Inception-v4 network
    local fe_CNN = nn.Sequential()
    
    print("-- Stem")
    fe_CNN:add(Stem())           -- 1x1211x1083 ==> 352x35x31
--    fe_CNN:add(Stem_version2(false))     -- 1x1211x1083 ==> 352x35x31
    
    print("-- Inception-A x 3")
    fe_CNN:add(Inception_A(352,false))  -- 352x35x31 ==> 384x35x31
    fe_CNN:add(Inception_A(384,false))  -- 384x35x31 ==> 416x35x31
    fe_CNN:add(Inception_A(416,false))  -- 416x35x31 ==> 416x35x31
    
    print("-- Reduction-A")
    fe_CNN:add(Reduction_A(false))    -- 416x35x31 ==> 1088x17x15
    
    print("-- Inception-B x 4")
    for i=1,4 do
        fe_CNN:add(Inception_B(i,false))  -- 1088x17x15 ==> 1088x17x15
    end  
    print("-- Reduction-B")
    fe_CNN:add(Reduction_B(false))    -- 1088x17x15 ==> 1600x8x7
    
    print("-- Inception-C x 2")
    fe_CNN:add(Inception_C(1600,false))  -- 1600x8x7 ==> 2048x8x7
    fe_CNN:add(Inception_C(2048,false)) -- 2048x8x7 ==> 2048x8x7



    print("-- Average Pooling")
    fe_CNN:add(nn.SpatialAveragePooling(8,7)) -- 2048x8x7 ==> 2048x1x1
--    fe_CNN:add(SpatialMaxPooling(17,15))
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

        net:add(fe_CNN):add(classifiers)
        
    elseif inception_options.classifier_type == "single" then    
        local classifier = nn.Sequential()
        classifier:add(nn.View(last_layer_dim))
        print("-- Dropout")
        classifier:add(nn.Dropout(0.2))
        print("-- Fully Connected")
        classifier:add(nn.Linear(last_layer_dim, inception_options.nOutputs))  -- last_layer_dim ==> nOutputs
        if inception_options.last_layer then
            print("-- SoftMax")
            classifier:add(nn.LogSoftMax())
        end

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

    
    net = makeDataParallel(net, inception_options.numGPU)
    
    return net,last_layer_dim
end
