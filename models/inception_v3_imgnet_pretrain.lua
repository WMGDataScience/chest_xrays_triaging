function createInceptionv3Model(inception_options)
    local hdf5 = require 'hdf5' 
    local last_fe_layer_dim
    local std_epsilon = 0.0010000000475
    
    local SpatialConvolution = cudnn.SpatialConvolution
    local SpatialMaxPooling = cudnn.SpatialMaxPooling
    local SpatialAveragePooling = cudnn.SpatialAveragePooling
    local ReLU = cudnn.ReLU
    local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
    if opt.tensor_precision == 'half' then
        SpatialBatchNormalization = nn.SpatialBatchNormalization
    end
    
    local final_pool_size = 8
    if inception_options.input_size[2] == 299 and inception_options.input_size[3] == 299 then
        final_pool_size = 8
    elseif inception_options.input_size[2] == 256 and inception_options.input_size[3] == 256 then
        final_pool_size = 6
--    else
--        error('unknown input size: ' .. inception_options.input_size[2] .."x"..inception_options.input_size[3])    
    end

    
    -- Adds to `net` a convolution - Batch Normalization - ReLU series
    local function ConvBN(gname, net)
        local h5f = hdf5.open(inception_options.dump_files_path.. gname..".h5", 'r')
        local strides = h5f:read("strides"):all()
        local padding = h5f:read("padding"):all()
        -- TensorFlow weight matrix is of order: height x width x input_channels x output_channels
        -- make it Torch-friendly: output_channels x input_channels x height x width
        local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
        local ich, och = weights:size(2), weights:size(1)
        local kH, kW = weights:size(3), weights:size(4)
    
        --print(string.format("%s: %d -> %d, kernel (%dx%d), strides (%d, %d), padding (%d, %d)",
        --gname, ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1]))
        
        if ich == 3 then --first conv layer
            if inception_options.input_size[1] == 1  then ich = 1 weights = weights:narrow(2,1,1) end --- checks if you want to use greyscale images
        
            if inception_options.input_size[2] == 598 and inception_options.input_size[3] == 598 then
                strides[2],strides[3] = 4,4
            end        
        end
        
        local conv = SpatialConvolution(ich, och, kW, kH, strides[3], strides[2], padding[2], padding[1])
        if inception_options.imageNetPreInit then conv.weight:copy(weights) end
        -- IMPORTANT: there are no biases in the convolutions
        conv:noBias()

        net:add(conv)
    
        local bn = SpatialBatchNormalization(och, std_epsilon, nil, true)
        local beta = h5f:read("beta"):all()
        local gamma = h5f:read("gamma"):all()
        local mean = h5f:read("mean"):all()
        local std = h5f:read("std"):all()
        if inception_options.imageNetPreInit then bn.running_mean:copy(mean) end
        if inception_options.imageNetPreInit and bn.running_std ~= nil then bn.running_std:copy(std:add(std_epsilon):sqrt():pow(-1)) end

        if inception_options.imageNetPreInit then bn.weight:copy(gamma) end
        if inception_options.imageNetPreInit then bn.bias:copy(beta) end
        net:add(bn)
    
        net:add(ReLU(true))
        h5f:close()
    end  
    
    -- Adds to `net` Spatial Pooling, either Max or Average
    local function Pool(gname, net)
        local h5f = hdf5.open(inception_options.dump_files_path.. gname..".h5", 'r')
        local strides = h5f:read("strides"):all()
        local padding = h5f:read("padding"):all()
        local ksize = h5f:read("ksize"):all()
        local ismax = h5f:read("ismax"):all()
        
        if ismax[1]==1 then
            --print(string.format("%s(Max): (%dx%d), strides (%d, %d), padding (%d, %d)",
            --gname, ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]))
            net:add( SpatialMaxPooling(ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]) )
        else
            --print(string.format("%s(Avg): (%dx%d), strides (%d, %d), padding (%d, %d)",
            --gname, ksize[3], ksize[2], strides[3], strides[2], padding[2], padding[1]))
            net:add(nn.SpatialAveragePooling(ksize[3], ksize[2],strides[3], strides[2],padding[2], padding[1]):setCountExcludePad())
        end
    end

    -- Adds to `net` Final SoftMax (and its weights) layer
    local function Softmax(net)
        local h5f = hdf5.open(inception_options.dump_files_path.."softmax.h5", 'r')
        local weights = h5f:read("weights"):all():permute(2, 1)
        --local biases = h5f:read("biases"):all()
    
        net:add(nn.View(-1):setNumInputDims(3))
        local m = nn.Linear(weights:size(2), inception_options.nOutputs)
        --if inception_options.imageNetPreInit then m.weight:copy(weights) m.bias:copy(biases) end
        net:add(m)
        if inception_options.last_layer then
            net:add(inception_options.last_layer())
        end
        
        last_fe_layer_dim = weights:size(2)
    end
    
    -- Creates an Inception Branch (SubNetwork), usually called Towers
    -- trailing_net is optional and adds trailing network at the end of the tower
    local function Tower(names, trailing_net)
        local tower = nn.Sequential()
        for i=1,#names do
            -- separate convolutions / poolings
            if string.find(names[i], "pool") then
                Pool(names[i], tower)
            else
                ConvBN(names[i], tower)
            end
        end
        if trailing_net then
            tower:add(trailing_net)
        end
      return tower
    end
    
    -- Creates the suitable branching to host towers
    local function Inception(net, towers)
        local concat = nn.DepthConcat(2)
        for i=1,#towers do
            concat:add(towers[i])
        end
        net:add(concat)
    end
    
    local net = nn.Sequential()
    local fe_CNN = nn.Sequential()
    
    --print("Adding first convolutional layers:")
    ConvBN("conv", fe_CNN)
    ConvBN("conv_1", fe_CNN)
    ConvBN("conv_2", fe_CNN)
    Pool("pool", fe_CNN)
    ConvBN("conv_3", fe_CNN)
    ConvBN("conv_4", fe_CNN)
    Pool("pool_1", fe_CNN)
    
    --print("\nAdding Inception 1:")
    Inception(fe_CNN,
      {
        Tower({"mixed_conv"}),
        Tower({"mixed_tower_conv", "mixed_tower_conv_1"}),
        Tower({"mixed_tower_1_conv", "mixed_tower_1_conv_1", "mixed_tower_1_conv_2"}),
        Tower({"mixed_tower_2_pool", "mixed_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 2:")
    Inception(fe_CNN,
      {
        Tower({"mixed_1_conv"}),
        Tower({"mixed_1_tower_conv", "mixed_1_tower_conv_1"}),
        Tower({"mixed_1_tower_1_conv", "mixed_1_tower_1_conv_1", "mixed_1_tower_1_conv_2"}),
        Tower({"mixed_1_tower_2_pool", "mixed_1_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 3:")
    Inception(fe_CNN,
      {
        Tower({"mixed_2_conv"}),
        Tower({"mixed_2_tower_conv", "mixed_2_tower_conv_1"}),
        Tower({"mixed_2_tower_1_conv", "mixed_2_tower_1_conv_1", "mixed_2_tower_1_conv_2"}),
        Tower({"mixed_2_tower_2_pool", "mixed_2_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 4:")
    Inception(fe_CNN,
      {
        Tower({"mixed_3_conv"}),
        Tower({"mixed_3_tower_conv", "mixed_3_tower_conv_1", "mixed_3_tower_conv_2"}),
        Tower({"mixed_3_pool"})
      }
    )
    
    --print("\nAdding Inception 5:")
    Inception(fe_CNN,
      {
        Tower({"mixed_4_conv"}),
        Tower({"mixed_4_tower_conv", "mixed_4_tower_conv_1", "mixed_4_tower_conv_2"}),
        Tower({"mixed_4_tower_1_conv", "mixed_4_tower_1_conv_1", "mixed_4_tower_1_conv_2", "mixed_4_tower_1_conv_3", "mixed_4_tower_1_conv_4"}),
        Tower({"mixed_4_tower_2_pool", "mixed_4_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 6:")
    Inception(fe_CNN,
      {
        Tower({"mixed_5_conv"}),
        Tower({"mixed_5_tower_conv", "mixed_5_tower_conv_1", "mixed_5_tower_conv_2"}),
        Tower({"mixed_5_tower_1_conv", "mixed_5_tower_1_conv_1", "mixed_5_tower_1_conv_2", "mixed_5_tower_1_conv_3", "mixed_5_tower_1_conv_4"}),
        Tower({"mixed_5_tower_2_pool", "mixed_5_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 7:")
    Inception(fe_CNN,
      {
        Tower({"mixed_6_conv"}),
        Tower({"mixed_6_tower_conv", "mixed_6_tower_conv_1", "mixed_6_tower_conv_2"}),
        Tower({"mixed_6_tower_1_conv", "mixed_6_tower_1_conv_1", "mixed_6_tower_1_conv_2", "mixed_6_tower_1_conv_3", "mixed_6_tower_1_conv_4"}),
        Tower({"mixed_6_tower_2_pool", "mixed_6_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 8:")
    Inception(fe_CNN,
      {
        Tower({"mixed_7_conv"}),
        Tower({"mixed_7_tower_conv", "mixed_7_tower_conv_1", "mixed_7_tower_conv_2"}),
        Tower({"mixed_7_tower_1_conv", "mixed_7_tower_1_conv_1", "mixed_7_tower_1_conv_2", "mixed_7_tower_1_conv_3", "mixed_7_tower_1_conv_4"}),
        Tower({"mixed_7_tower_2_pool", "mixed_7_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 9:")
    Inception(fe_CNN,
      {
        Tower({"mixed_8_tower_conv", "mixed_8_tower_conv_1"}),
        Tower({"mixed_8_tower_1_conv", "mixed_8_tower_1_conv_1", "mixed_8_tower_1_conv_2", "mixed_8_tower_1_conv_3"}),
        Tower({"mixed_8_pool"})
      }
    )
    

    
    local final_conv_layers =  nn.Sequential()  
    --print("\nAdding Inception 10:")
    -- Note that in the last two Inceptions we have "Inception in Inception" cases
    local incept1, incept2 = nn.Sequential(), nn.Sequential()
    Inception(incept1,
      {
        Tower({"mixed_9_tower_mixed_conv"}),
        Tower({"mixed_9_tower_mixed_conv_1"})
      }
    )
    Inception(incept2,
      {
        Tower({"mixed_9_tower_1_mixed_conv"}),
        Tower({"mixed_9_tower_1_mixed_conv_1"})
      }
    )
    Inception(final_conv_layers,
      {
        Tower({"mixed_9_conv"}),
        Tower({"mixed_9_tower_conv"}, incept1),
        Tower({"mixed_9_tower_1_conv", "mixed_9_tower_1_conv_1"}, incept2),
        Tower({"mixed_9_tower_2_pool", "mixed_9_tower_2_conv"})
      }
    )
    
    --print("\nAdding Inception 11:")
    incept1, incept2 = nn.Sequential(), nn.Sequential()
    Inception(incept1,
      {
        Tower({"mixed_10_tower_mixed_conv"}),
        Tower({"mixed_10_tower_mixed_conv_1"})
      }
    )
    Inception(incept2,
      {
        Tower({"mixed_10_tower_1_mixed_conv"}),
        Tower({"mixed_10_tower_1_mixed_conv_1"})
      }
    )
    Inception(final_conv_layers,
      {
        Tower({"mixed_10_conv"}),
        Tower({"mixed_10_tower_conv"}, incept1),
        Tower({"mixed_10_tower_1_conv", "mixed_10_tower_1_conv_1"}, incept2),
        Tower({"mixed_10_tower_2_pool", "mixed_10_tower_2_conv"})
      }
    )

    --print("\nAdding Pooling and SoftMax:")

    final_conv_layers:add(SpatialAveragePooling(final_pool_size, final_pool_size,1, 1,0,0))
    

    if inception_options.classifier_type == "multi" and inception_options.mod_flag then
        local classifier = nn.Sequential()
        Softmax(classifier)        
        final_conv_layers:add(classifier)
        local final_layers_branches = nn.Concat(2)
        for i = 1,inception_options.num_classifiers do
            final_layers_branches:add(final_conv_layers:clone())
        end
        net:add(fe_CNN):add(final_layers_branches)
    elseif inception_options.classifier_type == "multi" and not inception_options.mod_flag then
        local classifiers = nn.Concat(2)
        for i = 1,inception_options.num_classifiers do
            local classifier = nn.Sequential()
            Softmax(classifier)
            classifiers:add(classifier) 
        end
        if inception_options.predict_age then
            local classifier = nn.Sequential()
            inception_options.nOutputs = 1
            Softmax(classifier)
            classifiers:add(classifier) 
        end 
        fe_CNN:add(final_conv_layers)          
        net:add(fe_CNN):add(classifiers)
    elseif inception_options.classifier_type == "single" then
        local classifier = nn.Sequential()
        Softmax(classifier)
        fe_CNN:add(final_conv_layers)
        net:add(fe_CNN):add(classifier) 
    elseif inception_options.classifier_type == "RNN" then
        fe_CNN:add(final_conv_layers)
        return fe_CNN,last_fe_layer_dim      
    end
  
    net = makeDataParallel(net, inception_options.numGPU) 
    return net,last_fe_layer_dim
end
