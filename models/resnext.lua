--
--  Copyright (c) 2017, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNeXt model definition
--

function createResNeXtModel(resnet_options)
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

   
   -- The original bottleneck residual layer
   local function resnet_bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end
   
   -- The aggregated residual transformation bottleneck layer, Form (B)
   local function split(nInputPlane, d, c, stride)
      local cat = nn.ConcatTable()
      for i=1,c do
         local s = nn.Sequential()
         s:add(Convolution(nInputPlane,d,1,1,1,1,0,0))
         s:add(SBatchNorm(d))
         s:add(ReLU(true))
         s:add(Convolution(d,d,3,3,stride,stride,1,1))
         s:add(SBatchNorm(d))
         s:add(ReLU(true))
         cat:add(s)
      end
      return cat
   end
   
   local function resnext_bottleneck_B(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local D = math.floor(n * (resnet_options.baseWidth/64))
      local C = resnet_options.cardinality

      local s = nn.Sequential()
      s:add(split(nInputPlane, D, C, stride))
      s:add(nn.JoinTable(2))
      s:add(Convolution(D*C,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n*4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end
    
   -- The aggregated residual transformation bottleneck layer, Form (C)
   local function resnext_bottleneck_C(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local D = math.floor(n * (resnet_options.baseWidth/64))
      local C = resnet_options.cardinality

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,D*C,1,1,1,1,0,0))
      s:add(SBatchNorm(D*C))
      s:add(ReLU(true))
      s:add(Convolution(D*C,D*C,3,3,stride,stride,1,1,C))
      s:add(SBatchNorm(D*C))
      s:add(ReLU(true))
      s:add(Convolution(D*C,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n*4))

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

   local bottleneck
   if resnet_options.bottleneckType == 'resnet' then 
      bottleneck = resnet_bottleneck
      print('Deploying ResNet bottleneck block')
   elseif resnet_options.bottleneckType == 'resnext_B' then 
      bottleneck = resnext_bottleneck_B
      print('Deploying ResNeXt bottleneck block form B')
   elseif resnet_options.bottleneckType == 'resnext_C' then 
      bottleneck = resnext_bottleneck_C
      print('Deploying ResNeXt bottleneck block form C (group convolution)')
   else
      error('invalid bottleneck type: ' .. resnet_options.bottleneckType)
   end
   

  -- Configurations for ResNet:
  --  num. residual blocks, num features, residual block function
  local cfg = {
     [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
     [101] = {{3, 4, 23, 3}, 2048, bottleneck},
     [152] = {{3, 8, 36, 3}, 2048, bottleneck},
  }

  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, final_features_size, block = table.unpack(cfg[depth])
  iChannels = 64
  print(' | ResNeXt-' .. depth )

    local final_pool_size = 7
    if (resnet_options.input_size[2] == 299 and resnet_options.input_size[3] == 299) or 
        (resnet_options.input_size[2] == 320 and resnet_options.input_size[3] == 320) then
        final_pool_size = 10
    elseif resnet_options.input_size[2] == 256 and resnet_options.input_size[3] == 256 then
        final_pool_size = 8      
    end
    
    local first_conv_stride = 2
    if resnet_options.input_size[2] == 448 and resnet_options.input_size[3] == 448 then
        first_conv_stride = 4
    end

    iChannels = 64
    local fe_CNN = nn.Sequential()   
    
  -- The ResNet ImageNet model
  fe_CNN:add(Convolution(resnet_options.input_size[1],64,7,7,first_conv_stride,first_conv_stride,3,3))
  fe_CNN:add(SBatchNorm(64))
  fe_CNN:add(ReLU(true))
  fe_CNN:add(Max(3,3,2,2,1,1))
  fe_CNN:add(layer(block, 64, def[1]))
  fe_CNN:add(layer(block, 128, def[2], 2))
  fe_CNN:add(layer(block, 256, def[3], 2))
  fe_CNN:add(layer(block, 512, def[4], 2))
  fe_CNN:add(Avg(final_pool_size, final_pool_size, 1, 1))

    fe_CNN:get(1).gradInput = nil

    local net = nn.Sequential()
    if resnet_options.classifier_type == "multi"  then
        local classifiers = nn.Concat(2)
        for i = 1,resnet_options.num_classifiers do
            local classifier = nn.Sequential()
            classifier:add(nn.View(final_features_size):setNumInputDims(3)):add(nn.Dropout(0.2)):add(nn.Linear(final_features_size,resnet_options.nOutputs))
            if resnet_options.last_layer then
                classifier:add(resnet_options.last_layer())
            end
            classifiers:add(classifier) 
        end
        if resnet_options.predict_age then     
            local classifier = nn.Sequential() 
            classifier:add(nn.View(final_features_size)):add(nn.Dropout(0.2)):add(nn.Linear(final_features_size,512))classifier:add(cudnn.ReLU(true))    
            classifier:add(nn.Dropout(0.2)):add(nn.Linear(512, 1))  
            classifiers:add(classifier)         
        end        
        toCuda(classifiers) 
        net:add(fe_CNN):add(classifiers)
    elseif resnet_options.classifier_type == "single" then
        local classifier = nn.Sequential()
        classifier:add(nn.View(final_features_size):setNumInputDims(3))
        classifier:add(nn.Linear(final_features_size, resnet_options.nOutputs))
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

   if opt.cudnn == 'deterministic' then
      net:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end


   return makeDataParallel(net, resnet_options.numGPU)  ,final_features_size
end

