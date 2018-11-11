function makeFullyConvolutional(model)
    model:replace(function(m) if torch.typename(m) == 'nn.View' then return nn.Identity() else return m end end)

    local linear_nodes, container_nodes = model:findModules('nn.Linear')
    for i = 1, #linear_nodes do
      -- Search the container for the current Linear node
      for j = 1, #(container_nodes[i].modules) do
        if container_nodes[i].modules[j] == linear_nodes[i] then
          -- Replace with a new instance
          local current_linear = linear_nodes[i]:clone()
          container_nodes[i].modules[j] = cudnn.SpatialConvolution(current_linear.weight:size(2),current_linear.weight:size(1),1,1)
          container_nodes[i].modules[j].weight:copy(current_linear.weight)
          container_nodes[i].modules[j].bias:copy(current_linear.bias)
        end
      end
    end
    
    return toCuda(model)
end

function makeDataParallel(model, nGPU)
    require 'cudnn'
    assert(torch.type(model) ~= 'nn.DataParallelTable', "Error: your model is already a data parallel table")

    if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark
      
      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local rnn = require 'rnn'
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt
    end      
    return toCuda(model)
end

function clearState(model)
    if torch.type(model) == 'nn.DataParallelTable' then
        model.impl:exec(function(m,i) 
                            m:clearState() 
                        end)      
    else
        model:clearState()
    end
end

function saveDataParallel(filename, model)
    if torch.type(model) == 'nn.DataParallelTable' then
        model.impl:exec(function(m,i) 
                            m:clearState() 
                            if i==1 then
                                torch.save(filename, m:clone()) 
                            end
                        end)      
    else
        model:clearState()
        torch.save(filename, model)
    end
end

function getNumberParameters(m)
    local count = 0
    for k,v in pairs(m:parameters()) do 
        local inner_count = 1 
        for i=1,#v:size() do
            inner_count = inner_count * v:size(i)
        end 
    count = count + inner_count
    end 
    return count
end

--function reBuildSavedModel(saved_model)
--    if torch.type(saved_model) == 'nn.DataParallelTable' then
--        return  reBuildSavedModel(saved_model:get(1))
--    elseif torch.type(saved_model) == 'nn.Sequential' then
--        local model = nn.Sequential()
--        for i=1,#saved_model do
--            model:add(reBuildSavedModel(saved_model:get(i)))
--        end
--        return model
--    elseif torch.type(saved_model) == 'nn.Concat' then
--        local model = nn.Concat(saved_model.dimension)
--        for i=1,#saved_model do
--            model:add(reBuildSavedModel(saved_model:get(i)))
--        end
--        return model
--    elseif torch.type(saved_model) == 'nn.DepthConcat' then
--        local model = nn.DepthConcat(saved_model.dimension)
--        for i=1,#saved_model.modules do
--            model:add(reBuildSavedModel(saved_model:get(i)))
--        end
--        return model        
--    else
--        if torch.type(saved_model) == 'cudnn.SpatialConvolution' then
--            local weights_size = saved_model.weight:size()
--            local conv = cudnn.SpatialConvolution(weights_size[2],weights_size[1],weights_size[3],weights_size[4],saved_model.dW,saved_model.dH,saved_model.padW,saved_model.padH,saved_model.groups)
--            conv.weight:copy(saved_model.weight)
--            if saved_model.bias == nil then
--                conv.bias = nil    
--            else
--                conv.bias:copy(saved_model.bias)
--            end
--            return conv
--        elseif torch.type(saved_model) == 'nn.Linear' then 
--            local weights_size = saved_model.weight:size()
--            local linear = nn.Linear(weights_size[2],weights_size[1])
--            linear.weight:copy(saved_model.weight)
--            linear.bias:copy(saved_model.bias)
--            return linear
--        elseif torch.type(saved_model) == 'cudnn.ReLU' then  
--            return(cudnn.ReLU(saved_model.inplace))
--        elseif torch.type(saved_model) == 'cudnn.SpatialMaxPooling' then
--            return(cudnn.SpatialMaxPooling(saved_model.kW,saved_model.kH,saved_model.dW,saved_model.dH,saved_model.padW,saved_model.padH))
--        elseif torch.type(saved_model) == 'cudnn.SpatialAveragePooling' or torch.type(saved_model) == 'nn.SpatialAveragePooling' then
--            return(cudnn.SpatialAveragePooling(saved_model.kW,saved_model.kH,saved_model.dW,saved_model.dH,saved_model.padW,saved_model.padH))  
--        elseif torch.type(saved_model) == 'cudnn.SpatialBatchNormalization' then
--            return(cudnn.SpatialBatchNormalization(saved_model.weight:size(1),saved_model.eps,saved_model.momentum,saved_model.affine))              
--        elseif torch.type(saved_model) == 'nn.View' then
--            return nn.View(saved_model.size)
--        elseif torch.type(saved_model) == 'nn.Normalize' then
--            return nn.Normalize(saved_model.p,saved_model.eps)
--        else
--            print(torch.type(saved_model))
--            error('Unknown element')
--        end        
--    end       
--end

--function loadDataParallel(filename, nGPU)
--    local model = reBuildSavedModel(torch.load(filename))     
--    return makeDataParallel(model:cuda(), nGPU)
--end

function replaceResetMaskedSelect(m)
    local MaskedSelect_nodes, container_nodes = m:findModules('nn.MaskedSelect')

    for i = 1, #MaskedSelect_nodes do
      -- Search the container for the current Linear node
      for j = 1, #(container_nodes[i].modules) do
        if container_nodes[i].modules[j] == MaskedSelect_nodes[i] then
            -- Replace with a new instance
            container_nodes[i].modules[j] = nn.MaskedSelect() 
        end
      end
    end      
    return m
end

function loadDataParallel(filename, nGPU)
    local model = torch.load(filename)
    model = replaceResetMaskedSelect(model)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1), nGPU)
    elseif torch.type(model) == 'nn.Sequential' then
        return makeDataParallel(model, nGPU)
    else
      error('The loaded model is not a Sequential or DataParallelTable module.')
    end
end

function getLastFeatureExtractionLayer(model)
    if torch.type(model) == 'nn.DataParallelTable' then
        local output_size = model:get(1):get(1).output:size()
        local batchSize = output_size[1]
        output_size[1] = batchSize * opt.numGPU

        local out = torch.FloatTensor(output_size)
        model.impl:exec(function(m,i) out[{{(i-1)*batchSize+1,i*batchSize}}]:copy(m:get(1).output)  end) 
        return out
    elseif torch.type(model) == 'nn.Sequential' then
        return model:get(1).output
    else
      error('The loaded model is not a Sequential or DataParallelTable module.')
    end
end
