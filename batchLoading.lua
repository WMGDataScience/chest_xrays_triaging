require 'utils'



function loadRandomlyOverClasses(index)
    local dataTimer = torch.Timer()
    dataTimer:reset()

    collectgarbage()
    if info.covariate_variables_dim > 0 then 
        covariate_variables = torch.FloatTensor(info.opt.batchSize,info.covariate_variables_dim)
    end
    if info.opt.path_CH_embeddings ~= "" then
        covariate_variables = torch.FloatTensor(info.opt.batchSize,2,info.max_CH_len,info.CH_emb_size):fill(0)         
    end

    
    local labels_batch = nil
    local imgs_batch = torch.FloatTensor(info.opt.batchSize,info.opt.input_size[1],info.opt.input_size[3],info.opt.input_size[2])
    if info.opt.multi_task then
        labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes+4) -- 3 for priority classes + 1 abnormal
    elseif info.opt.classifier_type == "multi" then
        if info.opt.predict_age then
            labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes+1)
        else
            labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes)
        end
    elseif info.opt.classifier_type == "single" then 
        labels_batch = torch.FloatTensor(info.opt.batchSize)
    end

    for i=1,info.opt.batchSize do

        local current_img_info = info.imgs_paths_and_infos[info.shuffle[index%#info.imgs_paths_and_infos + 1]]
        local loaded_img = getImage(current_img_info,info)
        if loaded_img ~= nil then   
            imgs_batch[i]:copy(loaded_img)
            if info.covariate_variables_dim > 0 then 
                covariate_variables[i]:copy(getCovariateVariables(current_img_info,info.opt.covariate_variables))                        
            end
            if info.opt.path_CH_embeddings ~= "" then
                local CH_file_path = info.opt.path_CH_embeddings .. current_img_info.acc_numb .. ".hdf5"
                local curr_embeddings = getCH_embeddings(CH_file_path,info.params)
                curr_embeddings = curr_embeddings[{{1,math.min(info.max_CH_len,curr_embeddings:size(1))},{}}] --cut sequences longer than max_CH_len
                
                covariate_variables[i][1][{{1,curr_embeddings:size(1)},{1,curr_embeddings:size(2)}}]:copy(curr_embeddings)                        
            end
    --      image.save("test_" .. i .. ".jpg",imgs_batch[i])

            if info.opt.multi_task then
                local current_priority_class = getPriorityClass(current_img_info.classes,info.priority_list,info.per_priority_mapping)
                labels_batch[{{i},{1,#info.classes}}] = getLabelMulti(current_img_info,info.classes,info)
                labels_batch[{{i},{#info.classes+1,#info.classes+3}}]:copy(getPriorityClassVector(current_priority_class,info.priority_list)+1)
                labels_batch[i][#info.classes+4] = getLabelMulti(current_img_info,{"normal"}, info)
            elseif info.opt.classifier_type == "multi" then
                labels_batch[i] = getLabelMulti(current_img_info,info.classes,info)
            elseif info.opt.classifier_type == "single" then
                labels_batch[i] = getLabelSingle(info.classes,random_class)
            end
    --      print(labels_batch[i])
        else
            error('Image ' .. current_img_info.path .. " loading error")
        end
    end


    if info.opt.verbose >= 2 then print((">imgs loading [no preprocess] = %.2f seconds<"):format(dataTimer:time().real)) end     
    
    preprocess(imgs_batch, info.params['mean'], info.params['var'])
    
    if info.opt.verbose >= 1 then print((">Time to load data from disk = %.2f seconds<"):format(dataTimer:time().real)) end 

    if info.covariate_variables_dim > 0 or info.opt.path_CH_embeddings ~= "" then    
        if info.CH_emb_size > 0 then
            covariate_variables[{{},{2}}]:copy(getMaskFromTensor(covariate_variables[{{},{1}}]))
        end       
        return {imgs_batch,covariate_variables}, labels_batch
    else
        return imgs_batch, labels_batch
    end
end

function loadUniformlyOverClasses(index)
    local dataTimer = torch.Timer()
    dataTimer:reset()
    
    collectgarbage()
    if info.covariate_variables_dim > 0 then 
        covariate_variables = torch.FloatTensor(info.opt.batchSize,info.covariate_variables_dim)
    end
    if info.opt.path_CH_embeddings ~= "" then
        covariate_variables = torch.FloatTensor(info.opt.batchSize,2,info.max_CH_len,info.CH_emb_size):fill(0)         
    end
        
    local labels_batch = nil
    local imgs_batch = torch.FloatTensor(info.opt.batchSize,info.opt.input_size[1],info.opt.input_size[3],info.opt.input_size[2])
    if info.opt.multi_task then
        labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes+4) -- 3 for priority classes + 1 abnormal
    elseif info.opt.classifier_type == "multi" then
        if info.opt.predict_age then
            labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes+1)
        else
            labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes)--*2)
        end
    elseif info.opt.classifier_type == "single" then 
        labels_batch = torch.FloatTensor(info.opt.batchSize)
    end

    local input_classes = {}
    local class_i = 1
    for k,v in pairs(info.imgs_paths_and_infos) do
        input_classes[class_i] = k
        class_i = class_i + 1
    end
    local class_shuffle = torch.randperm(#input_classes)  --torch.range(1,#input_classes) 
    
--    local sumt1,sumt2 = 0,0
    local i = 1
    local k = 1
    while i<=info.opt.batchSize do
        for j=1,#input_classes do
            if i<=info.opt.batchSize then
                local random_class = input_classes[class_shuffle[j]]
                local current_img_info = info.imgs_paths_and_infos[random_class][info.shuffle[random_class][(index+k-1)%#info.imgs_paths_and_infos[random_class] + 1]]

--                local loaded_img,t1,t2 = getImage(current_img_info,info)
--                sumt1, sumt2 = sumt1 + t1, sumt2 + t2
                local loaded_img = getImage(current_img_info,info)
                if loaded_img ~= nil then

                    imgs_batch[i]:copy(loaded_img)
                    if info.covariate_variables_dim > 0 then 
                        covariate_variables[i]:copy(getCovariateVariables(current_img_info,info.opt.covariate_variables))                        
                    end
                    if info.opt.path_CH_embeddings ~= "" then
                        local CH_file_path = info.opt.path_CH_embeddings .. current_img_info.acc_numb .. ".hdf5"
                        local curr_embeddings = getCH_embeddings(CH_file_path,info.params)
                        curr_embeddings = curr_embeddings[{{1,math.min(info.max_CH_len,curr_embeddings:size(1))},{}}] --cut sequences longer than max_CH_len

                        covariate_variables[i][1][{{1,curr_embeddings:size(1)},{1,curr_embeddings:size(2)}}]:copy(curr_embeddings)                        
                    end
--                    image.save(current_img_info.acc_numb .. "_" .. i .. ".jpg",imgs_batch[i])
                    if info.opt.multi_task then
                        local current_priority_class = getPriorityClass(current_img_info.classes,info.priority_list,info.per_priority_mapping)
                        labels_batch[{{i},{1,#info.classes}}] = getLabelMulti(current_img_info,info.classes,info)
                        labels_batch[{{i},{#info.classes+1,#info.classes+3}}]:copy(getPriorityClassVector(current_priority_class,info.priority_list)+1)
                        labels_batch[i][#info.classes+4] = getLabelMulti(current_img_info,{"normal"}, info)
                        
                    elseif info.opt.classify_per_priority == 1 then
                        local current_priority_class = getPriorityClass(current_img_info.classes,info.priority_list,info.per_priority_mapping)
                        if info.opt.ordinal_regression == 1 then
                            labels_batch[i]:copy(getPriorityClassVector(current_priority_class,info.priority_list)+1)
                        else
                            labels_batch[i] = info.priority_list[current_priority_class]
                        end
                    else
                        if info.opt.classifier_type == "multi" then
                            labels_batch[i] = getLabelMulti(current_img_info,info.classes,info)
                        elseif info.opt.classifier_type == "single" then
                            labels_batch[i] = getLabelSingle(info.classes,random_class)
                        end
                    end
--                    print(labels_batch[i])
                    i=i+1
                elseif info.opt.verbose >= 2 then
                    print(current_img_info['path'] .. " equal to nil" )
                end                
            end
        end
        k = k + 1
    end
--    if info.opt.verbose >= 2 then print((">img_loading_time = %.2f seconds, img_augment_time = %.2f seconds<"):format(sumt1,sumt2))  end 
    
    if info.opt.verbose >= 2 then print((">Time to load data from disk [no preprocess] = %.2f seconds<"):format(dataTimer:time().real)) dataTimer:reset() end     
    

    preprocess(imgs_batch, info.params['mean'], info.params['var'])
   

    if info.opt.verbose == 1 then print((">Time to load data from disk = %.2f seconds<"):format(dataTimer:time().real)) 
    elseif info.opt.verbose >= 2 then print((">Preprocess time = %.2f seconds<"):format(dataTimer:time().real)) end 

    if info.covariate_variables_dim > 0 or info.opt.path_CH_embeddings ~= "" then     
        if info.CH_emb_size > 0 then
            covariate_variables[{{},{2}}]:copy(getMaskFromTensor(covariate_variables[{{},{1}}]))
        end 
        return {imgs_batch,covariate_variables}, labels_batch
    else
        return imgs_batch, labels_batch
    end
end
         
function loadBatch_test(index)
    local dataTimer = torch.Timer()
    dataTimer:reset()
    
    collectgarbage() 
    local batch_dim = math.min(#info.imgs_paths_and_infos - (index-1),info.opt.batchSize)
    if info.covariate_variables_dim > 0 then 
        covariate_variables = torch.FloatTensor(batch_dim,info.covariate_variables_dim)
    end   
    if info.opt.path_CH_embeddings ~= "" then
        covariate_variables = torch.FloatTensor(batch_dim,2,info.max_CH_len,info.CH_emb_size):fill(0)         
    end
            
    local imgs_batch = nil
    if info.opt.fixed_size_input == 1 then
        imgs_batch = torch.FloatTensor(batch_dim,info.opt.input_size[1],info.opt.input_size[3],info.opt.input_size[2])
    else
        imgs_batch = {}
    end
    local labels_batch = nil
    if info.opt.multi_task then
        labels_batch = torch.FloatTensor(info.opt.batchSize,#info.classes+4) -- 3 for priority classes + 1 abnormal
    elseif info.opt.classifier_type == "multi" then
        if info.opt.predict_age then
            labels_batch = torch.FloatTensor(batch_dim,#info.classes+1)
        else
            labels_batch = torch.FloatTensor(batch_dim,#info.classes)--2*#info.classes)
        end
    elseif info.opt.classifier_type == "single" then 
        labels_batch = torch.FloatTensor(batch_dim,1)
    end
    
    local ids_batch = tds.Vec()--torch.FloatTensor(batch_dim)
    
    for i=1,batch_dim do
        local current_img_info = info.imgs_paths_and_infos[index+i-1]
        local loaded_img = getImage(current_img_info,info)
        if loaded_img ~= nil then
            if info.opt.fixed_size_input == 1 then
                imgs_batch[i]:copy(loaded_img)
            else
                imgs_batch[i] = loaded_img
            end
            if info.covariate_variables_dim > 0 then 
                covariate_variables[i]:copy(getCovariateVariables(current_img_info,info.opt.covariate_variables))                     
            end        
            if info.opt.path_CH_embeddings ~= "" then
                local CH_file_path = info.opt.path_CH_embeddings .. current_img_info.acc_numb .. ".hdf5"
                local curr_embeddings = getCH_embeddings(CH_file_path,info.params)
                curr_embeddings = curr_embeddings[{{1,math.min(info.max_CH_len,curr_embeddings:size(1))},{}}] --cut sequences longer than max_CH_len
                
                covariate_variables[i][1][{{1,curr_embeddings:size(1)},{1,curr_embeddings:size(2)}}]:copy(curr_embeddings)                        
            end
                        
            if info.opt.multi_task then
                local current_priority_class = getPriorityClass(current_img_info.classes,info.priority_list,info.per_priority_mapping)
                labels_batch[{{i},{1,#info.classes}}] = getLabelMulti(current_img_info,info.classes,info)
                labels_batch[{{i},{#info.classes+1,#info.classes+3}}]:copy(getPriorityClassVector(current_priority_class,info.priority_list)+1)
                labels_batch[i][#info.classes+4] = getLabelMulti(current_img_info,{"normal"}, info)
            elseif info.opt.classify_per_priority == 1 then
                local current_priority_class = getPriorityClass(current_img_info.classes,info.priority_list,info.per_priority_mapping)
                if info.opt.ordinal_regression == 1 then
                    labels_batch[i]:copy(getPriorityClassVector(current_priority_class,info.priority_list)+1)
                else
                    labels_batch[i] = info.priority_list[current_priority_class]
                end        
            else
                if info.opt.classifier_type == "multi" then
                    labels_batch[i] = getLabelMulti(current_img_info,info.classes,info)
                elseif info.opt.classifier_type == "single" then
                    local this_instace_class = fromClassLabelToString(getLabelMulti(current_img_info,info.opt.classes,info),info.opt.classes)
                    labels_batch[i] = getLabelSingle(info.classes,this_instace_class)
                end 
            end       
            ids_batch[i] = current_img_info.acc_numb--.id
        else
            error('Image ' .. current_img_info.path .. " loading error")
        end          
    end
   
    if info.opt.verbose >= 2 then print((">imgs loading [no preprocess] = %.2f seconds<"):format(dataTimer:time().real)) end     
    
    preprocess(imgs_batch, info.params['mean'], info.params['var'])

    if info.opt.verbose >= 1 then print((">Time to load data from disk = %.2f seconds<"):format(dataTimer:time().real)) end 
        
    
    if info.covariate_variables_dim > 0 or info.opt.path_CH_embeddings ~= "" then      
        if info.CH_emb_size > 0 then
            covariate_variables[{{},{2}}]:copy(getMaskFromTensor(covariate_variables[{{},{1}}]))
        end     
        return {imgs_batch,covariate_variables}, labels_batch, ids_batch
    else
        return imgs_batch, labels_batch, ids_batch
    end    
end
