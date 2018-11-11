require 'image'
require 'torch'
tds = require 'tds'
require 'hdf5'

local function log2(input_num)
    return  math.log(input_num) / math.log(2)
end

function getMaskFromTensor(t)
    t = t:reshape(t:size(1),t:size(3),t:size(4))
    local mask = t:ne(0)
    for batch_idx=1,mask:size(1) do
        local flag_zero_fill = false
        for seq_idx=1,mask:size(2) do
            if mask[batch_idx][seq_idx][1] == 0 and mask[batch_idx][seq_idx][2] == 0
                and mask[batch_idx][seq_idx][3] == 0 then
                if seq_idx <= 2 then
                    mask[batch_idx]:fill(0)
                    mask[batch_idx][{1,{}}]:fill(1)
                else
                    mask[batch_idx]:fill(0)
                    mask[batch_idx][{seq_idx-1,{}}]:fill(1)                
                end
                flag_zero_fill = true
                break
            end
        end
        if not flag_zero_fill then
            mask[batch_idx]:fill(0)
            mask[batch_idx][{mask:size(2),{}}]:fill(1)            
        end
    end
    mask = mask:view(mask:size(1),1,mask:size(2),mask:size(3))
    return mask
end 

local function pow2(input_num)
    local result = 1
    
    for i=1,input_num do
        result = result * 2
    end

    return result
end

function tableWithAllClassesInTheRightOrder(classes_sets)
    local out_tbl = {}
    table.insert(out_tbl,'normal')
    
    for i=1,#classes_sets do
        for j=1,#classes_sets[i] do
            table.insert(out_tbl,classes_sets[i][j])
        end
    end
    
    return out_tbl
end

function isTimeToStopTraining(train_tbl,val_tbl)
    if #val_tbl < 4 then 
        return false
    else
        local best_val = -1
        for i=1,#val_tbl-3 do
            if best_val < val_tbl[i] then
                best_val = val_tbl[i]
            end
        end

        if val_tbl[#val_tbl] < best_val and val_tbl[#val_tbl-1] < best_val and val_tbl[#val_tbl-2] < best_val then
            return true
        else
            return false
        end 
    end
end

function getClassesSets(train_infos,classes)
    
    local per_class_counter = tds.Hash()
    for i=1,#classes do
        per_class_counter[classes[i]] = 0
    end
        
    for i=1,#train_infos do
        for j=1,#train_infos[i].classes do
            per_class_counter[train_infos[i].classes[j]] = per_class_counter[train_infos[i].classes[j]] + 1
        end
    end

    local class_dim_range_all = {}
    class_dim_range_all[0] = 0
    for i=1,9 do    
        class_dim_range_all[i] = class_dim_range_all[i-1] + (pow2(i-1) * 100)
    end
    local class_dim_range = {}
    for i=1,#class_dim_range_all do
        if class_dim_range_all[i] > 1000 then
            table.insert(class_dim_range,class_dim_range_all[i])
        end
    end
    class_dim_range[0] = 0
    
    local classes_sets = tds.Vec()
    for cl,cl_count in pairs(per_class_counter) do
        local classes_sets_index = 0
        for i=#class_dim_range,0,-1 do
            if class_dim_range[i] < cl_count then
                classes_sets_index = i + 1
                break
            end
        end

        if classes_sets[classes_sets_index] == nil then
            classes_sets[classes_sets_index] = tds.Vec()
        end
        if cl ~= 'normal' then
            classes_sets[classes_sets_index]:insert(cl)
        end
    end

    
    -- check if there is ANY empty set and discard it
    local result_classes_sets = tds.Vec()
    for i=#classes_sets,1,-1 do
        if classes_sets[i] ~= nil and #classes_sets[i] > 0 then
            result_classes_sets:insert(classes_sets[i])
        end
    end

    return result_classes_sets
end

function definePriorityMapping(classes)
    local per_priority_mapping,priority_list = tds.Hash(),tds.Hash()

    priority_list["normal"],priority_list["non_urgent_finding"],priority_list["urgent"],priority_list["critical"] = 1,2,3,4
    for i=1,#classes do
        if classes[i] == "pneumomediastinum" or classes[i] == "pneumothorax" or classes[i] == "subcutaneous_emphysema" or classes[i] == "mediastinum_widened" 
           or classes[i] == "pneumoperitoneum" then
            per_priority_mapping[classes[i]] = "critical"
        elseif classes[i] == "consolidation" or classes[i] == "pleural_abnormality" or classes[i] == "parenchymal_lesion" or classes[i] == "pleural_effusion"
            or classes[i] == "right_upper_lobe_collapse" or classes[i] == "cavitating_lung_lesion" or classes[i] == "rib_fracture" or classes[i] == "interstitial_shadowing"
            or classes[i] == "left_lower_lobe_collapse" or classes[i] == "clavicle_fracture" or classes[i] == "paratracheal_hilar_enlargement" 
            or classes[i] == "dilated_bowel" or classes[i] == "ground_glass_opacification" or classes[i] == "left_upper_lobe_collapse" or classes[i] == "rib_lesion"
            or classes[i] == "right_lower_lobe_collapse" or classes[i] == "mediastinum_displaced" or classes[i] == "right_middle_lobe_collapse" 
            or classes[i] == "widened_paratracheal_stripe" or classes[i] == "enlarged_hilum" or classes[i] == "pleural_thickening" or classes[i] == "pleural_lesion" then
            per_priority_mapping[classes[i]] = "urgent"
        elseif classes[i] == "object" or classes[i] == "cardiomegaly" or classes[i] == "emphysema" or classes[i] == "atelectasis" or classes[i] == "bronchial_wall_thickening"
               or classes[i] == "scoliosis" or classes[i] == "hernia" or classes[i] == "hyperexpanded_lungs" or classes[i] == "unfolded_aorta" 
               or classes[i] == "dextrocardia" or classes[i] == "bulla" or classes[i] == "aortic_calcification" or classes[i] == "hemidiaphragm_elevated" then
            per_priority_mapping[classes[i]] = "non_urgent_finding"  
        elseif classes[i] == "normal" then
            per_priority_mapping[classes[i]] = "normal"               
        end
    end
    
    return per_priority_mapping,priority_list 
end

function toCuda(tensor)
    local tensor_precision = nil
    if opt == nil then
        tensor_precision = info.opt.tensor_precision
    else
        tensor_precision = opt.tensor_precision
    end
    if tensor_precision == 'half' then
        return tensor:cudaHalf()
    elseif tensor_precision == 'single' then
        return tensor:cuda()
    end
end
function createCudaTensor(sizes)
    local tensor_precision = nil
    if opt == nil then
        tensor_precision = info.opt.tensor_precision
    else
        tensor_precision = opt.tensor_precision
    end
    if tensor_precision == 'half' then
        if sizes == nil then
            return torch.CudaHalfTensor()
        else
            return torch.CudaHalfTensor(sizes)
        end
    elseif tensor_precision == 'single' then
        if sizes == nil then
            return torch.CudaTensor()    
        else
            return torch.CudaTensor(sizes)
        end
    end   
end    





function split_string(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; local i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
            t[i] = str
            i = i + 1
        end
        return t
end

function fromStringToDimensionsTable(inp_string,separator)
    local splitted_string = split_string(inp_string, 'x')
    return {tonumber(splitted_string[1]),tonumber(splitted_string[2]),tonumber(splitted_string[3])}
end

function table.copy(tbl)
    local tlb_copy = {}
    for i=1,#tbl do
        if torch.type(tbl[i]) == 'table' then
            table.insert(tlb_copy,table.copy(tbl[i]))
        else
            table.insert(tlb_copy,tbl[i])
        end
    end
    return tlb_copy
end



function table.contains(tbl,element)--,print_flag)
    for i=1,#tbl do
        --if print_flag then print(tbl[i] ..  " - " .. element) end
        if tbl[i] == element then
            --if print_flag then print("true") end
            return true
        end
    end
    --if print_flag then print("false") end
    return false
end

function table.find_and_remove(tbl,element)
    local out_tbl = {}
    for i=1,#tbl do
        if tbl[i] ~= element then
            table.insert(out_tbl,tbl[i])
        end
    end
    return out_tbl
end

function updateConfusions(net_out,label,confusions,classes)
    local predicted,label_class = {},{}
    local formatted_preds = tds.Vec()
    
    for i=1,#classes do
        predicted[classes[i]] = 2
        label_class[classes[i]] = 2
    end
    predicted["normal"] = 2  
    if label[1] == #classes + 1 then
        label_class["normal"] = 1
    else
        label_class["normal"] = 2
    end

    for i = 1, net_out:size(1) do
        local max,idx = torch.max(net_out[i],1)
         
        if idx[1] == #classes + 1 then
            break
        end 
        formatted_preds:insert(classes[idx[1]])          
 
    end
    if #formatted_preds == 0 then formatted_preds:insert("normal")  end

    local ss_counter = 0
    local flag_predicted = 0

    for i = 1, net_out:size(1) do
    
        local max,idx = torch.max(net_out[i],1)
        if idx[1] == #classes + 1 then --stopsignal
            ss_counter = ss_counter + 1 
        end
        if ss_counter == 0 then
            predicted[classes[idx[1]]] = 1
            flag_predicted = 1 --so it is not normal anymore
        end
        if label[i] ~= #classes + 1 then
            label_class[classes[label[i]]] = 1
        end

    end
    if flag_predicted == 0 then
        predicted["normal"] = 1    
    end
    
    for i=1,#classes do
        confusions[classes[i]]:add(predicted[classes[i]],label_class[classes[i]])
    end
    confusions["normal"]:add(predicted["normal"],label_class["normal"])
    
    return formatted_preds
end

function orderClassesByNumberOfOccurrency(data_table,classes,normal_flag)
    if normal_flag == nil then normal_flag = true end
    local class_count = tds.Hash()
    for j=1,#classes do
        class_count[classes[j]] = 0
    end  
    
    for i=1,#data_table do
        for j=1,#classes do
            if table.contains(data_table[i].classes,classes[j]) then
                class_count[classes[j]] = class_count[classes[j]] + 1
            end
        end        
    end
    
    local ordered_classes = {}
    local result_occurrency = tds.Hash()

    if opt.verbose >= 2 then
        for k,v in pairs(class_count) do
            print(k .. " = " .. v)   
        end  
    end
    while #class_count > 0 do
        local min = math.huge
        local min_class = nil
        for k,v in pairs(class_count) do
            
            if min > v then
                min = v
                min_class = k
            end   

        end
        class_count[min_class] = nil
        if (not normal_flag) or min_class ~= "normal" then
            table.insert(ordered_classes,min_class)
            result_occurrency[min_class] = min
        end
        if opt.verbose >= 1 then
            print(min_class .. " = " .. min)
        end
    end

    return ordered_classes,result_occurrency
end

function minValue(table)
    if not table then return 0 end
    local min = nil
    for k,v in pairs(table) do
        if type(v) == "table" then
            if min == nil or #v < min then
                min = #v
            end
        else
            if min == nil or v < min then
                min = v
            end        
        end
    end
    return min
end

function fromClassesToPriorityClass(classes,priority_list,per_priority_mapping)
    local current_class = "normal" 
    for i=1,#classes do
        if priority_list[per_priority_mapping[classes[i]]] > priority_list[current_class] then
            current_class = per_priority_mapping[classes[i]]
        end
    end
    return current_class
end

function fromLabelToPriorityClass(labels,classes,priority_list,per_priority_mapping)
    local current_class = "normal" 
    for i=1,labels:size(1) do
        if labels[i] == 1 and priority_list[per_priority_mapping[classes[i]]] > priority_list[current_class] then
            current_class = per_priority_mapping[classes[i]]
        end
    end
    return priority_list[current_class]
end



function directory_exists( sPath )
  if type( sPath ) ~= "string" then return false end
  local response = os.execute( "cd " .. sPath )

  return response
end

function file_exists(filename)
    -- ----------
    -- Check if the file exists
    -- 
    -- Arguments
    -- =========
    -- filename: string, file to check
    -- 
    -- Return
    -- ======
    -- Boolean, true: the file exists.
    -- ----------

    local f=io.open(filename,"r")

    if f~=nil then io.close(f) return true else return false end

end


function scandir(directory)
    -- ----------
    -- Return the list of all files into directory
    -- 
    -- Arguments
    -- =========
    -- directory: string, path of the directory
    -- 
    -- Return
    -- ======
    -- list of strings, each element is a filename
    -- ----------

    local i, t, popen = 0, {}, io.popen

    for filename in popen('ls "'..directory..'"'):lines() do
        i = i + 1
        t[i] = filename
    end

    return t
end

function getDirPath(filepath)
    local var = split_string(filepath,'/')
    local result = ""
    for i=1,#var-1 do
       result = result .. var[i] .. "/" 
    end
    return result    
end

function scandir_dirs(directory)
    -- ----------
    -- Return the list of all sub-directories in a directory
    -- 
    -- Arguments
    -- =========
    -- directory: string, path of the directory
    -- 
    -- Return
    -- ======
    -- list of strings, each element is a sub-directory
    -- ----------

    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -d '..directory..'*'):lines() do
        i = i + 1
        local var = split_string(filename, '/')
        t[i] = var[#var]
    end

    return t
end


function transfer_data(x)
    -- ----------
    -- Cast x to cuda
    -- 
    -- Arguments
    -- =========
    -- x: tensor
    -- 
    -- Return
    -- ======
    -- x casted to cuda
    -- ----------

    x = x:float()
    
    return x:cuda()
end


function load_data_hdf5(path_data_norm, path_data_comp,info)
    -- ----------
    -- Load data from disk to memory.
    -- It assumes that data files are in hdf5 format.
    -- Each hdf5 should have the following fields: data, labels, ids
    -- 
    -- Arguments
    -- =========
    -- path_data_norm: string, path of the class norm hdf5 file
    -- path_data_comp: string, path of the class comp hdf5 file 
    -- info (optional): contains additional infos that normally the function can access as global variables
    -- Return
    -- ======
    -- data, labels and ids for both classes
    -- ----------

    ----------------------------------------------------------------------
    --  Check if data directory exists
    ----------------------------------------------------------------------
    local path_temp = nil
    local cl_norm = nil
    local cl_comp = nil 
    local verbose = 0 
    
    if info then
        path_temp = info.opt.path_temp
        cl_norm = info.opt.classNorm
        cl_comp = info.opt.classComp
        verbose = info.opt.verbose
    else
        path_temp = opt.path_temp
        cl_norm = opt.classNorm
        cl_comp = opt.classComp 
        verbose = opt.verbose      
    end

    local split_path = split_string(path_data_norm, '/')
    -- get if we are in the training o test set and the filename of the chunk
    local set_type = split_path[#split_path-1]
    local filename_norm = split_path[#split_path]
    local filename_comp = split_string(path_data_comp, '/')[#split_path]

    -- build local paths
    local local_path_norm = path_temp  .. cl_norm .. '/' .. set_type .. '/' .. filename_norm
    local local_path_comp = path_temp  .. cl_comp .. '/' .. set_type .. '/' .. filename_comp

    -- download if necessary
    if file_exists(local_path_norm) == false then

        if verbose >= 0 then
            print("+++ Downloading from " .. path_data_norm)
        end  
        os.execute('cp ' .. path_data_norm .. " " .. local_path_norm)
    end

    if file_exists(local_path_comp) == false then

        if verbose >= 0 then 
            print("+++ Downloading from " .. path_data_comp)
        end 

        os.execute('cp ' .. path_data_comp .. " " .. local_path_comp)
    end


    if verbose >= 2 then 
        print("Loading data norm from:" .. local_path_norm)
    end
    

    local f_norm = hdf5.open(local_path_norm, 'r')
    local dat_norm = f_norm:read("data"):all()
    local lab_norm = f_norm:read("labels"):all()
    local ids_norm = f_norm:read("ids"):all()

   
    if verbose >= 2 then 
        print("Loading data comp from:" .. local_path_comp)
    end
    

    local f_comp = hdf5.open(local_path_comp, 'r')
    local dat_comp = f_comp:read("data"):all()
    local lab_comp = f_comp:read("labels"):all()
    local ids_comp = f_comp:read("ids"):all()

    lab_comp:fill(1)
    if verbose >= 1 then 
        print("Chunk <" .. local_path_norm .. "-" .. local_path_comp .. "> loaded")
    end

    return dat_norm, lab_norm, ids_norm, dat_comp, lab_comp, ids_comp
end



function sample_concatenate_data(norm_dat, norm_lab, norm_ids, comp_dat, comp_lab, comp_ids)
    -- ----------
    -- Concatenates data, labels and ids of both classes
    -- ----------
     
--    local sample_norm_dat, sample_norm_lab, sample_norm_ids = sample(norm_dat, norm_lab, norm_ids)
--    local sample_comp_dat, sample_comp_lab, sample_comp_ids = sample(comp_dat, comp_lab, comp_ids)

    local dat_concat = torch.cat(norm_dat:transpose(2,4):transpose(3,4), comp_dat:transpose(2,4):transpose(3,4), 1)  
    local lab_concat = torch.cat(norm_lab, comp_lab, 1)  
    local ids_concat = torch.cat(norm_ids, comp_ids, 1)  
    

    if lab_concat:min() == 0 then
        lab_concat = lab_concat + 1
    end

    return dat_concat, lab_concat, ids_concat
end




function save_array_txt(filename, array)
    -- ----------
    -- Save array to txt file
    -- 
    -- Arguments
    -- =========
    -- filename: string, path where to save the array
    -- ----------
    
    -- Opens a file in writing mode
    local file = io.open(filename, "w")

    -- Put each element on a different row
    for i = 1, #array do
      file:write(array[i])
      file:write("\n")
    end

    file:close()
end

function save_arrays_txt(save_dir_path, result_struct)
    
    for k,v in pairs(result_struct.false_neg) do
        save_array_txt(save_dir_path .. k .. "_false_neg.txt",result_struct.false_neg[k])
        save_array_txt(save_dir_path .. k .. "_false_pos.txt",result_struct.false_pos[k])
        save_array_txt(save_dir_path .. k .. "_true_neg.txt",result_struct.true_neg[k])
        save_array_txt(save_dir_path .. k .. "_true_pos.txt",result_struct.true_pos[k])
    end
    
end

function save_results_json(save_dir_path,result_struct)
    local json = require "cjson"
    local json_string = json.encode(result_struct)
    

    local file = io.open(save_dir_path .. "results.json", "w")
    file:write(json_string)  
    file:close()    
end

function getMeasures(conf)

    local TP = conf.mat[1][1]
    local TN = 0
    local FP = 0
    local FN = 0
        
    for i = 2,conf.mat:size(1) do
        FP = FP + conf.mat[i][1]
        for j = 2,conf.mat:size(2) do
            TN = TN + conf.mat[i][j]
        end
    end
    for j = 2,conf.mat:size(2) do
        FN = FN + conf.mat[1][j]
    end
    
    local sensitivity = 0
    local specificity = 0
    if (TP + FN) > 0 then
        sensitivity = TP / (TP + FN)
    end
    if (TN + FP) > 0 then
        specificity = TN / (TN + FP)
    end
    local acc = (TP+TN)/(TP+TN+FP+FN)
    local f1score = (2*TP) / (2*TP + FP + FN)
    local matthew = (TP*TN - FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    local aacc = 1/2*(TP/(TP+FN)+TN/(TN+FP))
    local f_measure = (sensitivity*specificity)/(sensitivity+specificity)*2


    if ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ==  0 then
      matthew = 0
    end

    return acc, f1score, matthew, aacc, f_measure
end

function getMeasures_SingleClassifier(conf)
    local final_acc,final_f1score,final_matthew,final_aacc,final_f_measure = 0,0,0,0,0
    for k=1,conf.mat:size(1) do
        local TP = conf.mat[k][k]
        local TN = 0
        local FP = 0
        local FN = 0
            
        for i = 1,conf.mat:size(1) do
            if k~=i then
                FP = FP + conf.mat[i][k]
                for j = 1,conf.mat:size(2) do
                    if j~=k then
                        TN = TN + conf.mat[i][j]
                    end
                end
            end
        end
        for j = 1,conf.mat:size(2) do
            if j~=k then
                FN = FN + conf.mat[k][j]
            end
        end
        
        local sensitivity = 0
        local specificity = 0
        if (TP + FN) > 0 then
            sensitivity = TP / (TP + FN)
        end
        if (TN + FP) > 0 then
            specificity = TN / (TN + FP)
        end
        local acc = (TP+TN)/(TP+TN+FP+FN)
        local f1score = (2*TP) / (2*TP + FP + FN)
        local matthew = (TP*TN - FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        local aacc = 1/2*(TP/(TP+FN)+TN/(TN+FP))
        local f_measure = (sensitivity*specificity)/(sensitivity+specificity)*2
    
    
        if ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ==  0 then
          matthew = 0
        end
        final_acc = final_acc + acc
        final_f1score = final_f1score + f1score
        final_matthew = final_matthew + matthew
        final_aacc = final_aacc + aacc
        final_f_measure = final_f_measure + f_measure
    end
    return final_acc/conf.mat:size(1),final_f1score/conf.mat:size(1),final_matthew/conf.mat:size(1),final_aacc/conf.mat:size(1),final_f_measure/conf.mat:size(1)
end

function getAvgMeasures(confusions,classes)
    local avg_measures = {}
    avg_measures["accuracy"] = 0
    avg_measures["f1score"] = 0
    avg_measures["matthew_correlation"] = 0 
    avg_measures["avg_accuracy"] = 0
    avg_measures["f_measure"] = 0
    for i=1,#classes do
        print(confusions[classes[i]]) 
        local acc, f1score, matthew, aacc,f_measure = getMeasures(confusions[classes[i]])
--        print(classes[i],("Accuracy = %.2f,"):format(acc),("F1 Score = %.2f,"):format(f1score),("Matthew corr. = %.2f,"):format(matthew),("Avg. accuracy = %.2f,"):format(aacc),("F measure = %.2f,"):format(f_measure))
        avg_measures["accuracy"] = avg_measures["accuracy"] + acc
        avg_measures["f1score"] = avg_measures["f1score"] + f1score
        avg_measures["matthew_correlation"] = avg_measures["matthew_correlation"] + matthew
        avg_measures["avg_accuracy"] = avg_measures["avg_accuracy"] + aacc
        avg_measures["f_measure"] = avg_measures["f_measure"] + f_measure
    end
    for k,v in pairs(avg_measures) do
        avg_measures[k] = v / #classes
        print(k .. " = " .. avg_measures[k])
    end
    return avg_measures
end
function getFMeasure(confusion)
    local TP = confusion.mat[2][2]
    local TN = confusion.mat[1][1]
    local FP = confusion.mat[1][2]
    local FN = confusion.mat[2][1]
    
    local sensitivity = 0
    local specificity = 0
    if (TP + FN) > 0 then
        sensitivity = TP / (TP + FN)
    end
    if (TN + FP) > 0 then
        specificity = TN / (TN + FP)
    end
    return (sensitivity*specificity)/(sensitivity+specificity)*2
end

function logInfosMultiThreshold(conf_tbl,thresholds)
    local result = {}

    for k,v in pairs(conf_tbl) do
        local conf_infos = {}
        local current_cm = v[0.5]
        if thresholds[k] ~= nil then 
            current_cm = v[thresholds[k]]
        end
        
        conf_infos["Accuracy"], conf_infos["F1 Score"],conf_infos["Matthew correlation"],conf_infos["Avg. accuracy"], conf_infos["F measure"] = getMeasures(current_cm)
        for i = 1, current_cm.mat:size(1) do
            for j=1,current_cm.mat:size(2) do
                conf_infos["mat_" .. i .. "_".. j] = current_cm.mat[i][j]
            end
        end        
        
        result[k] = conf_infos
    end

    
    
    return result

end

function getPath(str,sep)
    sep=sep or'/'
    return str:match("(.*"..sep..")")
end

function logInfos(conf)
    local result = {}
    if conf["add"] == nil then
        for k,v in pairs(conf) do
            local conf_infos = {}
            conf_infos["Accuracy"], conf_infos["F1 Score"],conf_infos["Matthew correlation"],conf_infos["Avg. accuracy"], conf_infos["F measure"] = getMeasures(conf[k])
            for i = 1, conf[k].mat:size(1) do
                for j=1,conf[k].mat:size(2) do
                    conf_infos["mat_" .. i .. "_".. j] = conf[k].mat[i][j]
                end
            end

            result[k] = conf_infos
        end
    else
        local conf_infos = {}
        conf_infos["Accuracy"], conf_infos["F1 Score"],conf_infos["Matthew correlation"],conf_infos["Avg. accuracy"], conf_infos["F measure"] = getMeasures(conf)
        for i = 1, conf.mat:size(1) do
            for j=1,conf.mat:size(2) do
                conf_infos["mat_" .. i .. "_".. j] = conf.mat[i][j]
            end
        end

        result = conf_infos
    end
    
    
    return result

end

function declareAndInitializeCoOccuranceCM()
    local FN_CoOccurence,FP_CoOccurence,Error_CoOccurence,FN_FP_CoOccurence = tds.Hash(),tds.Hash(),tds.Hash(),tds.Hash()
    for i=1,#classes do
        FN_CoOccurence[classes[i]],FP_CoOccurence[classes[i]],Error_CoOccurence[classes[i]],FN_FP_CoOccurence[classes[i]] = tds.Hash(),tds.Hash(),tds.Hash(),tds.Hash()
        for ii=1,#classes do
            if i ~= ii then
                FN_CoOccurence[classes[i]][classes[ii]],FP_CoOccurence[classes[i]][classes[ii]],Error_CoOccurence[classes[i]][classes[ii]],FN_FP_CoOccurence[classes[i]][classes[ii]] = 0,0,0,0
            end
        end
    end
    return FN_CoOccurence,FP_CoOccurence,Error_CoOccurence,FN_FP_CoOccurence
end

function updateErrorLists(FPs_list,FNs_list,errors_list,pos_pres,neg_pred,label,class)
    if pos_pres > neg_pred then
        if label == 2 then
            --FP
            FPs_list:insert(class) 
            errors_list:insert(class) 
        end    
    else
        if label == 1 then
            --FN
            FNs_list:insert(class)
            errors_list:insert(class)  
        end                  
    end
end

function updateCoOccuranceCM(CoOccurenceCM,list1,list2)
    for k =1,#list1 do
        for kk = 1,#list2 do
            if list1[k]~=list2[kk] then
                CoOccurenceCM[list1[k]][list2[kk]] = CoOccurenceCM[list1[k]][list2[kk]] + 1
            end
        end            
    end 
end

function getClassCount_InMiniBatch(minibatch_labels)
    local result = 0        
    
    for j=1,minibatch_labels:size(1) do

        if minibatch_labels[j] == 1 then
            result = result + 1
        end            

    end
    
    return result
end

function getCovariateVariablesDim(opt_covariate_variables)  
    local covariate_variables_dim = 0
    if opt_covariate_variables[1] == 1 then
        covariate_variables_dim = 1-- age  
    end
    if opt.covariate_variables[2] == 1 then
        covariate_variables_dim = covariate_variables_dim + 2 -- AP/PA 
    end
    if opt.covariate_variables[3] == 1 then
        covariate_variables_dim = covariate_variables_dim + 2 -- sex 
    end
    return covariate_variables_dim
end

function save_perfomances(filename, conf_matrix,class_name)
    -- ----------
    -- Save confusion matrix, accuracy, specificity and sensitivity on file
    -- 
    -- Arguments
    -- =========
    -- filename: string, path where to save the array
    -- confusion: confusion matrix
    -- ----------

    local perfomances = {}

    local TP = conf_matrix.mat[2][2]
    local TN = conf_matrix.mat[1][1]
    local FP = conf_matrix.mat[1][2]
    local FN = conf_matrix.mat[2][1]


    perfomances['totAccuracy'] = conf_matrix.totalValid
    perfomances['sensitivity'] = TP / (TP + FN)
    perfomances['specificity'] = TN / (TN + FP)

    local file = io.open(filename, "w")

    -- Write accuracy, sensitivity and specificy on file
    for key, value in pairs(perfomances) do
      file:write(key)
      file:write("\t")
      file:write(value)
      file:write("\n")
    end

    -- Write Confusion matrix on file
    file:write(class_name)
    file:write("\t")
    file:write(TN)
    file:write("\t")
    file:write(FP)
    file:write("\n")
    file:write("no_" .. class_name)
    file:write("\t")
    file:write(FN)
    file:write("\t")
    file:write(TP)  

    file:close()

end


function fromTensorToClassList(label_tensor,classes)
    local result = {}
    for i = 1, label_tensor:size(1) do
        if label_tensor[i] == 1 then
            table.insert(result,classes[i])
        end
    end
    return result
end

--[[
   Save Table to File
   Load Table from File
   v 1.0
   
   Lua 5.2 compatible
   
   Only Saves Tables, Numbers and Strings
   Insides Table References are saved
   Does not save Userdata, Metatables, Functions and indices of these
   ----------------------------------------------------
   table.save( table , filename )
   
   on failure: returns an error msg
   
   ----------------------------------------------------
   table.load( filename or stringtable )
   
   Loads a table that has been saved via the table.save function
   
   on success: returns a previously saved table
   on failure: returns as second argument an error msg
   ----------------------------------------------------
   
   Licensed under the same terms as Lua itself.
]]--

   -- declare local variables
   --// exportstring( string )
   --// returns a "Lua" portable version of the string
   local function exportstring( s )
      return string.format("%q", s)
   end

   --// The Save Function
   function table.save(  tbl,filename )
      local charS,charE = "   ","\n"
      local file,err = io.open( filename, "wb" )
      if err then return err end

      -- initiate variables for save procedure
      local tables,lookup = { tbl },{ [tbl] = 1 }
      file:write( "return {"..charE )

      for idx,t in ipairs( tables ) do
         file:write( "-- Table: {"..idx.."}"..charE )
         file:write( "{"..charE )
         local thandled = {}

         for i,v in ipairs( t ) do
            thandled[i] = true
            local stype = type( v )
            -- only handle value
            if stype == "table" then
               if not lookup[v] then
                  table.insert( tables, v )
                  lookup[v] = #tables
               end
               file:write( charS.."{"..lookup[v].."},"..charE )
            elseif stype == "string" then
               file:write(  charS..exportstring( v )..","..charE )
            elseif stype == "number" then
               file:write(  charS..tostring( v )..","..charE )
           elseif stype == "boolean" then
               file:write(  charS..tostring( v )..","..charE )
           end          
         end

         for i,v in pairs( t ) do
            -- escape handled values
            if (not thandled[i]) then
            
               local str = ""
               local stype = type( i )
               -- handle index
               if stype == "table" then
                  if not lookup[i] then
                     table.insert( tables,i )
                     lookup[i] = #tables
                  end
                  str = charS.."[{"..lookup[i].."}]="
               elseif stype == "string" then
                  str = charS.."["..exportstring( i ).."]="
               elseif stype == "number" then
                  str = charS.."["..tostring( i ).."]="
               elseif stype == "boolean" then
                  str = charS.."["..tostring( i ).."]="
               end                  
            
               if str ~= "" then
                  stype = type( v )
                  -- handle value
                  if stype == "table" then
                     if not lookup[v] then
                        table.insert( tables,v )
                        lookup[v] = #tables
                     end
                     file:write( str.."{"..lookup[v].."},"..charE )
                  elseif stype == "string" then
                     file:write( str..exportstring( v )..","..charE )
                  elseif stype == "number" then
                     file:write( str..tostring( v )..","..charE )
                  elseif stype == "boolean" then
                     file:write( str..tostring( v )..","..charE )
                  end                  
               end
            end
         end
         file:write( "},"..charE )
      end
      file:write( "}" )
      file:close()
   end
   
   --// The Load Function
   function table.load( sfile )
      local ftables,err = loadfile( sfile )
      if err then return _,err end
      local tables = ftables()
      for idx = 1,#tables do
         local tolinki = {}
         for i,v in pairs( tables[idx] ) do
            if type( v ) == "table" then
               tables[idx][i] = tables[v[1]]
            end
            if type( i ) == "table" and tables[i[1]] then
               table.insert( tolinki,{ i,tables[i[1]] } )
            end
         end
         -- link indices
         for _,v in ipairs( tolinki ) do
            tables[idx][v[2]],tables[idx][v[1]] =  tables[idx][v[1]],nil
         end
      end
      return tables[1]
   end
