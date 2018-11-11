require 'utils'

--function addCHEmbeddings(inp_infos,path_CH_file)
--    
--    for i=1, #inp_infos do
--        local ACC_NUMB_list = tds.Vec()
--        ACC_NUMB_list:insert(inp_infos[i].acc_numb)
--        local CH_embeddings = readTextEmbedding(path_CH_file,ACC_NUMB_list)[inp_infos[i].acc_numb]
--        if CH_embeddings ~= nil then
--            inp_infos[i].CH_embeddings = CH_embeddings
--        end 
--    end    
--    return inp_infos
--end
--
--function readTextEmbedding(path,ACC_NUMs)
--    require 'hdf5'
--
--    local myFile = hdf5.open(path, 'r')
--    local data = myFile:all()
--    myFile:close()
--    
--    if ACC_NUMs == nil then    
--        return  tds.Hash(data)
--    else
--        local result = tds.Hash()
--        for i=1, #ACC_NUMs do
--            local acc_numb = ACC_NUMs[i]
--            if data[acc_numb] ~= nil then
--                result[acc_numb] = data[acc_numb]
--            end
--        end
--        return result
--    end
--end

local function checkLocalBiggerVersion(path_dir,image_dims)
    local sub_dirs = scandir_dirs(path_dir)
    local smaller_dims = 100000
    local sub_dir_smaller = nil
    
    for i=1,#sub_dirs do
          
        local var = split_string(sub_dirs[i],'x')
        if tonumber(var[1]) ~= nil and tonumber(var[2]) ~= nil and tonumber(var[1]) > image_dims[2] and tonumber(var[2]) > image_dims[3] then
            if smaller_dims > tonumber(var[1]) then
                smaller_dims = tonumber(var[1])
                sub_dir_smaller = sub_dirs[i]
            end
        end
    end
    return sub_dir_smaller
end


local function downloadImage(remote_path,local_path,img_dim_string,background_color)
    -- check if data dir exists

    if directory_exists(getDirPath(local_path)) == nil then
        os.execute('mkdir -p ' .. getDirPath(local_path))   
    end 
    
    local command = ''
    if background_color ~= nil then
        command = 'convert -gravity center -background ' .. background_color .. ' -resize ' .. img_dim_string .. ' -extent ' .. img_dim_string .. ' '  .. remote_path .. " " .. local_path 
        error('this line of code is not yet tested, it should be impossible to arrive here')
    else   
        command = 'convert -resize ' .. img_dim_string .. '^ '  .. remote_path .. " " .. local_path       

    end
    if os.execute(command) == 1 then
        print("Error: unable to performe this command: " .. command)
        os.exit(0)    
    end    
end


local function insertBlackBox(img,color,box)
    local img_w = img:size(3)
    local img_h = img:size(2)

    if box == nil then

        local box_w = math.random(img_w/64,img_w/8)
        local box_h = math.random(img_h/64,img_h/8)
        local x1,y1
        local random_pos = math.random()
        if random_pos < 0.33 then
            x1 = math.random(0,img_w/8 - box_w)
            y1 = math.random(0,img_h - box_h)  
        elseif random_pos < 0.66 then
            x1 = math.random(img_w - img_w/8 ,img_w - box_w)
            y1 = math.random(0,img_h - box_h)          
        else
            local random_pos2 = math.random()
            if random_pos2>0.5 then
                x1 = math.random(0,img_w/4 - box_w)
                y1 = math.random(0,img_h/8 - box_h)
            else
                x1 = math.random(img_w - img_w/4 ,img_w - box_w)
                y1 = math.random(0,img_h/8 - box_h)            
            end               
        end
        box  =  {
            x1 = x1,
            y1 = y1,
            x2 = x1 + box_w,
            y2 = y1 + box_h,
            prob = 1        
        }
    else
        -- check if the box is on the border
        if (box.prob == nil or box.prob < 0.1)
            or (box.x1 > img_w/4 and box.x1 < img_w/4) 
            or (box.x2 > img_w/4 and box.x2 < img_w/4)
            or (box.y1 > img_h/4 and box.y1 < img_h/4)
            or (box.y2 > img_h/4 and box.y2 < img_h/4)  then
            return img, 0 --not modified, the box was in the center
        end
    end
    
    box.x1 = math.floor(math.min(math.max(0,box.x1),img_w))
    box.y1 = math.floor(math.min(math.max(0,box.y1),img_h))
    box.x2 = math.floor(math.min(math.max(box.x1+1,box.x2),img_w))
    box.y2 = math.floor(math.min(math.max(box.y1+1,box.y2),img_h))

    img[{{},{box.y1+1,box.y2},{box.x1 + 1,box.x2}}] = color
    return img, 1 
end

local function rescaleBoxes(boxes_list,current_size)
    local h = current_size[2]
    local w = current_size[3]
    
    if boxes_list == nil then
        return tds.Vec()
    end 
    
    local new_boxes_list = tds.Vec()
    for i = 1,#boxes_list do
--        print("Before: " .. boxes_list[i].x1 .. "-"..boxes_list[i].y1 .. "_" .. boxes_list[i].x2 .. "-"..boxes_list[i].y2)
        if (boxes_list[i].x2 > 0) and
           (boxes_list[i].y2 > 0) and
           (boxes_list[i].x1 < 1) and
           (boxes_list[i].y1 < 1) then
                
        
            local new_box = {
                x1 = boxes_list[i].x1 * w,
                y1 = boxes_list[i].y1 * h,
                x2 = boxes_list[i].x2 * w,
                y2 = boxes_list[i].y2 * h,
                prob = boxes_list[i].prob }
    --        print("After: " .. new_box.x1 .. "-"..new_box.y1 .. "_" .. new_box.x2 .. "-"..new_box.y2)
            new_boxes_list:insert(tds.Hash(new_box))
        end
    end 

    return(new_boxes_list)   
end

local function loadAndRescale(img_path,final_size,box_color,BBs)
   -- Rescale and crop the image in order to preserve the original proportion
   local ok, input = pcall(image.load, img_path)
   if not ok then
      print('Error, check img ' .. img_path)
      return nil
   end   
--   local input = image.load(img_path,final_size[1])
    
   local w_input = input:size(3) --2500
   local h_input = input:size(2) --2415
   
--   print(w_input .. "x" .. h_input)
   if w_input/h_input > final_size[2]/final_size[3] then  -- if 2500/2415 > 2500/2048
      input = image.scale(input, math.floor(w_input * final_size[3] / h_input),final_size[3],'bilinear')
   else
      input = image.scale(input,final_size[2], math.floor(h_input * final_size[2] / w_input),'bilinear')  -- 2500 x (2415 * 2500 / 2500)   
   end

   if box_color then
        -- add Black Bounding Boxes
        BBs = rescaleBoxes(BBs,input:size())
        local BB_inserted = 0 
    
        -- 1) add Black Bounding Boxes on text predicted area
        for i = 1,#BBs do
            local mod_flag = 0
            input,mod_flag = insertBlackBox(input,box_color,BBs[i])
            if mod_flag == 1 then
                BB_inserted = BB_inserted + 1
            end
        end
   end   
   collectgarbage()
   return input
end

local function loadRescaleAndPad(img_path,final_size,flags,box_color,BBs)
   local ok, input = pcall(image.load, img_path)
   if not ok then
      print('Error, check img ' .. img_path)
      return nil
   end 
    
   local w_input = input:size(3) --2500
   local h_input = input:size(2) --2415
   

    if flags.aug then
        --TODO: add randomness also during the rescaling        
       if torch.rand(1)[1] > 0.5 then
          -- invert black-white
          input = torch.abs(input:add(-1))
       end  
    end
    
   if w_input/h_input > final_size[2]/final_size[3] then  -- if 2500/2415 > 2500/2048
      input = image.scale(input,final_size[2], math.floor(h_input * final_size[2] / w_input),'bilinear')  -- 2500 x (2415 * 2500 / 2500)    
   else    
      input = image.scale(input, math.floor(w_input * final_size[3] / h_input),final_size[3],'bilinear') 
   end

    if box_color then
        -- add Black Bounding Boxes
        BBs = rescaleBoxes(BBs,input:size())
        local BB_inserted = 0 
    
        -- 1) add Black Bounding Boxes on text predicted area
        for i = 1,#BBs do
            local mod_flag = 0
            input,mod_flag = insertBlackBox(input,box_color,BBs[i])
            if mod_flag == 1 then
                BB_inserted = BB_inserted + 1
            end
        end
    
        -- 2) add random Black Bounding Boxes (only on the image border)
        if flags.aug then
           if BB_inserted < 9 then
             local num_random_boxes = math.random(2, 10 - #BBs)
             for i=1,num_random_boxes do
                input = insertBlackBox(input,box_color)     
             end
           end
        end   
    end
    
   local iW = final_size[2] --input:size(3)
   local iH = final_size[3] --input:size(2)

   local oW = input:size(3)
   local oH = input:size(2)
   local h1 = math.floor((iH-oH)/2)
   local w1 = math.floor((iW-oW)/2)
       
   if flags.aug then
       -- Augment data 
       if flags.flip and torch.rand(1)[1] > 0.5 then
          input = image.hflip(input)
       end
   
       local deg2rad = 0.0174532925
       -- rotation
       -- IMP: image.rotate has some problem better use graphicsmagick
       -- graphicsmagick needed valid image though
       local rot = 0
       if torch.rand(1)[1] > 0.4 then
         rot = torch.uniform(-20, 20)
         if rot > 3 then
             rot = rot
         elseif rot < -3 then
             rot = (360 + rot)
         else
             rot = 0
         end
       end
       if rot ~=0 then
          rot = rot * deg2rad
          input = image.rotate(input, rot, 'bilinear')
       end
        h1 = math.floor(torch.uniform((iH-oH)/4,(iH-oH)/2))
        w1 = math.floor(torch.uniform((iW-oW)/4,(iW-oW)/2)) 
               
   end
   local result = torch.FloatTensor(1,iH,iW)
    if flags.aug then
        local rand_numb = torch.rand(1)[1]        
        if box_color then     
            if rand_numb < 1/3 then
                result:fill(box_color)
            elseif rand_numb < 2/3 then
                result:fill(0)            
            else
                result:fill(1)            
            end
        else
            if rand_numb < 0.5 then
                result:fill(0)            
            else
                result:fill(1)            
            end                
       end 
   else
       if box_color then
        result:fill(box_color)
       else
        result:fill(0)
       end
   end
   result[{{},{h1+1,h1+oH},{w1+1,w1+oW}}]:copy(input)
    
--   local filename = split_string(img_path,'/')[#split_string(img_path,'/')] 
--   image.save(filename,result)

   collectgarbage()
   return result
end

local function loadRescaleAndCrop(img_path,final_size,flags,box_color,BBs)

   local ok, input = pcall(image.load, img_path)
   if not ok then
      print('Error, check img ' .. img_path)
      return nil
   end 
   
--   local img_load_time = dataTimer:time().real
--   dataTimer:reset()
   
    local w_input = input:size(3) --2500
    local h_input = input:size(2) --2415

    if flags.aug then
        --TODO: add randomness also during the rescaling      
       if torch.rand(1)[1] > 0.5 then
          -- invert black-white
          input = torch.abs(input:add(-1))
       end         
    end

   if w_input/h_input > final_size[2]/final_size[3] then  -- if 2500/2415 > 2500/2048
      input = image.scale(input, math.floor(w_input * final_size[3] / h_input),final_size[3],'bilinear')
   else
      input = image.scale(input,final_size[2], math.floor(h_input * final_size[2] / w_input),'bilinear')  -- 2500 x (2415 * 2500 / 2500)   
   end

    if box_color then
        -- add Black Bounding Boxes
        BBs = rescaleBoxes(BBs,input:size())
        local BB_inserted = 0 
    
        -- 1) add Black Bounding Boxes on text predicted area
        for i = 1,#BBs do
            local mod_flag = 0
            input,mod_flag = insertBlackBox(input,box_color,BBs[i])
            if mod_flag == 1 then
                BB_inserted = BB_inserted + 1
            end
        end
    
        -- 2) add random Black Bounding Boxes (only on the image border)
        if flags.aug then
           if BB_inserted < 9 then
             local num_random_boxes = math.random(2, 10 - #BBs)
             for i=1,num_random_boxes do
                input = insertBlackBox(input,box_color)     
             end
           end
        end   
    end

   local iW = input:size(3)
   local iH = input:size(2)

   -- take the central crop
   local oW = final_size[2]
   local oH = final_size[3]
   local h1 = math.floor((iH-oH)/2)
   local w1 = math.floor((iW-oW)/2)
       
   if flags.aug then
       -- Augment data 
       if flags.flip and torch.rand(1)[1] > 0.5 then
          input = image.hflip(input)
       end
       local deg2rad = 0.0174532925
       -- rotation
       -- IMP: image.rotate has some problem better use graphicsmagick
       -- graphicsmagick needed valid image though
       local rot = 0
       if torch.rand(1)[1] > 0.4 then
         rot = torch.uniform(-20, 20)
         if rot > 3 then
             rot = rot
         elseif rot < -3 then
             rot = (360 + rot)
         else
             rot = 0
         end
       end
       if rot ~=0 then
          rot = rot * deg2rad
          input = image.rotate(input, rot, 'bilinear')
       end
       
        
        -- random crop

        h1 = math.floor(torch.uniform((iH-oH)/4,(iH-oH)/2))
        w1 = math.floor(torch.uniform((iW-oW)/4,(iW-oW)/2))       
   end
 
--   local img_aug_time = dataTimer:time().real


   collectgarbage()

   return input[{{},{h1+1,h1+oH},{w1+1,w1+oW}}]--,img_load_time,img_aug_time
end

function getImage(input_elem, info)

    local path = input_elem.path
    local final_size = input_elem.classes
    local BBs = input_elem.rects
    local script_opt = nil
    local box_color = nil
    
    local flags = {
        aug = false,
        flip = false    
    }
    

    if info then
        script_opt = info.opt
        if info.opt.text_BBs_file ~= "" then
            box_color = info.params['mean'][1]    
        end
        flags.aug = info.aug_flag
    else
        script_opt = opt        
    end

    local dataTimer = nil
    
    if script_opt.verbose >= 2 then   
        dataTimer = torch.Timer()
        dataTimer:reset()
    end

    if flags.aug then 
        flags.flip = true  
        for i =1,#info.to_not_flip_classes do
            if table.contains(info.classes,info.to_not_flip_classes[i]) then
                flags.flip = false 
                break   
            end
        end
    end

    
    local img_dim_string = script_opt.downloadImage_size[2] .. 'x' .. script_opt.downloadImage_size[3]
    
    local var = split_string(path,"/")
    local local_path = script_opt.path_temp  .. img_dim_string .. '/' .. var[#var]
    
    if file_exists(local_path) == false then   
        local dir_highres_name = checkLocalBiggerVersion(script_opt.path_temp,script_opt.downloadImage_size)
        
        if dir_highres_name ~= nil then
            local local_path_other_res = script_opt.path_temp .. dir_highres_name .. '/' .. var[#var]
            if file_exists(local_path_other_res) == false then
                if script_opt.verbose >= 2 then print("+++ Downloading from " .. path) end      
                downloadImage(path,local_path,img_dim_string)
                if info.opt.verbose >= 2 then print((">Downloading time = %.2f seconds<"):format(dataTimer:time().real)) end    
            else
                if script_opt.verbose >= 2 then print("+++ Downloading from " .. local_path_other_res) end
                downloadImage(local_path_other_res,local_path,img_dim_string)
                if info.opt.verbose >= 2 then print((">Downloading time = %.2f seconds<"):format(dataTimer:time().real)) end
                --local_path = local_path_other_res
            end
        else
            if script_opt.verbose >= 2 then print("+++ Downloading from " .. path) end
            downloadImage(path,local_path,img_dim_string)
            if info.opt.verbose >= 2 then print((">Downloading time = %.2f seconds<"):format(dataTimer:time().real)) end  
        end  
    end 

    if script_opt.fixed_size_input == nil or script_opt.fixed_size_input == 1 then --crop_image 
        if script_opt.fixed_size_input_mode == nil or script_opt.fixed_size_input_mode == 'crop' then
            return loadRescaleAndCrop(local_path,script_opt.input_size,flags,box_color,BBs)
        elseif script_opt.fixed_size_input_mode == 'pad' then
            return loadRescaleAndPad(local_path,script_opt.input_size,flags,box_color,BBs)
        elseif script_opt.fixed_size_input_mode == 'pad_or_crop' then
            -- this kind of loading is used only for training augmentation, during validation we use the standard central crop
            if flags.aug then 
                if torch.rand(1)[1] > 0.5 then return loadRescaleAndCrop(local_path,script_opt.input_size,flags,box_color,BBs) 
                else return loadRescaleAndPad(local_path,script_opt.input_size,flags,box_color,BBs) end 
            else
                return loadRescaleAndCrop(local_path,script_opt.input_size,flags,box_color,BBs) 
            end
        else
            error('wrong value for script_opt.fixed_size_input_mode = '.. script_opt.fixed_size_input_mode)      
        end
    else
        return loadAndRescale(local_path,script_opt.input_size,box_color,BBs)    
    end
end

function fromClassLabelToString(label,classes)
    local result_string = nil
    
    for i=1,#classes do
        if label[i] == 1 then
           if result_string == nil then result_string = classes[i]
           else result_string = result_string .. "-" .. classes[i] end
        end
    end
    
    if result_string == nil then
        return "normal"
    else
        return result_string
    end    
end

function getLabelMulti(instance,classes_used, info,noisy_flag)
    local script_opt = nil
    if info then
        script_opt = info.opt
    else
        script_opt = opt        
    end

    
    local label = nil
    if script_opt.predict_age then
        label = torch.FloatTensor(#classes_used+1):fill(2)
    else
        label = torch.FloatTensor(#classes_used):fill(2)
    end    
--    for j=1,#classes_used do
--        label[((j-1)*2)+1] = 0
--        label[((j-1)*2)+2] = 1    
--    end

    local instance_classes = instance.classes
    if noisy_flag ~= nil then
        assert(instance.noisy_classes~=nil,instance)
        instance_classes = instance.noisy_classes
    end
    
    for i=1,#instance_classes do
        for j=1,#classes_used do
            if instance_classes[i] == classes_used[j] then
--                label[((j-1)*2)+1] = 1
--                label[((j-1)*2)+2] = 0
                  label[j] = 1
            end
        end
    end
    
    if script_opt.predict_age then
        label[label:size(1)] = instance.age
        return label
    else
        return label
    end
end        

function getLabelsSequence(el_classes,to_learn_classes)
    local labels_sequence = torch.FloatTensor(#to_learn_classes)
    local el_count = 0
    for i = 1,#to_learn_classes do
        if table.contains(el_classes,to_learn_classes[i],true) then
            el_count = el_count + 1
            labels_sequence[el_count] = i 
--            print(to_learn_classes[i])
        end
        
    end
    
    -- Fill the vector with stop signal
    for i = el_count + 1,#to_learn_classes do
        labels_sequence[i] = #to_learn_classes + 1   
    end
--    print(labels_sequence)
    return labels_sequence
end

function getLabelSingle(all_class,string_class)

    for i = 1, #all_class do
        if all_class[i] == string_class then return i end
    end
    
    error("the classes table do not contains the " .. string_class .." class")
end

local function getMinElementsClass(in_table,element)
    local min_elements = nil
    local class_min_elements = nil
    
    -- check if cl_list1 is a subset of cl_list2
    local function isASubset(cl_list1,cl_list2)
        for i=1,#cl_list1 do
            local element_ok = false
            for j=1,#cl_list2 do
                if cl_list1[i] == cl_list2[j] then
                    element_ok = true
                end        
            end
            if not element_ok then return false end
        end
        return true
    end
    
    for k,v in pairs(in_table) do 
        local split_result = split_string(k,"-")
        if isASubset(split_result, element.classes) and #split_result == 1 then
            if min_elements == nil or min_elements > #v then
                min_elements = #v
                class_min_elements = k                
            end
        end
    end
    return class_min_elements
end

local function downloadSet(set,info)
    for i=1,#set do
        getImage(set[i].path,nil,info)
    end
end

local function existNonOverlappingClass(c1,c2)
    local res = {}
   
    for i=1,#c1 do   
        for j=1,#c2 do       
            if c1[i] == c2[j] then
                table.insert(res,c1[i])
                if #res > 1 then
                    return nil
                end
            end
        end
    end

    return res[1]
end

function filterElementWithOverlappingLabels(to_filter_table,classes)
    local out_table = tds.Vec()
    for i=1,#to_filter_table do
        local single_class = existNonOverlappingClass(to_filter_table[i].classes,classes)
--        print(to_split_table[i])
        if single_class then
            out_table:insert(to_filter_table[i])
        end
    end
    return out_table
end

-- splits the input table in sub-tables (for all the input classes)
-- all the elements with overlapping  classes are ignored if "ordered_classes_flag" is equal to nil
-- while it is inserted in the set related to the class with less example if "ordered_classes_flag"
-- is not nil (in that case the variable "classes" contains the name of the classes ordered by number 
-- of elements in the training set)
function splitPerClass(to_split_table, classes, ordered_classes_flag)

    local out_table = tds.Hash()
    for i=1,#classes do
        out_table[classes[i]] = tds.Vec()
    end
    
    if ordered_classes_flag == nil then
        for i=1,#to_split_table do
            
            local single_class = existNonOverlappingClass(to_split_table[i].classes,classes)
    --        print(to_split_table[i])
            if single_class then
                out_table[single_class]:insert(to_split_table[i])
            end
        end
    else
--        print(classes)
        for i=1,#to_split_table do
            for j=1,#classes do
                if table.contains(to_split_table[i].classes,classes[j]) then
--                    print(to_split_table[i])
                    out_table[classes[j]]:insert(to_split_table[i])
                    break
                end
            end
        end    
    end
--    for k,v in pairs(out_table) do
--        print(k .. ": " .. #v)
--    end
--    
    return(out_table)
end


-- splits the input table in sub-tables (for each possible classes overlap)
-- all the overlapping class with less than 3000 elements are subpressed 
-- all the single class with less than 1000 elements are subpressed if "flag_merge" is equal to true
function splitPerClassWithOverlap(in_tbl,classes,flag_merge)
    local out_tbl = tds.Hash()  
    
    if #classes > 1 then
        local filt_classes = table.find_and_remove(classes,"normal")
        for i=1,#in_tbl do
            local label = fromClassLabelToString(getLabelMulti(in_tbl[i],filt_classes),filt_classes)
            if out_tbl[label] == nil then 
                out_tbl[label] = tds.Vec()
            end    
            out_tbl[label]:insert(in_tbl[i]) 
        end
    
        local function smallerClass(table_in,num_elm)
            if num_elm == nil then num_elm = 1 end
            local min_k = nil
            for k,v in pairs(table_in) do
                local split = split_string(k,"-")
                if #split > num_elm and (min_k == nil or #v < #table_in[min_k]) then
                    min_k = k
                end
            end
            return min_k
        end
        -- cycle all over the train data and put all the list with less than 3000 elemnts in other bigger classes
        while true do
            local key = smallerClass(out_tbl)
            
            if key ~= nil and #out_tbl[key] < 3000 then
                for i=1,#out_tbl[key] do
                    out_tbl[getMinElementsClass(out_tbl,out_tbl[key][i])]:insert(out_tbl[key][i]) 
                end
                out_tbl[key] = nil 
            else 
                break                   
            end
        end
        
        -- merge all the list with less than 1000 elements in bigger listes
        if flag_merge then
            while true do
                local key = smallerClass(out_tbl,0)
                
                if key ~= nil and #out_tbl[key] < 1000 then
                    
                    local new_list = tds.Vec()
                    for i=1,#out_tbl[key] do
                        new_list:insert(out_tbl[key][i]) 
                    end
                    out_tbl[key] = nil 
                    local key2 = smallerClass(out_tbl,0) 
                    for i=1,#out_tbl[key2] do
                        new_list:insert(out_tbl[key2][i]) 
                    end  
                    out_tbl[key2] = nil
                    out_tbl[(key .. "/" .. key2)] = new_list          
                else 
                    break                   
                end
            end    
        end
    else
        out_tbl["normal"] = tds.Vec()
        out_tbl[classes[1]] = tds.Vec()
        for i=1,#in_tbl do
            
            if table.contains(in_tbl[i].classes,classes[1]) then 
                out_tbl[classes[1]]:insert(in_tbl[i])   
            else
                out_tbl["normal"]:insert(in_tbl[i]) 
            end    

        end    
    end
    
    local tbl_keys = tds.Vec()
    if opt.verbose >= 1 then print("-->>Classes overlaps: \n") end
    for k,v in pairs(out_tbl) do
        if opt.verbose >= 1 then print(k .." = " .. #v) end
        tbl_keys:insert(k)
    end    
    
    return out_tbl,tbl_keys     
end


-- Return a table with elements with at least one class contained in the "classes" input parameter
function classBasedFiltering(tbl,classes)
    local out_tbl = tds.Vec()
    if #classes > 1 then     
        for i=1,#tbl do
            for j=1,#classes do
                if table.contains(tbl[i].classes,classes[j]) then
                    out_tbl:insert(tbl[i])
                    break
                end
            end
        end     
    else
        for i=1,#tbl do
            
            if table.contains(tbl[i].classes,classes[1]) then 
                tbl[i].classes = tds.Vec({classes[1]})  
            else
                tbl[i].classes = tds.Vec({'normal'})  
            end    
            out_tbl:insert(tbl[i])
        end 
    end
    return out_tbl
end

function excludeGLInfosFromTrain(train_infos,gl_infos)
    local train_res,gl_res = tds.Vec(),tds.Vec()
    
    local gl_infos_hash = tds.Hash()
    for i=1,#gl_infos do
        gl_infos_hash[gl_infos[i]["id"]] = i--gl_infos[i] 
    end
    
    for i=1,#train_infos do
        if gl_infos_hash[train_infos[i]["id"]] == nil then
            train_res:insert(train_infos[i])
        else
            local gl_elm = gl_infos[gl_infos_hash[train_infos[i]["id"]]]
            gl_elm.noisy_classes = train_infos[i].classes    
            gl_res:insert(gl_elm)
        end
    end
    
    return train_res,gl_res
end

function getCH_embeddings(CH_file_path,params)
    if file_exists(CH_file_path) then
        local myFile = hdf5.open(CH_file_path, 'r')
        local ok, data = pcall(myFile.all,myFile)
        myFile:close()
        if not ok or data['CH_embedding']:dim() ~= 2 then
            return torch.FloatTensor(1,1):fill(0)
        else
            preprocess_text(data['CH_embedding'],params['text_emb_mean'], params['text_emb_var'])
            return data['CH_embedding']
        end
    else 
        return torch.FloatTensor(1,1):fill(0)
    end
end

function getCovariateVariables(img_info,opt_tbl)
    local vec_dim = 0
    if opt_tbl[1] == 1 then
        vec_dim = vec_dim + 1
    end
    if opt_tbl[2] == 1 then
        vec_dim = vec_dim + 2
    end
    if opt_tbl[3] == 1 then
        vec_dim = vec_dim + 2
    end        
    
    local covariate_variables = torch.FloatTensor(vec_dim)
    local vec_index = 1
    
    if opt_tbl[1] == 1 then
        covariate_variables[1] =  (img_info.age - 0.42208716184106015) / 0.22091466121167486 --age normalized age_norm = (age -mean)/std
        vec_index = vec_index + 1
    end
    if opt_tbl[2] == 1 then
        if img_info.view == "AP" then
            covariate_variables[vec_index] =  1
            covariate_variables[vec_index+1] =  0
        elseif img_info.view == "PA" then
            covariate_variables[vec_index] =  0
            covariate_variables[vec_index+1] =  1
        else
            covariate_variables[vec_index] =  0
            covariate_variables[vec_index+1] =  0                    
        end
        vec_index = vec_index+2
    end
    if opt_tbl[3] == 1 then
        if img_info.sex == "M" then
            covariate_variables[vec_index] =  1
            covariate_variables[vec_index+1] =  0
        elseif img_info.view == "F" then
            covariate_variables[vec_index] =  0
            covariate_variables[vec_index+1] =  1
        else
            covariate_variables[vec_index] =  0
            covariate_variables[vec_index+1] =  0                    
        end 
    end    
        
    return covariate_variables
end

if opt~=nil and 
  (split_string(opt.input_csv_dir,"/")[#split_string(opt.input_csv_dir,"/")] == "split_v10_15_Mar_17_mauro"   or
   split_string(opt.input_csv_dir,"/")[#split_string(opt.input_csv_dir,"/")] == "split_v11_7_June_7_emanuele" or
   split_string(opt.input_csv_dir,"/")[#split_string(opt.input_csv_dir,"/")] == "split_v12_27_Jun_17_mauro"   or
   split_string(opt.input_csv_dir,"/")[#split_string(opt.input_csv_dir,"/")] == "split_v13_4_July_emanuele"   or
   split_string(opt.input_csv_dir,"/")[#split_string(opt.input_csv_dir,"/")] == "v11" or    
   split_string(opt.input_csv_dir,"/")[#split_string(opt.input_csv_dir,"/")] == "original_labels")   then

    function read_csv(path)
        
        local res = tds.Vec()
        local res2 = tds.Hash()
        local sep = ',%c'
--        local class_counter = tds.Hash()
        local file = assert(io.open(path, "r"))
        for line in file:lines() do
    
            local fields = split_string(line, sep)
            local path_fields = split_string(fields[6], ".")
            assert(path_fields[#path_fields] == "png", 'Error: ' .. fields[6] .. " is not a png image")
            
            local age_in_days = tonumber(fields[3])
            
            -- only people with more than 16 years old
            if age_in_days ~= nil and age_in_days > 365 * opt.ageLimit then
                local element = {
                    path = fields[6],
                    acc_numb = fields[2],
                    id = tonumber(fields[1]),
                    classes = {},
                    sex =  fields[5],    
                    age = age_in_days/365/120,
                    view = fields[7]
                }
                for i = 8,#fields do
                    element.classes[i-7] = fields[i]
                end
                if #element.classes > 0 and not (#element.classes == 1 and element.classes[1] == 'unclassifiable') then
--                    for j=1,#element.classes do
--                        local cl = element.classes[j]
--                        if class_counter[cl] == nil then class_counter[cl] = 0 end
--                        class_counter[cl] = class_counter[cl] + 1
--                    end
                    if opt.remove_exams_overlapping_classes == 1 then
                        if #element.classes == 1 then
                            res:insert(tds.Hash(element))  
                            res2[element.acc_numb] = tds.Hash(element)
                        end
                    else 
                        res:insert(tds.Hash(element))
                        res2[element.acc_numb] = tds.Hash(element)
                    end
                end
            end        
        end
        
        file:close()
--        for k,v in pairs(class_counter) do
--            print(k,v)
--        end
--        print(#res)
        return res,res2
    end

else

    function read_csv(path)
        local res = tds.Vec()
        local res2 = tds.Hash()
        local sep = ',%c'

        local file = assert(io.open(path, "r"))
        for line in file:lines() do
    
            local fields = split_string(line, sep)
            local path_fields = split_string(fields[6], ".")
            assert(path_fields[#path_fields] == "png", 'Error: ' .. fields[6] .. " is not a png image")
            
            local age_in_days = tonumber(fields[3])
            
            -- only people with more than 16 years old
            if age_in_days ~= nil and age_in_days > 365 * opt.ageLimit then
                local element = {
                    path = fields[6],
                    id = tonumber(fields[1]),
                    acc_numb = split_string(fields[6], "/")[#split_string(fields[6], "/")-1],
                    classes = {},
                    sex =  fields[5],    
                    age = age_in_days/365/120
                }
                for i = 7,#fields do
                    element.classes[i-6] = fields[i]
                end
                if #element.classes > 0 or (#element.classes == 1 and element.classes[1] == 'unclassifiable') then
                    if opt.remove_exams_overlapping_classes == 1 then
                        if #element.classes == 1 then
                            res:insert(tds.Hash(element))
                            res2[element.acc_numb] = tds.Hash(element)    
                        end
                    else 
                        res:insert(tds.Hash(element))
                        res2[element.acc_numb] = tds.Hash(element)
                    end
                end
            end        
        end
        
        file:close()
        
        return res,res2
    end
    
end

function readIDL(IDL_path)
    local file = assert(io.open(IDL_path, "r"))
    local BBoxes_table = tds.Hash()
    local current_img
    local i = 1

    print("Reading ILD file...")
  
    for line in file:lines() do
        i = i + 1
        local splitted_s = split_string(split_string(line, ';')[1],'"')
        local path = splitted_s[1]
        local rect_strings = {}
        local rects = tds.Vec()

        if #splitted_s > 1 then
            rect_strings = split_string(splitted_s[2],': ( , )')
        end    
        for i=1,#rect_strings,6 do
            local rect = {
                x1 = tonumber(rect_strings[i+1]),
                y1 = tonumber(rect_strings[i+2]),
                x2 = tonumber(rect_strings[i+3]),
                y2 = tonumber(rect_strings[i+4]),
                prob = tonumber(rect_strings[i+5])
            }
            rects:insert(tds.Hash(rect))
        end       
        BBoxes_table[path] = rects      
    end  
    print("ILD file readed.")
    file:close()
    return BBoxes_table
end

function addTextBoxInfos(input_table,text_infos_table) 
    local res = tds.Vec()
    for i=1,#input_table do
        
        local element = {}
        for k,v in pairs(input_table[i]) do
            element[k] = v
        end
        element["rects"] = text_infos_table[input_table[i].path]
        
        res:insert(tds.Hash(element))
    end

    return res
end

local function keep_only_overlaps(tds_vec1,tds_vec2)
    local result = tds.Vec()
    
    for i=1,#tds_vec2 do
        if table.contains(tds_vec1,tds_vec2[i]) then
            result:insert(tds_vec2[i])
        end
    end
    
    return result
end


local function getStringFromVec(tds_vec)
    tds_vec:sort(function(x1,x2) 
                    if x1<x2 then 
                        return true
                    else
                        return false
                    end
                 end
                    )
    local result = ""
    for i=1,#tds_vec do
        if result == "" then
            result = tds_vec[i]
        else
            result = result .. "/" .. tds_vec[i]
        end
        
    end
    return result
end


function splitInTrainAndValidation(input_vec,classes)
    local train,val = tds.Vec(),tds.Vec()
    local per_class_set = tds.Hash()

    for i=1,#input_vec do
        local current_elm_classes = getStringFromVec(keep_only_overlaps(input_vec[i].classes,classes))
        if per_class_set[current_elm_classes] == nil then
            per_class_set[current_elm_classes] = tds.Vec()
        end
        per_class_set[current_elm_classes]:insert(input_vec[i])
    end
    
    
    -- val set will be 10% of the total data and the data distribution should be proportional accross all classes
    for k,v in pairs(per_class_set) do
        local val_elements_num
        if #v>= 10 then
            if torch.uniform() > 0.5 then
                val_elements_num = math.floor(#v*0.1)
            else
                val_elements_num = math.ceil(#v*0.1)
            end
        else
            if torch.uniform() > 0.5 * (#v - 1)/#v then
                val_elements_num = math.floor(#v*0.1)
            else
                val_elements_num = math.ceil(#v*0.1)
            end
        end
        for i=1,#v do
            if i<= val_elements_num then
                val:insert(v[i])
            else
                train:insert(v[i])
            end
        end
    end
 
    return train,val
end

function loadGLInfosAndFilterThenFromTrainingSet(train_infos,gl_filepath,classes,IDL_infos)
    local gl_infos,new_train_infos = read_csv(gl_filepath),nil
    new_train_infos,gl_infos = excludeGLInfosFromTrain(train_infos,gl_infos)
    gl_infos = classBasedFiltering(gl_infos,classes)
    if IDL_infos ~= nil then gl_infos = addTextBoxInfos(gl_infos,IDL_infos) end
    local gl_train_infos, gl_val_infos = splitInTrainAndValidation(gl_infos,classes)
    return new_train_infos,gl_train_infos, gl_val_infos
end   

function computesRNNWeights(classes,num_occurrency,dim_train_set)
    local weights_criterion = torch.FloatTensor(#classes+1)
    for i=1,#classes do
        weights_criterion[i] = dim_train_set/num_occurrency[classes[i]]/(#classes+1)
        if opt.verbose >= 1 then print("criterion weight for " .. classes[i] .. ": " .. weights_criterion[i]) end
    end
    weights_criterion[#classes+1] = 1 -- STOP class, weight equal to 1
    if opt.verbose >= 1 then print("criterion weight for stop class: " .. weights_criterion[#classes+1]) end
    return weights_criterion
end

function computeWeightsMonoClass(train_infos,class_name)
    local weights_criterion = torch.FloatTensor(1,2)
    local class_num_elements,total_elements = 0,0
    for cl,v in pairs(train_infos) do
        total_elements = total_elements + #v
        if cl==class_name then
            class_num_elements = #v
        end    
    end
--    if total_elements/class_num_elements > opt.batchSize then
--        weights_criterion[1][1] = opt.batchSize
--        weights_criterion[1][2] = 1/opt.batchSize
--    else
        weights_criterion[1][1] = total_elements/class_num_elements
        weights_criterion[1][2] = 1/(total_elements/class_num_elements)
--    end 
    if opt.verbose >= 1 then print("criterion weight for " .. class_name .. ": " .. weights_criterion[1][1] .. " - " .. weights_criterion[1][2]) end      
end

function computeMultiClassifiersWeights(classes,filtered_classes)             
    local weights_criterion = torch.FloatTensor(#filtered_classes,2)     

    
    for i =1,#filtered_classes do

        local count_classes = 0
        for j = 1, #classes do
            local split_res = split_string(classes[j], "-")
            for k =1, #split_res do 
                if split_res[k] == filtered_classes[i] then
                    count_classes = count_classes + 1
                end
            end    
        end

        weights_criterion[i][1] = #classes / count_classes / 2 -- the 2 represents the binary choice for that class
        weights_criterion[i][2] = #classes / (#classes - count_classes) / 2
        if opt.verbose >= 1 then print("criterion weight for " .. filtered_classes[i] .. ": " .. weights_criterion[i][1] .. " - " .. weights_criterion[i][2]) end   
    end 
    return weights_criterion
end

function computeSingleClassifierWeights(classes,train_infos)            
    local weights_criterion = torch.FloatTensor(#classes):zero()
    local biggest_class_dimension = -1
    for k,v in pairs (train_infos) do
        if biggest_class_dimension < #v then biggest_class_dimension = #v end
    end 
    for i =1,#classes do
        weights_criterion[i] = biggest_class_dimension/#train_infos[classes[i]]
    end
    for i=1,weights_criterion:size(1) do
        if opt.verbose >= 1 then print("criterion weight for " .. i .. ": " .. weights_criterion[classes[i]]) end
    end 
    opt.overlapping_classes = classes   
     
end 

function splitPredictions(preds_vec,classes,opt_flag)
    local result = tds.Hash()
    if opt_flag == nil then opt_flag = "no_flag" end
    
    if opt_flag == "multi_task" then
        if preds_vec:size(1) == (#classes +4) * 2 then
            for i=1,#classes do
                result[classes[i]] = preds_vec[{{(i*2)-1,i*2}}]:clone()
            end 
            local priorities = {"critical","urgent","non_urgent_finding"}
            
            for j=1,#priorities do
                local i = #classes + j
                result[priorities[j]] = preds_vec[{{(i*2)-1,i*2}}]:clone() 
            end 
            result['normal'] = preds_vec[{{(#classes+#priorities)*2+1,(#classes+#priorities+1)*2}}]:clone()        
        else
            error('Input size not supported '..#classes ..'  ' .. preds_vec:size(1))
        end    
    else
        if preds_vec:size(1) == #classes then
            -- single multi-class classifier, each prediction corresponds to a class
            for i=1,#classes do
                result[classes[i]] = preds_vec[i]
            end
        elseif preds_vec:size(1) == #classes * 2 then
            -- multi binary classifiers, each pair in preds_vec corresponds to a class
            for i=1,#classes do
                if opt_flag == "ordinal_regression" then
                    result[classes[#classes - i + 1]] = preds_vec[{{(i*2)-1,i*2}}]:clone() 
                else
                    result[classes[i]] = preds_vec[{{(i*2)-1,i*2}}]:clone()
                end
            end  
        else
            error('Input size not supported '..#classes ..'  ' .. preds_vec:size(1))
        end
    end
    return result
end

function fromVecToClass(binary_vec,priority_list)
    local res = torch.FloatTensor(binary_vec:size(1)):fill(1)
    for j=1,binary_vec:size(1) do
        if binary_vec[j]:size(1)==#priority_list-1 then
            for i=binary_vec[j]:size(1),1,-1 do
                if binary_vec[j][i] == 1 then
                    break
                else
                    res[j] = res[j] + 1
                end   
            end 
        elseif binary_vec[j]:size(1)==(#priority_list-1)*2 then
            local a=torch.reshape(binary_vec[j],binary_vec[j]:size(1)/2,2)
            for i=a:size(1),1,-1 do
                if a[i][1] > a[i][2] then
                    break
                else
                    res[j] = res[j] + 1
                end   
            end         
        else
            error('Error! Input vector dimensions greater than 2.')
        end
           
    end

    return res
end

function getPriorityClassVector(priority_class,priority_list)
    local result = torch.FloatTensor(#priority_list-1):fill(0)
    
    for i=#priority_list-1,#priority_list+1-priority_list[priority_class],-1 do
        result[i] = 1
    end

    return result
end

function getPriorityClass(elm_classes,priority_list,per_priority_mapping)
    local current_class = "normal"
    for j=1,#elm_classes do
        if priority_list[per_priority_mapping[elm_classes[j]]] > priority_list[current_class] then
            current_class = per_priority_mapping[elm_classes[j]]   
        end              
    end
    return current_class
end

function remapInfosToPriorityClasses(infos_tbl,priority_list,per_priority_mapping)
    local out_tbl = tds.Hash()
    for k,v in pairs(priority_list) do
        out_tbl[k] = tds.Vec()        
    end
    

    for i=1,#infos_tbl do
        if #infos_tbl[i].classes > 0 then
            local priority_class = getPriorityClass(infos_tbl[i].classes,priority_list,per_priority_mapping)
            out_tbl[priority_class]:insert(infos_tbl[i]) 
        end   
    end   


    return out_tbl
end

function save_table_as_csv(path,t,balance_batch_flag)
    if balance_batch_flag == true then
        local file = io.open(path, "w")
        local counter = 0
        for key,value in pairs(t) do
            for i =1,#value do
                local classes_string = ""
                for j=1,#value[i].classes do
                    classes_string = classes_string .. "," .. value[i].classes[j]
                end
                file:write(value[i].id .. "," .. value[i].path  .. classes_string .. "\n")
            end
            counter = counter + #value    
        end
        if opt.verbose >= 1 then print("Saving imgs paths on a file (" .. path .. "): " .. counter) end
    
        file:close()
    else
        local file = io.open(path, "w")   
        for i=1, #t do
            local classes_string = ""
            for j=1,#t[i].classes do
                classes_string = classes_string .. "," .. t[i].classes[j]
            end
            file:write(t[i].path .. "," .. t[i].id  .. classes_string .. "\n")
        end
        if opt.verbose >= 1 then print("Saving imgs paths on a file (" .. path .. "): " .. #t) end
    
        file:close()
    end       
end
