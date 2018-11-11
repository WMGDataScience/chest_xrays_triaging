require 'torch'
require 'xlua'
require 'data_loading_utils'


function calculate_mean_std(imgs_infos,single_multi)

    local num_elements = 10000
    local imgs_batch = torch.FloatTensor(num_elements,opt.input_size[1],opt.input_size[3],opt.input_size[2])
    local index=1
    
    if single_multi == "single_table" then
        for i=1,#imgs_infos do
            xlua.progress(index, num_elements)
            if index <= num_elements then
                imgs_batch[index] = getImage(imgs_infos[i])
                index = index + 1
            else
                break
            end    
        end   
    else

        local elements_per_class = math.ceil(num_elements/#imgs_infos)      

        for key,value in pairs(imgs_infos) do
            for i=1,math.min(elements_per_class,#value) do
                xlua.progress(index, num_elements)
                if index <= num_elements then
                    imgs_batch[index] = getImage(value[i])
                    index = index + 1
                else
                    break
                end    
            end
        end
    end
    
    local std = {}
    local mean = {}

    for i=1,imgs_batch:size(2) do
        std[i] = imgs_batch[{{},i,{},{}}]:std()
        mean[i] = imgs_batch[{{},i,{},{}}]:mean()
    end 
    
    return mean,std 
end

function calculate_mean_std_text(imgs_infos)

    local num_elements = 30000
    local text_emb_batch = torch.FloatTensor(num_elements,CH_emb_size)
    local index=1
    
    local elements_per_class = math.ceil(num_elements/(#imgs_infos*2)) 
    for key,value in pairs(imgs_infos) do    
        for i=1,math.min(elements_per_class,#value) do 
            xlua.progress(index, num_elements)        
            local CH_file_path = opt.path_CH_embeddings .. value[i].acc_numb .. ".hdf5"
            if file_exists(CH_file_path) then                
                local myFile = hdf5.open(CH_file_path, 'r')
                local ok, data = pcall(myFile.all,myFile)
                myFile:close()
                if ok and data['CH_embedding']:dim() == 2 then
                    for j=1,data['CH_embedding']:size(1) do
                        if index <= num_elements then
                            text_emb_batch[index] = data['CH_embedding'][j]
                            index = index + 1
                        else
                            break
                        end                     
                    end
                    if index > num_elements then
                        break
                    end
                end
            end   
        end
        if index > num_elements then
            break
        end           
    end
    
    local std = text_emb_batch[{{1,index-1}}]:std()
    local mean = text_emb_batch[{{1,index-1}}]:mean()
    
    return mean,std 
end

function preprocess(data,mean,std)
    if type(data) == 'table' then
        for j=1,#data do
            for i=1,data[j]:size(1) do
                data[j][{i,{},{}}]:add(-mean[i]):div(std[i])
            end        
        end
    else
    	for i=1,data:size(2) do
    		data[{{},i,{},{}}]:add(-mean[i]):div(std[i])
    	end	
    end
end

function preprocess_text(data,mean,std)
    data:add(-mean):div(std)
end
