--------------------------------------------------------------------------------
-- This script is the main driver for training a x-rays chest images classifier using
-- deep learning models. 
-- 
-- 
-- Details:
-- +  It assumes that data are images stored in tiff files, the paths for all this
--    files are written in a csv input file that has the following schema:
--    ID,ACC_NUM,age(in days),exam date,sex,path,view,[class1,...,classn]
--    
--    Look in /montana-storage/shared/data/chest_xrays/gstt/data_info/
      
-- +  It allows multithread data loading and multi GPU computing.
-- 
-- Authors: Mauro Annarumma
--------------------------------------------------------------------------------

require 'torch'

--------------------------------------------------------------------------------
--  Global options
--------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()

cmd:text('SVHN Loss Function')
cmd:text()
cmd:text('Options:')

-- global:
cmd:option('-seed',     1, 		'fixed input seed for repeatable experiments')
cmd:option('-gpuid',    1,      'ID of the GPU to be used as master')
cmd:option('-numGPU',   1,      'number of GPUs')
cmd:option('-verbose',  0,      'level of verbosity, useful for debugging')
cmd:option('-remove_exams_overlapping_classes', 0,  '0 | 1: if equal to 1 during the data loading we exclude all the exams with more than one label')
cmd:option('-use_only_clean_images',  0,      '0|1: use/not use the set with only clean images (clean means only from non-mobile xrays scanners)')
cmd:option('-ageLimit',         16, "Load all the patiants with age > of this value (in year)")
cmd:option('-covariate_variables',      "0x0x0", "0 | 1 : use not use [age, PA/AP, sex] as covariate variable")
cmd:option('-progressive_train',        0, "0 | 1 : during the progressing train we start training with only the biggest classes and progressivly we also use smaller")
cmd:option('-classify_per_priority',    0,  '0 | 1: use standard classes | use priority classes ')
cmd:option('-ordinal_regression',       0,'0 | 1: when use priority classes takes in account the order between them')
cmd:option('-tensor_precision',         'single', 'single | half precision')
cmd:option('-fixed_size_input_mode',    'crop', 'crop | pad | pad_or_crop: pad the image, crop the image or randomly do both (the last one is the best for data augmentation)')
cmd:option('-multi_task', false, 'enable | disable multi-task learning')
--cmd:option('-delete_object_class',  false,  'if this flag is equal to true we do not use the object class')

-- paths
cmd:option('-path_root', '.', 		'root path')
cmd:option('-path_temp', './temp/', 'directory for the temp files')
cmd:option('-path_save', 'results', 'directory for the output of the network')
cmd:option('-path_test_results', 'results/test_img_error/', 'subdirectory where to save/log infos about misclassification during the test')
cmd:option('-path_preprocess_param', "results/preprocess_param.bin",     'path preprocess param')
cmd:option('-input_csv_dir', '/montana-storage/shared/data/chest_xrays/gstt/data_info/split_v12_27_Jun_17_mauro/', 'path of the csv file with all the dataset infos')
cmd:option('-text_BBs_file',    "/montana-storage/shared/data/chest_xrays/gstt/data_info/text_bounding_boxes_xrays_images.idl","idl file with the predicted text BBs for all the images")
cmd:option('-path_CH_embeddings', '', 'file path of the Clinical History, if empty we do not use this information during the training')

-- model:
cmd:option('-model', 'inception_v3', 'type of model to construct: VGG | resnet_18 | resnet_50 | resnet_152 | inception_v3 | inception_v4 | bigInputNet')
cmd:option('-input_size',           "1x299x299", 'input size in format c x W x H')
cmd:option('-downloadImage_size',   "3x448x448", 'input size in format c x W x H')
cmd:option('-classifier_type',  'multi', "single|multi|RNN. single: one classifier for all the classes. multi: one classifier for each class")
-- RNN options:
cmd:option('-LSTM_hidden_layers', 512, "number of LSTM hidden units")
-- text infos:
cmd:option('-text_embedding_net', "LSTM", "LSTM | sum_linear | sum: feed the non-fixed length sequence to an LSTM or just sum the vectors")

-- loss:
cmd:option('-loss', 	'ce', 	'type of loss function to minimize: ce | nll | mse | smc | bce')
cmd:option('-weighted_criterion',   0,"0 | 1 : not use / use the weights for the criterion")

-- pretraining:
cmd:option('-preTrainedCNN',    "","pretrained CNN for feature extraction")
cmd:option('-preTrainedCHLSTM',    "","pretrained LSTM for CH sequence features processing")
cmd:option('-preTrainingEpochs',    0,"number of epochs for doing pretraining (only for RRN classifier_type)")

-- training:
cmd:option('-optimization',     'adadelta', 	'optimization method: SGD | ASGD | CG | LBFGS | adagrad | adadelta')
cmd:option('-learningRate',     0.003, 	'learning rate at t=0')
cmd:option('-maxEpoch',         100, 	'max number of epochs')
cmd:option('-epoch_size',       30000,   'size of each epoch')
cmd:option('-batchSize',        128,	'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay',      1e-4, 		'weight decay (SGD only)')
cmd:option('-momentum',         0.9, 		'momentum (SGD only)')
cmd:option('-t0',               1, 		'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter',          10,		'maximum nb of iterations for CG and LBFGS')
cmd:option('-dataLoadingThreads',   4, 'if equal to zero start mono-thread version of data loading')
cmd:option('-classes', '',      'class in format class1,class2,..,classn')
cmd:option('-restart_training', '', "if empty string: start train from scratch, else use the model in the specified path")
cmd:option('-predict_age',      false , "true: add a regressor for the age prediction")
cmd:option('-augment_train_data',   1,"0 | 1 : not use / use data augmentation for training")
cmd:option('-balance_classes', 0, "0 | 1 : not use / use mini-batch class balancing")
cmd:option('-doPreprocess',     true, 'true: do the preprocessing')

-- testing
cmd:option('-test_epoch_size',  10000,   'size of each epoch')
cmd:option('-testMode',         '', 'if empty string: do not run in test mode, else use the model in the specified path')
cmd:option('-test_on_golden_labels',         0, 'if equal to 1 we use the annotations made by clinicians')
cmd:option('-train_opt',        '', 'used only if test mode is ON, contains the path to the dump of the opt table used during training')
cmd:option('-fixed_size_input',       1,  '0 | 1: crop/pad or do not crop/pad image during testing')
cmd:option('-test_abnormalvsnormal',       0,  '0 | 1: classify one vs all | classify normal vs classX ')
cmd:option('-test_per_priority',    0,  '0 | 1: use priority classes when you test a model that was trained on standard classes')

-- extract feature
cmd:option('-feature_extraction',  false,   'run the current mode (with path in opt.testMode or opt.preTrainedCHLSTM and opt.preTrainedCHLSTM) and extract features')

cmd:text()

-- Acquiring options
opt = cmd:parse(arg or {})

require 'preprocess'
require 'utils'
tds = require 'tds'


-- convert the strctured input string to a table
opt.input_size = fromStringToDimensionsTable(opt.input_size, 'x') 
opt.downloadImage_size = fromStringToDimensionsTable(opt.downloadImage_size, 'x')
opt.covariate_variables = fromStringToDimensionsTable(opt.covariate_variables, 'x')
-- assert input options make sense
assert(opt.downloadImage_size[1]>=opt.input_size[1] and opt.downloadImage_size[2]>=opt.input_size[2] and opt.downloadImage_size[3]>=opt.input_size[3], 'check the input size, these current values are wrong')
assert((opt.progressive_train == 1 and (opt.classifier_type  == "multi" or opt.classifier_type  == "RNN")) or opt.progressive_train == 0,'Progressive train implemented only for multi and RNN classifier type')

if opt.classes == "" then
    dofile "all_classes.lua"
    opt.classes = classes   
else
    opt.classes = split_string(opt.classes, ',')
end

if opt.verbose >= 0 then print '\n==> Input options correctly acquired\n' end

-----------------------
-- Set global variables
-----------------------
thresholds = {0.1,0.2,0.3,0.4,0.5}

-- if we use data augmentation we must not flip those images
to_not_flip_classes = {'dextrocardia','right_upper_lobe_collapse','right_middle_lobe_collapse','right_lower_lobe_collapse','left_upper_lobe_collapse','left_lower_lobe_collapse' }

-- dimension of the input vector with the covariate variables
covariate_variables_dim = getCovariateVariablesDim(opt.covariate_variables)
CH_emb_size, max_CH_len = 4096, 10
  
--------------------------------------------------------------------------------
--  Read infos about images
--------------------------------------------------------------------------------

local IDL_infos = nil
if opt.text_BBs_file ~= "" then
    -- read predicted bounding boxes for text 
    IDL_infos = readIDL(opt.text_BBs_file)
end

-- define which file to read for train, validation and test
local train_filename,test_filename,val_filename,gl_filename = "xrays_dataset_train.csv","xrays_dataset_test.csv","xrays_dataset_val.csv","xrays_dataset_gl.csv"
if opt.use_only_clean_images==1 then
    train_filename,test_filename,val_filename,gl_filename = "xrays_dataset_clean_train.csv","xrays_dataset_clean_test.csv","xrays_dataset_clean_val.csv","xrays_dataset_clean_gl.csv"
end

     

if opt.testMode == '' and (not opt.feature_extraction) then

    if opt.classify_per_priority == 1 or opt.test_per_priority == 1 or opt.multi_task then
        per_priority_mapping, priority_list = definePriorityMapping(classes)
    end 

    -- custom result directory
    opt.path_save = opt.path_save .. "/" .. os.date('%y%m%d%H%M%S')

    --create temp dir if it does not exist
    if not directory_exists(opt.path_temp) then os.execute('mkdir -p ' .. opt.path_temp) end  
    if not directory_exists(opt.path_save) then os.execute('mkdir -p ' .. opt.path_save) end  

    train_infos, _ = read_csv(opt.input_csv_dir .. train_filename)
    val_infos, test_data_infos = read_csv(opt.input_csv_dir .. val_filename)
    
    if IDL_infos ~= nil then
        train_infos = addTextBoxInfos(train_infos,IDL_infos)
        val_infos = addTextBoxInfos(val_infos,IDL_infos)
    end   
    
    if opt.classify_per_priority == 1 then
        classes = {"normal","non_urgent_finding","urgent","critical"}
        train_infos = remapInfosToPriorityClasses(train_infos,priority_list,per_priority_mapping) 
        if opt.ordinal_regression == 1 then
            assert(opt.classifier_type == "multi","when you train a classifier on the priority classes, with ordinal regression, you must use #classes-1 binary classifiers")
            
            local filt_classes = table.find_and_remove(classes,"normal") 
            weights_criterion = computeMultiClassifiersWeights(classes,filt_classes)        
            opt.classes = classes
            classes = filt_classes        
        else
            assert(opt.classifier_type == "single","when you train a classifier on the priority classes you must use a single classifier")        
            weights_criterion = computeSingleClassifierWeights(classes,train_infos)
        end
    else        
        
        train_infos = classBasedFiltering(train_infos,opt.classes)
        val_infos = classBasedFiltering(val_infos,opt.classes)
                 
        if opt.classifier_type == "RNN" then 
            if opt.progressive_train == 1 then
                original_train_infos = train_infos 
                classes_sets = getClassesSets(train_infos,opt.classes)
                local all_classes = table.copy(classes_sets[1])
                table.insert(all_classes,'normal') 
                train_infos = classBasedFiltering(original_train_infos,all_classes)
                original_val_infos = val_infos
                val_infos = classBasedFiltering(original_val_infos,all_classes)  
                opt.classes = tableWithAllClassesInTheRightOrder(classes_sets)
                classes,num_occurrency  = orderClassesByNumberOfOccurrency(train_infos,classes_sets[1]) 
                opt.ordered_classes,_ = orderClassesByNumberOfOccurrency(original_train_infos,opt.classes)  
                classes = tds.Vec(classes)
            else
                classes,num_occurrency  = orderClassesByNumberOfOccurrency(train_infos,opt.classes) 
                opt.ordered_classes = classes             
            end        
               
            weights_criterion = computesRNNWeights(classes,num_occurrency,#train_infos)                   
        else
            if opt.balance_classes == 1 then
                if opt.progressive_train == 1 then
                    nonSplittedTrainInfo = train_infos
                    classes_sets = getClassesSets(train_infos,opt.classes) 
                    local all_classes = table.copy(classes_sets[1])
                    table.insert(all_classes,'normal') 
                    train_infos,classes = splitPerClassWithOverlap(classBasedFiltering(nonSplittedTrainInfo,all_classes),all_classes,false)
                    original_val_infos = val_infos
                    val_infos = classBasedFiltering(original_val_infos,all_classes)
                    opt.classes = tableWithAllClassesInTheRightOrder(classes_sets)
                else
                    train_infos,classes = splitPerClassWithOverlap(train_infos,opt.classes,true)
                end 
            else
                assert(opt.progressive_train == 0,"progressive train not implemented yet for unbalanced data loading")
                classes,num_occurrency  = orderClassesByNumberOfOccurrency(train_infos,opt.classes,false)
            end
            
            if #opt.classes == 1 then table.insert(opt.classes,'normal')  end             
            if opt.classifier_type == "multi" then 
                local filt_classes = table.find_and_remove(opt.classes,"normal")
                if #filt_classes == 1 then
                    weights_criterion = computeWeightsMonoClass(train_infos,filt_classes[1])
                else
                    --TODO: fix weights_criterion initialization when opt.progressive_train == 1
                    weights_criterion = computeMultiClassifiersWeights(classes,filt_classes) 
                    if opt.progressive_train == 1 then  classes = classes_sets[1]
                    else  classes = filt_classes end
                end
            elseif opt.classifier_type == "single" then
                weights_criterion = computeSingleClassifierWeights(classes,train_infos)
            end 
                          
        end
    end

    
    table.save(opt, opt.path_save .. '/options.txt')
else
    os.execute('mkdir -p ' .. opt.path_test_results) 
    
    local or_flag = table.load(opt.train_opt).ordinal_regression
    if or_flag ~= nil then  opt.ordinal_regression = or_flag  end
    
    opt.classify_per_priority = table.load(opt.train_opt).classify_per_priority
    opt.classifier_type = table.load(opt.train_opt).classifier_type
    opt.input_size = table.load(opt.train_opt).input_size
    opt.downloadImage_size = table.load(opt.train_opt).downloadImage_size
    opt.text_embedding_net = table.load(opt.train_opt).text_embedding_net
    opt.multi_task = table.load(opt.train_opt).multi_task
    
    if opt.classify_per_priority == 1 or opt.test_per_priority == 1 or  opt.multi_task  then
        per_priority_mapping, priority_list = definePriorityMapping(classes)
    end 
    
    if (opt.test_on_golden_labels == 0) then
        test_infos, test_data_infos = read_csv(opt.input_csv_dir .. test_filename)   
    elseif (opt.test_on_golden_labels == 1) then
        test_infos, test_data_infos = read_csv(opt.input_csv_dir .. "consensus_golden_labels.csv")
    elseif (opt.test_on_golden_labels == 2) then
        test_infos, test_data_infos = read_csv(opt.input_csv_dir .. "Sam_golden_labels.csv")
    elseif (opt.test_on_golden_labels == 3) then
        test_infos, test_data_infos = read_csv(opt.input_csv_dir .. "xrays_dataset_batch3.csv")
    elseif (opt.test_on_golden_labels == 4) then
        test_infos, test_data_infos = read_csv(opt.input_csv_dir .. "xrays_dataset_val.csv") 
    elseif (opt.test_on_golden_labels == 5) then
        test_infos, test_data_infos = read_csv(opt.input_csv_dir .. "xrays_dataset_train.csv")  
    elseif (opt.test_on_golden_labels == 6) then
        test_infos, test_data_infos = read_csv(opt.input_csv_dir .. "CXR8.csv")                                 
    end 
    if IDL_infos ~= nil then
        test_infos = addTextBoxInfos(test_infos,IDL_infos)
    end
      
    if opt.classify_per_priority == 1 then
        classes = {"normal","non_urgent_finding","urgent","critical"}
    else
        test_infos = classBasedFiltering(test_infos,opt.classes)   
    end
    opt.covariate_variables = table.load(opt.train_opt).covariate_variables
    if opt.covariate_variables ~= nil then 
        covariate_variables_dim = getCovariateVariablesDim(opt.covariate_variables) 
    else
        covariate_variables_dim = 0 
    end
    
    if opt.classifier_type == "single" then
--        classes = table.load(opt.train_opt).classes
    elseif opt.classifier_type == "multi" then
        opt.classes = table.load(opt.train_opt).classes
        classes = table.find_and_remove(opt.classes,"normal") 
        if opt.testMode ~= '' and file_exists(getPath(opt.testMode).. "best_F1Score_thresholds.t7") then
            best_thresholds = torch.load(getPath(opt.testMode).. "best_F1Score_thresholds.t7")
        else
            best_thresholds = tds.Hash()
            for i=1,#classes do best_thresholds[classes[i]] = 0.5 end
        end
    elseif opt.classifier_type == "RNN" then
        classes = table.load(opt.train_opt).classes
        opt.classes = classes  
    end 
    table.save(opt, opt.path_test_results .. 'options.txt') 
end

opt = tds.Hash(opt)

--------------------------------------------------------------------------------
--  Set Cuda, number of threads
--------------------------------------------------------------------------------


torch.setdefaulttensortype('torch.FloatTensor')
require 'cutorch' 
require 'cunn'

torch.setnumthreads(1)


--------------------------------------------------------------------------------
-- Experiments: Preprocessing
--------------------------------------------------------------------------------
local tm = torch.Timer()


-- Load (or calculate) the preprocess parameteres
params = {}


if opt.testMode == '' then
    if not file_exists(opt.path_preprocess_param) then

      if opt.verbose >= 0 then
        print('==> Calculating preprocessing parameters')
        print ' '
      end
      if opt.classifier_type == "RNN" then
        params['mean'], params['var'] = calculate_mean_std(train_infos,"single_table")
      else
        params['mean'], params['var'] = calculate_mean_std(train_infos,"multi_table")
      end
      
      if opt.path_CH_embeddings ~= "" and (params['text_emb_mean'] == nil or params['text_emb_var'] == nil) then
            params['text_emb_mean'],params['text_emb_var'] = calculate_mean_std_text(train_infos)
      end
      torch.save(opt.path_save .. '/preprocess_param.bin', params)

    else
      if opt.verbose >= 0 then
        print('==> Loading preprocessing parameters')
        print ' '
      end
      
      params = torch.load(opt.path_preprocess_param)
      if opt.path_CH_embeddings ~= "" and (params['text_emb_mean'] == nil or params['text_emb_var'] == nil) then
            params['text_emb_mean'],params['text_emb_var'] = calculate_mean_std_text(train_infos)
      end      
      torch.save(opt.path_save .. '/preprocess_param.bin', params)
    end
else
    assert(file_exists(opt.path_preprocess_param),"Error, " .. opt.path_preprocess_param .. " does not exist")
    params = torch.load(opt.path_preprocess_param)
end 



if opt.feature_extraction then

    dofile 'extract_features.lua' 
    if opt.testMode == '' then
        require '2_model.lua'
        model = createModel()
    else
        if opt.path_CH_embeddings ~= "" and opt.text_embedding_net == 'LSTM' then  require 'rnn' end  
        if opt.fixed_size_input == 0 then
            model = loadDataParallel(opt.testMode,1)
            model = makeFullyConvolutional(model)
            if opt.numGPU > 1 then model = makeDataParallel(model,opt.numGPU) end            
        else
            model = loadDataParallel(opt.testMode,opt.numGPU)
        end    
    end
    extractFeaturesFromData(test_infos)
else

    if opt.testMode == '' then
        if opt.classifier_type == "RNN" then
            require "CNN_plus_LSTM"
            model = createModel_LSTM(opt)       
            dofile '3_loss.lua'
            if opt.balance_classes == 1 then
                local all_classes = table.copy(classes)
                table.insert(all_classes,'normal')
                train_infos = splitPerClass(train_infos,all_classes,true)
            end      
            dofile 'train_RNN.lua'  
        else
            require '2_model.lua'
            model = createModel()
    
            dofile '3_loss.lua'
            dofile 'train_multithread.lua'
            dofile 'test_multithread.lua' 
        end  
    
        local ep = 1
        if opt.restart_training ~= '' then
           model = loadDataParallel(opt.restart_training.."model.net",opt.numGPU) --"best_F1score_model.net",opt.numGPU)
           optimState = torch.load(opt.restart_training.."optimState.t7")
        end    
        
        local avgF1Score_train_tbl,avgF1Score_val_tbl = {}, {}
        local classes_sets_index = 1
        while ep < opt.maxEpoch do
            print("==========> Epoch " .. ep)
            if opt.progressive_train == 1 and isTimeToStopTraining(avgF1Score_train_tbl,avgF1Score_val_tbl) and classes_sets_index < #classes_sets then
                opt.epoch_size = opt.epoch_size - (opt.epoch_size / 5)
                best_F1score_so_far = 0         
                avgF1Score_train_tbl,avgF1Score_val_tbl = {}, {}
                classes_sets_index = classes_sets_index + 1
                --model = loadDataParallel(paths.concat(opt.path_save, 'best_F1score_model.net'),opt.numGPU)
                if opt.classifier_type == "multi" or opt.multi_task then
                    model = addClassifiers(model,#classes_sets[classes_sets_index])
                    criterions = addCriterions(criterions,#classes_sets[classes_sets_index])
                else
                    model = updateClassifier(model,#classes_sets[classes_sets_index])
                end
 
                optimState, optimMethod = initOptimState()

                for i=1,#classes_sets[classes_sets_index] do
                    classes:insert(classes_sets[classes_sets_index][i])
                end
                local all_classes = table.copy(classes)
                table.insert(all_classes,'normal')
                if opt.classifier_type == "multi" or opt.multi_task then
                    train_infos,_ = splitPerClassWithOverlap(classBasedFiltering(nonSplittedTrainInfo,all_classes),all_classes,false) 
                elseif opt.classifier_type == "RNN" then           
                    train_infos = classBasedFiltering(original_train_infos,all_classes)
                    classes,_  = orderClassesByNumberOfOccurrency(train_infos,classes) 
                    classes = tds.Vec(classes)
                    print(classes)
                    if opt.balance_classes == 1 then
                        train_infos = splitPerClass(train_infos,all_classes,true)
                    end                       
                end
                val_infos = classBasedFiltering(original_val_infos,all_classes) 
            elseif #opt.classes <=2 and isTimeToStopTraining(avgF1Score_train_tbl,avgF1Score_val_tbl) and ep >= (opt.maxEpoch/10) then 
                break        
            end
            table.insert(avgF1Score_train_tbl,train(train_infos))
            table.insert(avgF1Score_val_tbl,test(val_infos))
            ep = ep + 1 
        end
        print(avgF1Score_train_tbl)
        print(avgF1Score_val_tbl)
    else
        require "model_utils"
        require "cudnn"
        
        if opt.classifier_type == "RNN" then
            require "rnn"
            model = loadDataParallel(opt.testMode,opt.numGPU)
            classes = table.load(opt.train_opt).ordered_classes
            dofile 'test_RNN.lua'
        else
            if opt.path_CH_embeddings ~= "" and opt.text_embedding_net == 'LSTM' then  require 'rnn' end  
            if opt.fixed_size_input == 0 then
                model = loadDataParallel(opt.testMode,1)
                model = makeFullyConvolutional(model)
                if opt.numGPU > 1 then model = makeDataParallel(model,opt.numGPU) end            
            else
                model = loadDataParallel(opt.testMode,opt.numGPU)
            end
            dofile 'test_multithread.lua' 
        end
        test(test_infos)
    end
end
print('==> time took: ' .. tm:time().real)
print('==> script terminated.')

