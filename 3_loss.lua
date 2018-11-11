----------------------------------------------------------------------
-- This script defines loss functions
--   
-- Loss functions defined:
--   +   negative-log likelihood, using log-normalized output units (SoftMax)
--   +   mean-square error
--   +   margin loss (SVM-like)
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions


----------------------------------------------------------------------
-- Loss function definitions
----------------------------------------------------------------------

local function defineListOfCriterions(classes,criterion,weights_criterion)
    local criterions = {}
    for i =1,#classes do
        if weights_criterion and opt.weighted_criterion == 1  then
            criterions[i] = toCuda(criterion(weights_criterion[i]))
        else
            criterions[i] = toCuda(criterion())
        end
    end
    return criterions
end

if opt.verbose >= 0 then
  print '==> define loss'
  print ' '
end


if opt.loss == 'nll' then

    criterion = nn.ClassNLLCriterion

elseif opt.loss == "ce" then

    criterion = nn.CrossEntropyCriterion    
      
elseif opt.loss == "bce" then 

    criterion = nn.BCECriterion
    
elseif opt.loss == "smc" then 

    criterion = nn.SoftMarginCriterion
    
else
    error('unknown -loss')
end

----------------------------------------------------------------------
if opt.verbose >= 0 then
    print '==> here is the loss function:'
    print(criterion())
end

if opt.multi_task then
    local priorities = {"non_urgent_finding","urgent","critical"}

    criterions = {
        priority = defineListOfCriterions(priorities,criterion,nil),
        labels = defineListOfCriterions(classes,criterion,weights_criterion),
        normal = toCuda(criterion())
    }
    if opt.predict_age then
       criterions["age"] = toCuda(nn.MSECriterion()) 
    end   
else
    if opt.classifier_type == "multi" then
        criterions = defineListOfCriterions(classes,criterion,weights_criterion)

    elseif opt.classifier_type == "single" then
        if weights_criterion and opt.weighted_criterion == 1 then
            criterion = toCuda(criterion(weights_criterion))
        else
            criterion = toCuda(criterion())        
        end
    elseif opt.classifier_type == "RNN" then 
    
        if weights_criterion and opt.weighted_criterion == 1  then
            criterion = toCuda(nn.SequencerCriterion(criterion(weights_criterion)))
        else
            criterion = toCuda(nn.SequencerCriterion(criterion()))      
        end  
    
    end
    if opt.predict_age then
       age_criterion = toCuda(nn.MSECriterion()) 
    end     
end


----------------------------------------------------------------------
-- Configuring Optimizer
----------------------------------------------------------------------
require 'optim'   -- an optimization package, for online and batch methods

if opt.verbose >= 0 then
  print '==> configuring optimizer...'
  print ' '
end

function initOptimState(old_optimState)
    local optimState,optimMethod
    if opt.optimization == 'CG' then
       optimState = {
          maxIter = opt.maxIter
       }
       optimMethod = optim.cg
    
    elseif opt.optimization == 'LBFGS' then
       optimState = {
          learningRate = opt.learningRate,
          maxIter = opt.maxIter,
          nCorrection = 10
       }
       optimMethod = optim.lbfgs
    
    elseif opt.optimization == 'SGD' then
        if old_optimState ~= nil then      
--            local new_dfdx = torch.Tensor(getNumberParameters(model)):type(old_optimState.dfdx:type()):zero()
--            new_dfdx[{{1,old_optimState.dfdx:size(1)}}]:copy(old_optimState.dfdx)
--            old_optimState.dfdx = new_dfdx
            old_optimState.dfdx:resize(getNumberParameters(model))
            optimState = old_optimState
            optimState.learningRate = opt.learningRate
        else
            optimState = {
              learningRate = opt.learningRate,
              weightDecay = opt.weightDecay,
              momentum = opt.momentum,
              learningRateDecay = 0.0,
              nesterov = true,
              dampening = 0.0      
            }
        end
        optimMethod = optim.sgd
    
    elseif opt.optimization == 'ASGD' then
       optimState = {
          eta0 = opt.learningRate,
          t0 = trsize * opt.t0
       }
       optimMethod = optim.asgd
    elseif opt.optimization == 'adagrad' then
       optimState = {
          learningRate = opt.learningRate,
          weightDecay = opt.weightDecay,
          learningRateDecay = 1e-7
       }
       optimMethod = optim.adagrad
    elseif opt.optimization == 'adadelta' then
        optimState = {
            rho = 0.3,
            eps = 1e-6
        }  
       optimMethod = optim.adadelta
    elseif opt.optimization == 'adam' then
       optimState = {
          learningRate = opt.learningRate,
       }
       optimMethod = optim.adagrad
       
    elseif opt.optimization == 'RMSprop' then
       optimState = {
          learningRate = opt.learningRate,
          weightDecay = 0.9--opt.weightDecay,
       }
       optimMethod = optim.rmsprop
    else
       error('unknown optimization method')
    end
    return optimState,optimMethod
end

optimState,optimMethod = initOptimState(optimState)

function addCriterions(curr_criterions,num_criterions)
                
    for i = 1,num_criterions do
        if opt.multi_task then
            table.insert(curr_criterions['labels'],toCuda(criterion())) 
        else
            table.insert(curr_criterions,toCuda(criterion()))   
        end
    end
    
    return curr_criterions
end

if opt.verbose >= 2 then print('-- Optimizer configured') end
