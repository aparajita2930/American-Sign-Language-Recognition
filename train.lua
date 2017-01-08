-- Train

require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'LSTM'

require 'LRCN'
require 'get_data'
require 'gnuplot'

local utils = require 'utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-trainList', '') -- necessary
cmd:option('-valList', '') -- necessary
cmd:option('-testList', '') -- necessary
cmd:option('-numClasses', '') -- necessary
cmd:option('-dumpFrames', 1) -- fresh run assumes video frames should be dumped
cmd:option('-videoHeight', '') -- necessary
cmd:option('-videoWidth', '') -- necessary
cmd:option('-maxClipLength', 72) -- used to capture max length video
cmd:option('-numChannels', 3)
cmd:option('-desiredFPS', 3) --15
cmd:option('-batchSize', 1) -- batches of videos

-- Model options
cmd:option('-batchnorm', 1)
cmd:option('-dropout', 0.5)
cmd:option('-seqLength', 38) --16
cmd:option('-lstmHidden', 256)

-- Optimization options
cmd:option('-numEpochs', 1)
cmd:option('-learningRate', 1e-6)
cmd:option('-lrDecayFactor', 0.5)

-- Output options
cmd:option('-printEvery', 1) -- Print the loss after every n epochs
cmd:option('-checkpointEvery', 1) -- Save model, print train acc
cmd:option('-checkpointName', 'checkpoints/checkpoint') -- Save model

-- Backend options
cmd:option('-cuda', 1)

local opt = cmd:parse(arg)

-- Torch cmd parses user input as strings so we need to convert number strings to numbers
for k, v in pairs(opt) do
    if tonumber(v) then
        opt[k] = tonumber(v)
    end
end

assert(opt.trainList ~= '', "No training list provided")
assert(opt.testList ~= '', "No test list provided")
assert(opt.numClasses ~= '', "Specify number of classes")
if opt.dumpFrames == 1 then
    assert(opt.videoHeight ~= '', "Video frames are to be dumped; need native height.")
    assert(opt.videoWidth ~= '', "Video frames are to be dumped; need native width.")
end

-- Set up GPU
opt.dtype = 'torch.FloatTensor'
if opt.cuda == 1 then
    require 'cunn'
    --require 'cudnn'
    opt.dtype = 'torch.CudaTensor'
end


-- Initialize DataLoader to receive batch data
utils.print("Initializing DataLoader")
local loader = DataLoader(opt)

-- Frames have been dumped, so we don't want to do so when we load this again in testing
opt.dumpFrames = 0

-- Initialize model and criterion
utils.print("Initializing LRCN")
local model = LRCN(opt):type(opt.dtype)
--if opt.cuda == 1 then
--    local cudnn = require 'cudnn'
--    cudnn.convert(model, cudnn) 
--end
local criterion = nn.ClassNLLCriterion():type(opt.dtype)
print(model)

function train(model)
    utils.print(string.format("Starting training for %d epochs",opt.numEpochs))

	local train_loss_tbl = {}
	local val_loss_tbl = {}
	local val_error_tbl = {}
	local epoch_tbl = {}

    local trainLossHistory = {}
    local valLossHistory = {}
    local valLossHistoryEpochs = {}

    local config = {learningRate = opt.learningRate}
    local params, gradParams = model:getParameters()

    for i = 1, opt.numEpochs do
        collectgarbage()

        local epochLoss = {}
        local videosProcessed = 0

        if i % 5 == 0 then
            local oldLearningRate = config.learningRate
            config = {learningRate = oldLearningRate * opt.lrDecayFactor}
        end

        local batch = loader:nextBatch('train')

        while batch ~= nil do
            if opt.cuda == 1 then
                batch.data = batch.data:cuda()
                batch.labels = batch.labels:cuda()
            end

            videosProcessed = videosProcessed + (batch:size() / opt.seqLength)

            local function feval(x)
                collectgarbage()

                if x ~= params then
                    params:copy(x)
                end

                gradParams:zero()

                local modelOut = model:forward(batch.data)
                local frameLoss = criterion:forward(modelOut, batch.labels)
                local gradOutputs = criterion:backward(modelOut, batch.labels)
                local gradModel = model:backward(batch.data, gradOutputs)

                return frameLoss, gradParams
            end

            local _, loss = optim.adam(feval, params, config)
            table.insert(epochLoss, loss[1])

            batch = loader:nextBatch('train')
        end

        local epochLoss = torch.mean(torch.Tensor(epochLoss))
        table.insert(trainLossHistory, epochLoss)

        -- Print the epoch loss
        utils.print(string.format("Epoch %d training loss: %f",i, epochLoss))
		
		--table.insert(train_error_tbl, clerr:value{k = 1})
		table.insert(train_loss_tbl, epochLoss)
		table.insert(epoch_tbl, i)
		
        -- Save a checkpoint of the model, its opt parameters, the training loss history, and the validation loss history
        if (opt.checkpointEvery > 0 and i % opt.checkpointEvery == 0) or i == opt.numEpochs then
            local valLoss = test(model, 'val', 'loss')
            utils.print(string.format("Epoch %d validation loss: %f",i, valLoss))
			--utils.print(string.format("Epoch %d validation error: %f",i, valError))
            table.insert(valLossHistory, valLoss)
            table.insert(valLossHistoryEpochs, i)
			table.insert(val_loss_tbl, valLoss)
			--table.insert(val_error_tbl, valError)
			
            local checkpoint = {
                opt = opt,
                trainLossHistory = trainLossHistory,
                valLossHistory = valLossHistory
            }

            local filename
            if i == opt.numEpochs then
                filename = string.format('%s_%s.t7',opt.checkpointName, 'final')
            else
                filename = string.format('%s_%d.t7',opt.checkpointName, i)
            end

            -- Make sure the output directory exists before we try to write it
            paths.mkdir(paths.dirname(filename))

            -- Cast model to float so it can be used on CPU
            model:float()
            checkpoint.model = model
            torch.save(filename, checkpoint)

            -- Cast model back so that it can continue to be used
            model:type(opt.dtype)
            params, gradParams = model:getParameters()
            utils.print(string.format("Saved checkpoint model and opt at %s",filename))
            collectgarbage()
        end
    end

    utils.print("Finished training")

	--win_w1 = image.display{image=model.weight:reshape(10,28,28),nrow=5,win=win_w1,legend='Network weights',padding=2}
	
	filters = model:get(1).weight
	filters = image.toDisplayTensor(filters)
	image.save("weights.png",filters)

	utils.print("Print plots")
	epoch_array = torch.Tensor(epoch_tbl)
	--print(epoch_array:size())
	train_loss = torch.Tensor(train_loss_tbl)
	--print(train_loss:size())
	val_loss = torch.Tensor(val_loss_tbl)
	--val_error = torch.Tensor(val_error_tbl)
	
	gnuplot.pngfigure("train_loss_plot.png")
	gnuplot.plot({'Train_Loss', epoch_array, train_loss, '-'})
	gnuplot.xlabel('Epochs ----->')
	gnuplot.ylabel('Average Loss ----->')
	gnuplot.plotflush()
	
	gnuplot.pngfigure("validation_loss_plot.png")
	gnuplot.plot({'Validation_Loss', epoch_array, val_loss, '-'})
	gnuplot.xlabel('Epochs ----->')
	gnuplot.ylabel('Average Loss ----->')
	gnuplot.plotflush()
	
	--gnuplot.pngfigure("validation_error_plot.png")
	--gnuplot.plot({'Validation_Error', epoch_array, val_error, '-'})
	--gnuplot.xlabel('Epochs ----->')
	--gnuplot.ylabel('Average Error ----->')
	--gnuplot.plotflush()
end


function test(model, split, task)
    assert(task == 'loss')
    collectgarbage()
    utils.print(string.format("Starting %s testing on the %s split",task, split))

    local evalData = {}
    evalData.loss = 0 -- sum of losses
    evalData.numBatches = 0 -- total number of frames

    local batch = loader:nextBatch(split)

    while batch ~= nil do
        if opt.cuda == 1 then
            batch.data = batch.data:cuda()
            batch.labels = batch.labels:cuda()
        end

        local numData = batch:size()
        local scores = model:forward(batch.data)

        evalData.loss = evalData.loss + criterion:forward(scores, batch.labels)
        evalData.numBatches = evalData.numBatches + 1

        batch = loader:nextBatch(split)
    end

    return evalData.loss / evalData.numBatches
    
end

train(model)
