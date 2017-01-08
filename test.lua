require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'LSTM'

require 'get_data'

local utils = require 'utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Options
cmd:option('-checkpoint', '')
cmd:option('-split', 'test')
cmd:option('-cuda', 1)

local opt = cmd:parse(arg)

assert(opt.checkpoint ~= '', "Need a trained network file to load.")

-- Set up GPU
opt.dtype = 'torch.FloatTensor'
if opt.cuda == 1 then
	require 'cunn'
  opt.dtype = 'torch.CudaTensor'
end

-- Initialize model and criterion
utils.print("Initializing model")
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(opt.dtype)
local criterion = nn.ClassNLLCriterion():type(opt.dtype)

-- Initialize DataLoader to receive batch data
utils.print("Initializing DataLoader")
local loader = DataLoader(checkpoint.opt)

function test(model, split, task)
	assert(task == 'recognition')
  collectgarbage()
  utils.print(string.format("Starting %s testing on the %s split",task, split))

  local evalData = {}
  
  evalData.predictedLabels = {}
  evalData.trueLabels = {} -- true video or frame labels
  local batch = loader:nextBatch(split)

  while batch ~= nil do
    if opt.cuda == 1 then
      batch.data = batch.data:cuda()
      batch.labels = batch.labels:cuda()
    end

    local numData = batch:size() / checkpoint.opt.seqLength
    local scores = model:forward(batch.data)

    for i = 1, numData do
      local startIndex = (i - 1) * checkpoint.opt.seqLength + 1
      local endIndex = i * checkpoint.opt.seqLength
      local videoFrameScores = scores[{ {startIndex, endIndex}, {} }]
      local videoScore = torch.sum(videoFrameScores, 1) / checkpoint.opt.seqLength
      local maxScore, predictedLabel = torch.max(videoScore[1], 1)
      table.insert(evalData.predictedLabels, predictedLabel[1])
      table.insert(evalData.trueLabels, batch.labels[i])
    end

   	batch = loader:nextBatch(split)
  end

  if task == 'recognition' then
    evalData.predictedLabels = torch.Tensor(evalData.predictedLabels)
	  evalData.trueLabels = torch.Tensor(evalData.trueLabels)
    if task == 'recognition' then
      print(task)
      print("Predicted Labels")
      print(evalData.predictedLabels)
      print("True Labels")
      print(evalData.trueLabels)
    end
	  return torch.sum(torch.eq(evalData.predictedLabels, evalData.trueLabels)) / evalData.predictedLabels:size()[1]
  else
  	return evalData.loss / evalData.numBatches
  end
end

local testRecognitionAcc = test(model, 'test', 'recognition')
utils.print(string.format("Action recognition accuracy on the test set: %f",testRecognitionAcc))
