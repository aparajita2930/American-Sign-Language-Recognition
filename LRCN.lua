--LRCN

require 'nn'

require 'LSTM'

local utils = require 'utils'

-- Convenience layers
function convBatchNormTanhPool(model, inputLayers, hiddenLayers, cnnKernel, cnnStride, cnnPad, batchnorm, poolKernel, poolPad)
    model:add(nn.SpatialConvolution(inputLayers, hiddenLayers, cnnKernel, cnnKernel, cnnStride, cnnStride, cnnPad, cnnPad))
    if batchnorm == 1 then
        model:add(nn.SpatialBatchNormalization(hiddenLayers))
    end
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(poolKernel, poolKernel, poolPad, poolPad))
end

function convTanh(model, inputLayers, hiddenLayers, cnnKernel, cnnStride, cnnPad)
    model:add(nn.SpatialConvolution(inputLayers, hiddenLayers, cnnKernel, cnnKernel, cnnStride, cnnStride, cnnPad, cnnPad))
    model:add(nn.Tanh())
end

function convTanhPool(model, inputLayers, hiddenLayers, cnnKernel, cnnStride, cnnPad, poolKernel, poolPad)
    model:add(nn.SpatialConvolution(inputLayers, hiddenLayers, cnnKernel, cnnKernel, cnnStride, cnnStride, cnnPad, cnnPad))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(poolKernel, poolKernel, poolPad, poolPad))
end

--[[
  Construct an LRCN with specific parameters.
]]--
function LRCN(kwargs)
    assert(kwargs ~= nil)

    local batchnorm = utils.getArgs(kwargs, 'batchnorm')
    local dropout = utils.getArgs(kwargs, 'dropout')
    local scaledHeight = utils.getArgs(kwargs, 'videoHeight')
    local scaledWidth = utils.getArgs(kwargs, 'videoWidth')
    local seqLength = utils.getArgs(kwargs, 'seqLength')
    local numClasses = utils.getArgs(kwargs, 'numClasses')
    local numChannels = utils.getArgs(kwargs, 'numChannels')
    local lstmHidden = utils.getArgs(kwargs, 'lstmHidden')

    -- Should use about 3.5 GB VRAM. See comments AlexNet # of hidden layers.
    local cnn = {}
    cnn.stride = 1
    cnn.kernel1 = 7
    cnn.pad1 = 3
    cnn.kernel2 = 5
    cnn.pad2 = 2
    cnn.kernel3 = 3
    cnn.pad3 = 1
    cnn.numHidden1 = 64 --96 --64
    cnn.numHidden2 = 128 --256 --96
    cnn.numHidden3 = 256 --384 --128
    cnn.numHidden4 = 256 --384 --128
    cnn.numHidden5 = 512 --256 --196
    cnn.numHidden6 = 1024 --4096 --320
	cnn.numHidden7 = 1024 --2096 
	cnn.numHidden8 = 2096 --2096
    cnn.poolKernel = 2
    cnn.poolPad = 2
    cnn.reductionFactor = (1/2) ^ 3 -- three 2x2 stride 2 pool layers = (1/2)^3

    local lstm = {}
    lstm.numHidden = lstmHidden -- 256 default

    local model = nn.Sequential()
    convBatchNormTanhPool(model, numChannels, cnn.numHidden1, cnn.kernel1, cnn.stride, cnn.pad1, batchnorm, cnn.poolKernel, cnn.poolPad)
    convBatchNormTanhPool(model, cnn.numHidden1, cnn.numHidden2, cnn.kernel2, cnn.stride, cnn.pad2, batchnorm, cnn.poolKernel, cnn.poolPad)

    convTanh(model, cnn.numHidden2, cnn.numHidden3, cnn.kernel3, cnn.stride, cnn.pad3)
    convTanh(model, cnn.numHidden3, cnn.numHidden4, cnn.kernel3, cnn.stride, cnn.pad3)

    convTanhPool(model, cnn.numHidden4, cnn.numHidden5, cnn.kernel3, cnn.stride, cnn.pad3, cnn.poolKernel, cnn.poolPad)

    model:add(nn.View(cnn.numHidden5 * scaledWidth*cnn.reductionFactor * scaledHeight*cnn.reductionFactor))
    model:add(nn.Linear(cnn.numHidden5 * scaledWidth*cnn.reductionFactor * scaledHeight*cnn.reductionFactor, cnn.numHidden6))
    model:add(nn.Tanh())
	model:add(nn.Linear(cnn.numHidden6, cnn.numHidden7))
    model:add(nn.Tanh())
	--model:add(nn.Linear(cnn.numHidden7, cnn.numHidden8))
    --model:add(nn.Tanh())
    if dropout > 0 then
        model:add(nn.Dropout(dropout))
    end

    -- Reshape for LSTM; N items x T sequence length x H hidden size
    model:add(nn.View(-1, seqLength, cnn.numHidden7))

    model:add(nn.LSTM(cnn.numHidden7, lstm.numHidden))
    model:add(nn.View(-1, lstm.numHidden))
    if dropout > 0 then
        model:add(nn.Dropout(dropout))
    end
    model:add(nn.Linear(lstm.numHidden, numClasses))
    --model:add(nn.LogSoftMax())

    return model
end
