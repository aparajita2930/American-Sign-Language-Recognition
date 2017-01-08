require 'torch'
require 'image'
require 'ffmpeg'
local utils = require 'utils'

local DataLoader = torch.class('DataLoader')
function DataLoader:__init(kwargs)
  self.splits = {
    train = {},
    test = {},
    val = {}
  }

  self.opt = {}

  self.splits.train.list = utils.getArgs(kwargs, 'trainList')
  self.splits.test.list = utils.getArgs(kwargs, 'testList')
  self.splits.val.list = utils.getArgs(kwargs, 'valList')
  self.opt.dumpFrames = utils.getArgs(kwargs, 'dumpFrames')
  self.opt.seqLength = utils.getArgs(kwargs, 'seqLength')
  self.opt.videoHeight = utils.getArgs(kwargs, 'videoHeight')
  self.opt.videoWidth = utils.getArgs(kwargs, 'videoWidth')
  self.opt.scaledHeight = self.opt.videoHeight
  self.opt.scaledWidth = self.opt.videoWidth
  self.opt.maxClipLength = utils.getArgs(kwargs, 'maxClipLength')
  self.opt.numChannels = utils.getArgs(kwargs, 'numChannels')
  self.opt.desiredFPS = utils.getArgs(kwargs, 'desiredFPS')
  self.opt.batchSize = utils.getArgs(kwargs, 'batchSize')
  self.opt.cuda = utils.getArgs(kwargs, 'cuda')

  local loadImagesOpt = {
    seqLength = self.opt.seqLength,
    imageType = 'jpg',
    videoHeight = self.opt.videoHeight,
    videoWidth = self.opt.videoWidth,
    scaledHeight = self.opt.scaledHeight,
    scaledWidth = self.opt.scaledWidth,
    maxClipLength = self.opt.maxClipLength,
    numChannels = self.opt.numChannels,
    desiredFPS = self.opt.desiredFPS
  }

  for split, _ in pairs(self.splits) do
    self.splits[split].index = 1
    self.splits[split].file = paths.basename(self.splits[split].list)
    self.splits[split].paths, self.splits[split].labels = loadList(self.splits[split].list)
    self.splits[split].count = #self.splits[split].paths
    self.splits[split].dumpPath = paths.concat("data", self.splits[split].file .. '_videos')
    if self.opt.dumpFrames == 1 then
      dumpVideoFrames(self.splits[split].paths, self.splits[split].dumpPath, loadImagesOpt)
    end
  end
  self.splits.train.shuffle = torch.randperm(self.splits.train.count)
  utils.print("Getting training data mean frame")
  self.splits.train.mean = getMeanTrainingImage(self.splits.train.paths, self.splits.train.dumpPath, loadImagesOpt)
end


-- Get the next batch for the split
function DataLoader:nextBatch(split)
    assert(split == 'train' or split == 'test' or split == 'val')

    local videoData = {}
    local frameLabels = {}

    while self.splits[split].index <= self.splits[split].count and #videoData < self.opt.batchSize do
        local index
        if split == 'train' then
            index = self.splits[split].shuffle[self.splits[split].index]
        else
            index = self.splits[split].index
        end
        local videoPath = self.splits[split].paths[index]
        local videoLabel = self.splits[split].labels[index]

        -- get video file 
        local videoFilename = paths.basename(videoPath)

        -- get path of dumped frames for video
        videoPath = paths.concat("data", self.splits[split].file .. '_videos', videoFilename .. '_frames')

        -- check if this video qualified to be read (extracted self.opt.seqLength or more frames)
        if paths.dirp(videoPath) then
            local framePath = paths.concat(videoPath, 'frame%d.jpg')
            local videoTensor = torch.Tensor(self.opt.seqLength, self.opt.numChannels, self.opt.scaledHeight, self.opt.scaledWidth)
            for i = 1, self.opt.seqLength do
                local frame = image.load(string.format(framePath,i), self.opt.numChannels, 'double')
                image.scale(videoTensor[i], frame) -- image.load reads in channels x height x width
                videoTensor[i] = videoTensor[i] - self.splits.train.mean
            end
            table.insert(videoData, videoTensor)
            table.insert(frameLabels, torch.Tensor(self.opt.seqLength):fill(videoLabel))
        end
        self.splits[split].index = self.splits[split].index + 1
    end

    if #videoData > 0 then
        local batch = {
            data = torch.cat(videoData, 1):type('torch.FloatTensor'),
            labels = torch.cat(frameLabels, 1):type('torch.FloatTensor'),
        }

        setmetatable(batch, 
        {__index = function(t, k) 
                return {t.data[k], t.labels[k]} 
            end}
        );

        function batch:size() 
            return self.data:size(1)
        end

        return batch
    else
        -- reset counters
        self.splits[split].index = 1
        return nil
    end
end


--Get the list of tables and the labels

function loadList(videoListPath)
    local videoPaths = {}
    local videoLabels = {}
    local file, err = io.open(videoListPath, 'rb')
    if err then
        utils.print(err)
        return
    else
        while true do
            local line = file:read()
            if line == nil then
                break
            end

            -- get tokens from line containing video path and label
            local tokens = {}
            for token in string.gmatch(line, "([^%s]+)") do
                table.insert(tokens, token)
            end
            local videoPath, videoLabel = unpack(tokens)

            table.insert(videoPaths, videoPath)
            table.insert(videoLabels, tonumber(videoLabel))
        end
    end

    return videoPaths, videoLabels
end


--Dump the frames of the videos in the folder specific to the video
function dumpVideoFrames(videoPaths, fullDumpPath, opt)
    -- make fullDumpPath
    local success, err = paths.mkdir(fullDumpPath)

    local numDumped = 0
    local numOmitted = 0

    for _, videoPath in pairs(videoPaths) do
        -- get video file name only
        local videoFilename = paths.basename(videoPath)

        local fullVideoPath = paths.concat('../', videoPath)
        --local fullVideoPath = videoPath

        if paths.filep(fullVideoPath) then
            local video = ffmpeg.Video{
            path=fullVideoPath,
            width=opt.videoWidth,
            height=opt.videoHeight,
            fps=opt.desiredFPS,
            length=opt.maxClipLength,
            silent=true
            }

            local videoTensor = video:totensor({}) -- channels x height x width
            local numFrames = videoTensor:size()[1]
            if numFrames >= opt.seqLength then
                -- segment video frames into semi equally sized segments
                -- algorithm: http://stackoverflow.com/a/7788204
                -- TODO: make the +1's random rather than front loaded
                local segmentSize = math.floor(numFrames / opt.seqLength)
                local extra = numFrames % opt.seqLength
                local normal = opt.seqLength - extra

                local frameRange = torch.range(1, numFrames)
                local segments = {}
                local startIndex = 1
                for i = 1, extra do
                    local endIndex = startIndex + segmentSize
                    table.insert(segments, torch.totable(frameRange[{ {startIndex, endIndex} }]))
                    startIndex = endIndex + 1
                end

                for i = 1, normal do
                    local endIndex = startIndex + segmentSize - 1
                    table.insert(segments, torch.totable(frameRange[{ {startIndex, endIndex} }]))
                    startIndex = endIndex + 1
                end

                -- select random frame from each segment
                local frameIndices = {}
                for i = 1, opt.seqLength do
                    local segment = segments[i]
                    table.insert(frameIndices, segment[torch.random(1, #segment)])
                end

                videoOutputDirectory = paths.concat(fullDumpPath, videoFilename .. '_frames')
                local success, err = paths.mkdir(videoOutputDirectory)
                if err then
                    utils.print(string.format(err .. " for %s",videoOutputDirectory))
                end
        
                for k, v in pairs(frameIndices) do
                    local im = videoTensor[v]
                    --im = im - torch.mean(im)
                    --im = im/torch.std(im)
                    --im = image.rgb2y(im)
                    image.save(paths.concat(videoOutputDirectory, string.format('frame%d.%s',k, 'jpg')), im)
                end
                numDumped = numDumped + 1
            else
                numOmitted = numOmitted + 1
                utils.print(string.format("Did not dump '%s', on line %d, because it did not generate enough frames." ,videoPath, numDumped + numOmitted))
            end
        end
    end
    utils.print(string.format("Dumped %d videos into %s and omitted %d videos. Total = %d." ,numDumped, fullDumpPath, numOmitted, numDumped + numOmitted))
end


-- Get mean of training images

function getMeanTrainingImage(videoPaths, fullDumpPath, opt)
    local means = {0, 0, 0}
    local numFrames = 0

    for _, videoPath in pairs(videoPaths) do
        -- get video file 
        local videoFilename = paths.basename(videoPath)

        -- get path of dumped frames for video
        videoPath = paths.concat(fullDumpPath, videoFilename .. '_frames')

        -- check if this video qualified to be read (had opt.seqLength or more frames)
        if paths.dirp(videoPath) then
            numFrames = numFrames + opt.seqLength
            local framePath = paths.concat(videoPath, 'frame%d.jpg')
            for i = 1, opt.seqLength do
                local frame = image.load(string.format(framePath,i), opt.numChannels, 'double')
                frame = image.scale(frame, string.format('%dx%d',opt.scaledWidth, opt.scaledHeight)) -- scales with string 'WxH', outputs channels x height x width
                for j = 1, opt.numChannels do
                    means[j] = means[j] + frame[j]:sum() / (opt.scaledWidth * opt.scaledHeight)
                end
            end
        end
    end

    local meanImage = torch.Tensor(opt.numChannels, opt.scaledHeight, opt.scaledWidth)
    for i = 1, opt.numChannels do
        meanImage[i]:fill(means[i] / numFrames)
    end

    return meanImage
end
