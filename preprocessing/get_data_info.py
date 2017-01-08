import time
import numpy as np
import threading

from get_video_info import convert_video

def get_labels(infile):
    labels = []
    with open(infile) as f:
        lines = f.readlines()
    for line in lines:
        #print line
        tag = line.split(' ')
        label = tag[1].split('\n')[0]
        labels.append(label)
    labels = labels[1:]
    return labels

#print get_labels('data_temp/data_label.txt')

class VideoGenerator(object):
    def __init__(self, videos, invideos_path, invideos_extension, length, dimension=None):
        self.videos = videos
        self.num_videos = len(videos)
        self.flow_generator = self._flow_index(self.num_videos)
        self.lock = threading.Lock()
        self.invideos_path = invideos_path
        self.invideos_extension = invideos_extension
        self.length = length
        self.dimension = dimension

    def _flow_index(self, num_videos):
        i = 0
        while i < num_videos:
            i += 1
            yield i - 1

    def next(self):
        #video_name = []
        with self.lock:
            idx = next(self.flow_generator)
        #with open(self.labels_file) as f:
        #    lines = f.readlines()
        #for line in lines[1:]:
        #    video_name.append(line.split(' ')[0])
        t1 = time.time()
        video_name = self.videos[idx]
        video_loc = self.invideos_path + '/' + video_name + '.' + self.invideos_extension
        video_tensor = convert_video(video_loc, start_frame=0, resize=self.dimension)
        #print video_tensor.shape
        if video_tensor is not None:
            video_tensor = video_tensor.transpose(1, 0, 2, 3)
            num_frames = video_tensor.shape[0]
            num_clips = num_frames // self.length
            #print num_frames
            #print num_clips
            video_tensor = video_tensor[:num_clips*self.length,:,:,:]
            #print video_tensor.shape
            video_tensor = video_tensor.reshape((num_clips, self.length, 3,)+(self.dimension))
            #print video_tensor.shape
            video_tensor = video_tensor.transpose(0, 2, 1, 3, 4)

        t2 = time.time()
        print('Time to fetch {} video: {:.2f} seconds'.format(video_name, t2-t1))
        #print video_tensor.shape
        #print video_name.split('_')[0]
        return video_name, video_tensor

    def __next__(self):
        self.next()

#generator = VideoGenerator(['1_1','1_2'],'../data_temp','mov',16,(112,112))
#print next(generator)
#for i in range(2):
#    next(generator)
#    print 'Fetched ', i