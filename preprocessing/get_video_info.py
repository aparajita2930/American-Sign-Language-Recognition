import cv2
import numpy as np

def get_numframe_duration(invideo):
    info = []
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    CAP_PROP_FPS = cv2.CAP_PROP_FPS

    video = cv2.VideoCapture(invideo)
    if not video.isOpened():
        raise Exception('Could not open the video.\n')
    num_frames = int(video.get(CAP_PROP_FRAME_COUNT))
    info.append(num_frames)
    #print num_frames
    fps = float(video.get(CAP_PROP_FPS))
    #print fps
    duration = num_frames/fps
    info.append(duration)
    #print video.get(cv2.CAP_PROP_FOURCC)
    return info

def convert_video(invideo, resize=None, start_frame=0, end_frame=None, length=None):
    frames = []
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES

    video = cv2.VideoCapture(invideo)
    if not video.isOpened():
        raise Exception('Could not open the video.')

    num_frames = int(video.get(CAP_PROP_FRAME_COUNT))
    if start_frame >= num_frames or start_frame < 0:
        raise Exception('Invalid initial frame given.')

    #First frame
    video.set(CAP_PROP_POS_FRAMES, start_frame)
    #Read till end_frame or till length
    if end_frame:
        end_frame = end_frame if end_frame < num_frames else num_frames
    elif length:
        end_frame = start_frame + length
        end_frame = end_frame if end_frame < num_frames else num_frames
    else:
        end_frame = num_frames

    if end_frame < start_frame:
        raise Exception('Invalid ending position.')

    for i in range(start_frame, end_frame):
        ret, frame = video.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            return None

        if resize: #width by height
            frame = cv2.resize(frame, (resize[1], resize[0]))
        frames.append(frame)

    outvideo = np.array(frames, dtype=np.float32)
    outvideo = outvideo.transpose(3, 0, 1, 2)
    return outvideo

#print get_numframe_duration('../data_temp/1_1.mov')
#info =  get_numframe_duration('data_temp/1_1.mov')
#print info[0]
#print info[1]

#print convert_video('data_temp/1_2.mov')