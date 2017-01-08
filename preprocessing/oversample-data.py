import sys, re, os.path, os, json
pattern = re.compile(r'Stream.*Video.*([0-9]{3,})x([0-9]{3,})')
from subprocess import Popen, PIPE
os.chdir(os.path.dirname(os.path.realpath(__file__)))
ffmpegExe = "ffmpeg.exe"
def get_size(pathtovideo):
    cmd = ffmpegExe + " -i " + pathtovideo
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    match = pattern.search(stderr)
    if match:
        x, y = map(int, match.groups()[0:2])
    else:
        x = y = 0
    return x, y

def rotateVideo(videoFile, SavePath, angle):
    cmdCropVideo = ffmpegExe + " -i "+videoFile+" -filter:v rotate='" +str(angle) +"' -y " + SavePath
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    #-i C:\\Users\\Akshay\\Dropbox\\NYU\\Lectures\CV\Project\\data_temp\\1_1.mov -filter:v crop=256:328:0:0 -y C:\\Users\\Akshay\\Dropbox\\NYU\\Lectures\CV\Project\\data_temp\\1_1_cropped.mov', '-d'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Rotated Video by ", angle, " and saved as -> " , SavePath


with open('data/train.txt') as f:
    lines = f.read().splitlines()

f = open('oversampled/train.txt','w')
for line in lines:
    splitted = line.split()
    filepath = splitted[0]
    className = splitted[1]
    print filepath, className
    fileNameParts = filepath.split('/')
    filename = fileNameParts[1]
    savename = "oversampled\\" + filename
    tempName = "oversampled\\temp-" + filename
    tempName2 = "oversampled\\temp2-" + filename

    w,h = get_size(filepath)

    # Crop first 3 seconds
    cmdCropVideo = ffmpegExe + " -i "+filepath+" -ss 00:00:02.5 -vcodec copy -acodec copy -y " + tempName2
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Chopped 3 seconds ", filename

    # Creating loop of videos
    target = open('loop.txt', 'w')
    for i in range(0,5):
        target.write("file '" + tempName2+"'\n")
    target.close()
    cmdCropVideo = ffmpegExe + " -f concat -safe 0 -i loop.txt -c copy -y " + tempName
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Looping ", filename

    # Crop to equal length
    cmdCropVideo = ffmpegExe + " -i "+tempName+" -t 10 -vcodec copy -acodec copy -y " + savename
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Chopped to equal lengths ", filename
    
    continue
    # Crop bottom half
    cmdCropVideo = ffmpegExe + " -i "+tempName2+" -filter:v crop="+str(w)+":" + str(h/2) +":0:0 -y " + savename
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Cropped ", filename
    
    f.write(line+'\n')
    fileNameParts = filename.split('.')
    clockWiseFileName = '.'.join([fileNameParts[0] + "_cw",fileNameParts[1]])
    rotateVideo(savename,"oversampled/" +  clockWiseFileName ,'PI/40')
    f.write('data/' + clockWiseFileName +' '+className+'\n')
    antiClockWiseFileName = '.'.join([fileNameParts[0] + "_acw",fileNameParts[1]])
    rotateVideo(savename,"oversampled/" +  antiClockWiseFileName ,'-PI/40')
    f.write('data/' + antiClockWiseFileName +' '+className+'\n')

    try:
        os.remove(tempName2)
    except OSError:
        pass

    try:
        os.remove(tempName)
    except OSError:
        pass

f.close()

# Crop Validation Videos
with open('data/val.txt') as f:
    lines = f.read().splitlines()
from shutil import copyfile
copyfile('data/val.txt','oversampled/val.txt')
for line in lines:
    splitted = line.split()
    filepath = splitted[0]
    className = splitted[1]
    print filepath, className
    fileNameParts = filepath.split('/')
    filename = fileNameParts[1]
    savename = "oversampled\\" + filename
    tempName = "oversampled\\temp-" + filename
    tempName2 = "oversampled\\temp2-" + filename

    w,h = get_size(filepath)

    # Crop first 3 seconds
    cmdCropVideo = ffmpegExe + " -i "+filepath+" -ss 00:00:02.5 -vcodec copy -acodec copy -y " + tempName2
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Chopped 3 seconds ", filename

    # Creating loop of videos
    target = open('loop.txt', 'w')
    for i in range(0,5):
        target.write("file '" + tempName2+"'\n")
    target.close()
    cmdCropVideo = ffmpegExe + " -f concat -safe 0 -i loop.txt -c copy -y " + tempName
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Looping ", filename

    # Crop to equal length
    cmdCropVideo = ffmpegExe + " -i "+tempName+" -t 10 -vcodec copy -acodec copy -y " + savename
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Chopped to equal lengths ", filename
    continue
    # Crop bottom half
    cmdCropVideo = ffmpegExe + " -i "+tempName2+" -filter:v crop="+str(w)+":" + str(h/2) +":0:0 -y " + savename
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Cropped ", filename

    try:
        os.remove(tempName2)
    except OSError:
        pass

    try:
        os.remove(tempName)
    except OSError:
        pass


# Crop Test Videos too
with open('data/test.txt') as f:
    lines = f.read().splitlines()
copyfile('data/test.txt','oversampled/test.txt')
for line in lines:
    splitted = line.split()
    filepath = splitted[0]
    className = splitted[1]
    print filepath, className
    fileNameParts = filepath.split('/')
    filename = fileNameParts[1]
    savename = "oversampled\\" + filename
    tempName = "oversampled\\temp-" + filename
    tempName2 = "oversampled\\temp2-" + filename

    w,h = get_size(filepath)

    # Crop first 3 seconds
    cmdCropVideo = ffmpegExe + " -i "+filepath+" -ss 00:00:02.5 -vcodec copy -acodec copy -y " + tempName2
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Chopped 3 seconds ", filename

    # Creating loop of videos
    target = open('loop.txt', 'w')
    for i in range(0,5):
        target.write("file '" + tempName2+"'\n")
    target.close()
    cmdCropVideo = ffmpegExe + " -f concat -safe 0 -i loop.txt -c copy -y " + tempName
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Looping ", filename

    # Crop to equal length
    cmdCropVideo = ffmpegExe + " -i "+tempName+" -t 10 -vcodec copy -acodec copy -y " + savename
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Chopped to equal lengths ", filename
    
    continue
    # Crop bottom half
    cmdCropVideo = ffmpegExe + " -i "+tempName2+" -filter:v crop="+str(w)+":" + str(h/2) +":0:0 -y " + savename
    process = Popen(cmdCropVideo, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print "Cropped ", filename

    try:
        os.remove(tempName2)
    except OSError:
        pass

    try:
        os.remove(tempName)
    except OSError:
        pass