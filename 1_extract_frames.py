import os
import subprocess

ROOT = '/cvlabdata2/cvlab/datasets_hugonot/DATA/physic_overlay/DATA'
for root, dirs, files in os.walk(ROOT):
    for file in [x for x in files if x.endswith('.MP4')]:
        basename = file[:-4]
        video_fp = os.path.join(root, file)
        frames_dir = os.path.join(root, basename)
        if not os.path.exists(frames_dir): 
        	os.mkdir(frames_dir)
        print(basename, video_fp)
        ffmpeg_command = 'ffmpeg -i ' + video_fp + ' -qscale:v 2 ' + frames_dir+'/%05d.jpg'
        print(ffmpeg_command)
        # subprocess.call(ffmpeg_command,shell=True)
