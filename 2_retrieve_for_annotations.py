import os
import json
import cv2
import numpy as np
import shutil
import sys

font = cv2.FONT_HERSHEY_SIMPLEX




for setup_file in ['SETUP1.json', 'SETUP2.json']:
	loaded_json = json.load(open(setup_file))
	CAMERA_IDS = sorted(loaded_json['FRAMES_SYNCH'].keys())

	min_size = float('inf')
	list_list_frames = []
	for camera_id in CAMERA_IDS:
		frames_path = os.path.join(loaded_json['DATA_ROOT'], camera_id)
		
		assert os.path.exists(frames_path)

		list_frames = sorted(os.listdir(frames_path))[loaded_json['FRAMES_SYNCH'][camera_id]-1:]
		list_frames = list(map(lambda x:os.path.join(camera_id, x), list_frames))
		if len(list_frames) < min_size:
			min_size = len(list_frames)
			print(min_size)

		list_list_frames.append(list_frames)

	#sys.exit()


	for experiment in loaded_json['EXPERIMENTS']:
		beg, end  = loaded_json['EXPERIMENTS'][experiment]

		for idx, camera_id in enumerate(CAMERA_IDS):
			w = open(os.path.join('TO_ANNOTATE/' + camera_id.replace('/','-') + '_' + experiment+ '.txt'), 'w')
			video = cv2.VideoWriter('TO_ANNOTATE/' + camera_id.replace('/','-') + '_' + experiment+ '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (1920, 1080))

			list_frames = list_list_frames[idx][:min_size][beg:end]

			
			#elm in enumerate(list(zip(list_list_frames[0],list_list_frames[1],list_list_frames[2],list_list_frames[3]))[beg:end])[idx]:

				
			imgs = []
			for fp in list_frames:
				image_fp = os.path.join(loaded_json['DATA_ROOT'], fp)
				print(image_fp)
				assert os.path.exists(image_fp)
				w.write(image_fp + '\n')

				video.write(cv2.imread(image_fp))

				#new_fp = os.path.join(out_path, '-'.join(fp.split('/')))
				#shutil.copy(image_fp, new_fp)
				#print(fp)
				#print(new_fp)
				#print()
