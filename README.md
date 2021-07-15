# Procedure

* Gather the videos. Initially we recorded each of the 3 experiments from 4 cameras: "6_1", "6_2", "6_3" & "6_4".
* Extract the frames. "1_extract_frames.py"
* Synchronize the frames of the 4 cameras assuming the framerate is consistent from one camera to another. "SYNC1.json" & "SYNC2.json"
* Calibration of the system through annotations of the calibration sequences
*

# Annotation workflow

https://github.com/openvinotoolkit/cvat

* Merge the frames as a video to use the "track" feature of CVAT
* Annotate. As a starter I chose to annotate every 10th frame. The "track" feature of the tool will interpolate linearly the position of the point between the key frames
* Retrieve the xml file as "for image" and "for video"

# Do we care about occlusions ?

* To handle occlussions we need several cameras. Is the overhead worth it ?
* SolvePNP from one can provide nice results
