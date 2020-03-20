This code replaces keypoints2img function. Has some overhead (didn't measure) for class object instantiation, so store and reuse StickmanData_C object.
Do not use very big lineWidth, as code is not optimized for it and performance degrades.
Now it measures distance to a line not to a segment, which increases performance, but draws a bit incorrect line endings. It is not a problem with relatively small bone widths.
 
Correct Makefile paths to python and version of python on target machine to compile.

script example:

	import stickman as st
	import pickle
	import numpy as np
	import time
	import json

	w=1280
	h=720

	with open("/home/alex/vid2vid/cpp/inpt.pkl", 'rb') as f:
		data_new = pickle.load(f)

	inputs=np.concatenate((data_new['pose3d'],data_new['face'],data_new['rhand'],data_new['lhand']))
	print(inputs.shape)

	inputs=inputs.reshape(4*131)
	inputs=inputs.astype(np.float32)

	output=np.zeros(w*h*21,dtype=np.float32)

	stickman=st.StickmanData_C()

	stickman.drawStickman(output=output,input=inputs,w=w,h=h,lineWidthPose=3,lineWidthFaceHand=1)


returns CxHxW tensor in output (21,720,1280 in this example). 
Takes numpy 1darrays: 
	inputs: 524 elements;
	output: CxHxW elements;

Update: V2:

This version draws a smooth stickman.
Constructor parameters
is_07: True - OpenPose v. 0.7 (18 pose joints), False - OpenPose v. 1.4 (24 pose joints)
is_separate_hands: True - 20 maps for each hand, False - one map for each hand
global_z_min, global_z_max - z scaling parameters

    stickman = st.StickmanData_C(is_07, is_separate_hands, global_z_min, global_z_max)
    inputs=np.concatenate((data_new['pose3d'],data_new['face'],data_new['rhand'],data_new['lhand']))
	print(inputs.shape)
	inputs=inputs.reshape(4*137)
	inputs=inputs.astype(np.float32)
	n_pose_maps = 18
	n_hand_maps = 1
	if is_07:
	    n_pose_maps = 24
    if is_separate_hands:
        n_hand_maps = 20

	output=np.zeros(w*h*(n_pose_maps + 2*n_hand_maps + 1),dtype=np.float32)
    stickman.drawStickman2(output=output_new, input=input, w=w, h=h, lineWidthPose=3, lineWidthFaceHand=1)


returns CxHxW tensor in output (27,720,1280 in this example). 
Takes numpy 1darrays: 
	inputs: 548 elements;
	output: CxHxW elements;

