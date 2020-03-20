import json
import numpy as np
import stickman as st
import time
import skimage.io as io

def test():
    data = json.load(open("../test_stickman/14459_keypoints.json", "r"))
    p = data["people"][0]
    p3d = p["pose_keypoints_3d"]
    f3d = p["face_keypoints_3d"]
    hl3d = p["hand_left_keypoints_3d"]
    hr3d = p["hand_right_keypoints_3d"]
    input = np.concatenate([np.asarray(p3d), np.asarray(f3d), np.asarray(hl3d), np.asarray(hr3d)]).astype(np.float32)
    print (len(input))
    w = int(1920)
    h = int(1080)
    output = np.zeros(w * h * 21, dtype=np.float32)
    output_new = np.zeros(w * h * 21, dtype=np.float32)

    stickman = st.StickmanData_C()
    t0 = time.time()
    sz1 = len(output)
    stickman.drawStickman(output=output, input=input, w=w, h=h, lineWidthPose=3, lineWidthFaceHand=1)
    t1 = time.time()
    stickman.drawStickman2(output=output_new, input=input, w=w, h=h, lineWidthPose=3, lineWidthFaceHand=1)
    t2 = time.time()
    print('old: '+str(t1 - t0))
    print('new: ' + str(t2 - t1))
    output = np.transpose(output.reshape(21, h, w), (1,2,0))
    output_new = np.transpose(output_new.reshape(21, h, w), (1, 2, 0))
    for i in range(0, 21):
        img = output[:,:,i]
        img_new = output_new[:, :, i]
        img = img / np.max(img)
        img_new = img_new / np.max(img_new)
        io.imsave('./test/'+str(i)+'_old.png', img)
        io.imsave('./test/' + str(i) + '.png', img_new)


    output_avg = np.sum(output_new, axis=2)
    output_avg = output_avg / np.max(output_avg)
    io.imsave('./test/avg_new_07.png', output_avg)
    output_avg_old = np.sum(output, axis=2)
    output_avg_old = output_avg_old / np.max(output_avg_old)
    io.imsave('./test/avg_old_07.png', output_avg_old)



def test_draw():
    is_07 = True

    if is_07:
        data = json.load(open("../test_stickman/14459_keypoints.json", "r"))
        p = data["people"][0]
    else:
        data = json.load(open("../test_stickman/sample_12.json", "r"))
        p = data
    p3d = p["pose_keypoints_3d"]
    f3d = p["face_keypoints_3d"]
    hl3d = p["hand_left_keypoints_3d"]
    hr3d = p["hand_right_keypoints_3d"]
    input = np.concatenate([np.asarray(p3d), np.asarray(f3d), np.asarray(hl3d), np.asarray(hr3d)]).astype(np.float32)
    print (len(input))
    w = int(1920)
    h = int(1080)


    is_separate_hands = True
    is_separate_face = True
    is_earneck = True
    global_z_min = 100
    global_z_max = 500

    stickman = st.StickmanData_C(is_07, is_separate_hands, is_separate_face, is_earneck, global_z_min, global_z_max)

    n_pose_maps = 18
    n_hand_maps = 1
    n_face_maps = 1
    if not is_07:
        n_pose_maps = 24
    if is_separate_hands:
        n_hand_maps = 20
    if is_separate_face:
        n_face_maps = 39
    if is_earneck:
        n_pose_maps += 2

    map_num = n_pose_maps + 2 * n_hand_maps + n_face_maps

    print('output size ' +str(w*h*map_num))
    output_new = np.zeros(w * h * map_num, dtype=np.float32)

    t1 = time.time()
    stickman.drawStickman2(output_new, input, w, h, lineWidthPose=3, lineWidthFaceHand=1.5)
    t2 = time.time()
    print('time: ' + str(t2 - t1))

    # output = np.transpose(output.reshape(map_num, h, w), (1,2,0))
    output_new = np.transpose(output_new.reshape(map_num, h, w), (1, 2, 0))

    for i in range(0, map_num):
        # img = output[:,:,i]
        # img = img / np.max(img)

        img_new = output_new[:, :, i]
        img_new = img_new / np.max(img_new)
        # io.imsave('/home/alexander/materials/violet/stickman-drawer-master/python/test/'+str(i)+'_old.png', img)
        io.imsave('./test/' + str(i) + '.png', img_new)

    output_avg = np.sum(output_new, axis=2)
    output_avg = output_avg / np.max(output_avg)
    io.imsave('./test/avg_new.png', output_avg)

# test()
test_draw()