import json
import numpy as np
import cv2
import glob
import argparse
import tqdm
import os

from infer import InferenceModule


def load_pose(path):
    with open(path) as f:
        data = json.load(f)
    return np.pad(np.array(data['full']), [(0,0), (0,1)], 'constant', constant_values=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['person_1', 'person_2'])
    args = parser.parse_args()

    ckpt_dir = './data/checkpoints'

    if args.model == 'person_1':
        model_ckpt = './data/checkpoints/person_1/model_params_10'
        texture_ckpt = './data/checkpoints/person_1/texture_params_10'
        poses = './data/keypoints/person_1'
        out = './out/person_1'
    else:
        model_ckpt = './data/checkpoints/person_2/model_params_10'
        texture_ckpt = './data/checkpoints/person_2/texture_params_10'
        poses = './data/keypoints/person_2'
        out = './out/person_2'

    model = InferenceModule.get_model(model_ckpt, texture_ckpt)

    os.makedirs(out, exist_ok=True)

    for fn in tqdm.tqdm(glob.glob(poses + '/*.json')):
        pose = load_pose(fn)
        img = model.infer(pose)

        pose_short = fn.split('/')[-1].split('.')[0]
        cv2.imwrite(f'{out}/{args.model}_pose_{pose_short}.png', img)
