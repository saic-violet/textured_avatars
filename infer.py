import argparse
import time

import numpy as np
import json

import torch
import torch.nn.functional as F

from stickman_drawer import stickman as st

from models import translation_generator, texture_model


def get_args(parser):
    parser.add_argument('--gpu_id', type=int, default=0)

    # Log model
    parser.add_argument('--path_to_save', type=str, default='./results/')
    parser.add_argument('--log_dir', type=str, default='./log_validation/')

    # From checkpoint
    parser.add_argument('--checkpoint_path', type=str, default='')

    parser.add_argument('--checkpoint_path_texture', type=str, default='')

    parser.add_argument('--num_classes', type=int, default=24)

    parser.add_argument('--mode', type=str, default='uv', help='uv|rgb')

    parser.add_argument('--np_array', action='store_true')

    parser.add_argument('--ft_texture', action='store_true')

    parser.add_argument('--is_half', action='store_true')

    parser.add_argument('--use_enhancement', action='store_true')

    translation_generator.get_args(parser)
    texture_model.get_args(parser)



class InferenceModule(object):
    def __init__(self, opt, use_cuda=True, output_bgr=True, debug=False):
        self.use_cuda = use_cuda
        self.output_bgr = output_bgr
        self.debug = debug

        # for k,v in opt._get_kwargs():
        #     print(k,v)

        # Load model params
        self.model = translation_generator.get_net(opt)
        self.model.load_state_dict(torch.load(opt.checkpoint_path, map_location=torch.device('cpu')))
        self.model.eval()
        if self.use_cuda: self.model.cuda()

        # Load textures
        self.texture = texture_model.get_net(opt)
        self.texture.load_state_dict(torch.load(opt.checkpoint_path_texture, map_location=torch.device('cpu')))
        self.texture.eval()
        if self.use_cuda: self.texture.cuda()

        if opt.is_half:
            print('[ImageRenderer]: FP16 mode')
            if self.use_cuda:
                self.to_tensor = lambda x: torch.HalfTensor(x).cuda()
            else:
                self.to_tensor = lambda x: torch.HalfTensor(x)
            self.model.half()
            self.texture.half()
        else:
            if self.use_cuda:
                self.to_tensor = lambda x: torch.FloatTensor(x).cuda()
            else:
                self.to_tensor = lambda x: torch.FloatTensor(x)

        # Load stickman drawer
        self.stickman_drawer = st.StickmanData_C(False)

        self.opt = opt

    @staticmethod
    def get_model(model_ckpt, texture_ckpt, use_cuda=True, output_bgr=True, debug=False):
        opts={'checkpoint_path': model_ckpt, 'checkpoint_path_texture': texture_ckpt}

        parser = argparse.ArgumentParser()
        get_args(parser)
        opt = parser.parse_known_args()[0]
        opt.gen_num_input_channels = opts.get('gen_num_input_channels', '27')
        opt.model = opts.get('model', 'residual')
        opt.gen_input_size = opts.get('gen_input_size', 512)
        opt.gen_output_size = opts.get('gen_output_size', 512)
        opt.gen_norm_layer = opts.get('gen_norm_layer', 'instance')
        opt.gen_upsampling_layer = opts.get('gen_upsampling_layer', 'upsampling_nearest')
        opt.mode = opts.get('mode', 'rgb')
        opt.checkpoint_path = opts['checkpoint_path']
        opt.checkpoint_path_texture = opts['checkpoint_path_texture']
        opt.checkpoint_path_face_enhancer = opts.get('checkpoint_path_face_enhancer', '')
        opt.is_half = opts.get('is_half', False)
        opt.use_enhancement = opts.get('use_enhancement', False)
        opt.gen_max_channels = opts.get('gen_max_channels', 256)
        opt.gen_num_output_channels = opts.get('gen_num_output_channels', '25;48')
        opt.gen_output_nonlinearities = opts.get('gen_output_nonlinearities', 'none;tanh')
        opt.use_true_z = opts.get('use_true_z', False)
        assert opt.checkpoint_path_face_enhancer or not opt.use_enhancement, 'Specify path to enhancer checkpoint'

        return InferenceModule(opt, use_cuda, output_bgr, debug)

    def infer(self, pose):
        w, h = 512, 512
        pose = pose.astype(np.float32)

        pose_scaled = np.copy(pose)
        initial_scale = 512 / 1080
        pose_scaled[:, 0] *= initial_scale
        pose_scaled[:, 1] *= initial_scale
        pose_scaled[:, 2] *= 100

        stickman = np.zeros(h * w * 27, dtype=np.float32)

        self.stickman_drawer.drawStickman2(output=stickman,
                                           input=pose_scaled.reshape((4 * pose_scaled.shape[0])),
                                           w=w, h=h, lineWidthPose=4, lineWidthFaceHand=2)

        stickman = self.to_tensor(stickman)
        stickman = stickman.view(-1, h, w)[None]
        stickman = stickman / 127.5 - 1.

        with torch.no_grad():
            _, (classes_logits, uv) = self.model.forward([stickman])
            generated_image = self.texture.forward(uv, classes_logits, color=-1)

        generated_image = generated_image * 0.5 + 0.5
        generated_image = generated_image[0]
        generated_image = torch.flip(generated_image, [0])  # bgr -> rgb
        generated_image = generated_image.permute(1, 2, 0) * 255.
        generated_image = generated_image.cpu().numpy().astype(np.uint8)

        return generated_image
