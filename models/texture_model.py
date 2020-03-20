import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_args(parser):
    """Add texture model specific options to the parser"""
    parser.add_argument(
        '--texture_size', default=256, type=int, 
        help='texture size')

    parser.add_argument('--texture_path', type=str, default='')
    parser.add_argument('--train_texture', action='store_true')


def get_net(opt):
    return TextureModel(opt)

class TextureModel(nn.Module):
    """Class for uv texturing."""
    def __init__(self, opt):
        super(TextureModel, self).__init__()

        if opt.texture_path == '':
            textures = np.random.randn(opt.num_classes, 3, opt.texture_size, opt.texture_size)
        else:   
            textures = 2*np.load(opt.texture_path).transpose(0, 3, 2, 1) - 1
        
        self.texture_variable = nn.Parameter(torch.FloatTensor(textures))
        #self.texture_addition = torch.FloatTensor(textures * 0)
        
        if opt.train_texture:
            self.texture_variable.requires_grad = True
        else:
            self.texture_variable.requires_grad = False
        self.num_densepose_classes = opt.num_classes

    def recover_image_batch(self, class_proba, u, v):
        
        batch_size = u.shape[0]
        height = u.shape[-2]
        width = u.shape[-1]

        u_coords_batch = torch.transpose(u, 0, 1)
        v_coords_batch = torch.transpose(v, 0, 1)
        u_coords_batch = u_coords_batch.contiguous().view(self.num_densepose_classes, 1,
            batch_size*height, width, 1)
        v_coords_batch = v_coords_batch.contiguous().view(self.num_densepose_classes, 1,
            batch_size*height, width, 1)
        uv_coords_batch = torch.cat((u_coords_batch, v_coords_batch), 4)

        for densepose_class in range(1, self.num_densepose_classes + 1):

            class_textures = self.texture_variable[densepose_class - 1][None, :, :, :]
            #                 self.texture_addition[densepose_class - 1][None, :, :, :]

            
            recovered_class_image_batch = nn.functional.grid_sample(class_textures,
                    uv_coords_batch[densepose_class - 1])[0]

            class_mask = class_proba[:, densepose_class].contiguous().view(1,
                    batch_size*height, width)
            
            if densepose_class == 1:
                recovered_image_batch = recovered_class_image_batch*class_mask
            else:
                recovered_image_batch += recovered_class_image_batch*class_mask

        recovered_image_batch = recovered_image_batch.contiguous().view(3, batch_size, height, width)
        recovered_image_batch = torch.transpose(recovered_image_batch, 0, 1)
        
        return recovered_image_batch

    def forward(self, uv_gen, uv_seg_gen, color=-1.0, proba=False):
        u_coordinates = uv_gen[:, :self.num_densepose_classes]
        v_coordinates = uv_gen[:, self.num_densepose_classes:]

        if not proba:
            classes_proba = F.softmax(uv_seg_gen, dim=1)
        else:
            classes_proba = uv_seg_gen

        recovered_image = self.recover_image_batch(classes_proba, u_coordinates, v_coordinates)
        # Add background
        return recovered_image + color*classes_proba[:, :1]

    # def set_addition