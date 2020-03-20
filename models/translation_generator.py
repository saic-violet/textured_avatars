from torch import nn
from models import common
from math import log
import copy
import time
import torch


def get_args(parser):
    """Add generator-specific options to the parser"""

    parser.add_argument(
        '--gen_num_channels', default=32, type=int,
        help='initial number of channels in convolutions')

    parser.add_argument(
        '--gen_max_channels', default=256, type=int,
        help='maximum number of channels in convolutions')

    parser.add_argument(
        '--gen_kernel_size', default=4, type=int,
        help='kernel size in downsampling convolutions')

    parser.add_argument(
        '--gen_latent_size', default=32, type=int,
        help='spatial size of a tensor at which residual blocks are operating')

    parser.add_argument(
        '--gen_num_res_blocks', default=7, type=int,
        help='number of residual blocks')

    parser.add_argument(
        '--gen_conv_layer', default='conv', type=str,
        help='convolutional module,'
             'allowed values: conv|grid_conv')

    parser.add_argument(
        '--gen_norm_layer', default='batch', type=str,
        help='type of the normalization layer,'
             'allowed values: instance|batch|none')

    parser.add_argument(
        '--gen_upsampling_layer', default='conv_transpose', type=str,
        help='upsampling module,'
             'allowed values: conv_transpose|pixel_shuffle|upsampling_nearest')

    parser.add_argument(
        '--gen_num_input_channels', default='21', type=str, 
        help='number of input channels in each branch of generator,'
             'comma-separated list with input dimensions')

    parser.add_argument(
        '--gen_num_output_channels', default='25;48', type=str, 
        help='number of output channels in each output branch of generator,'
             '\';\' goes for different deep branch,' 
             '\',\' goes for different shallow branch')
    
    parser.add_argument(
        '--gen_output_nonlinearities', default='none;tanh', type=str, 
        help='nonlinearities after each output branch of generator,'
             '\';\' goes for different deep branch,' 
             '\',\' goes for different shallow branch')

    parser.add_argument(
        '--gen_input_size', default=256, type=int, 
        help='size of the image on which generator operates')

    parser.add_argument(
        '--gen_output_size', default=256, type=int, 
        help='size of the image on which generator operates')

    parser.add_argument('--light', action='store_true', 
    	help='generate additive light')

    parser.add_argument('--gen_output_size_light', default=128, 
    	type=int, help='generate low-res light')


def get_net(opt):
    return Generator(opt)


class Generator(nn.Module):
    """Translation generator architecture by Johnston et al"""

    def __init__(self, opts):
        super(Generator, self).__init__()

        self.light = opts.light

        conv_type = opts.gen_conv_layer

        # Get constructors of the required modules
        conv_layer = common.get_conv_layer(conv_type)
        norm_layer = common.get_norm_layer(opts.gen_norm_layer)
        upsampling_layer = common.get_upsampling_layer(opts.gen_upsampling_layer)

        # Calculate the amount of downsampling and upsampling convolutional blocks
        num_down_blocks = int(log(
            opts.gen_input_size // 
            opts.gen_latent_size, 2))

        num_up_blocks = int(log(
            opts.gen_output_size // 
            opts.gen_latent_size, 2))

        if opts.light:
            num_up_blocks_light = int(log(
               opts.gen_output_size_light // 
               opts.gen_latent_size, 2))

        # Read parameters for convolutional blocks
        in_channels = opts.gen_num_channels
        padding = (opts.gen_kernel_size - 1) // 2
        bias = norm_layer != nn.BatchNorm2d
        #bias = True # only for 1 experiment

        # Downsampling layer
        down_layers = []

        # Downsampling blocks
        for i in range(num_down_blocks):

            # Increase the number of channels by 2x
            out_channels = min(in_channels * 2, opts.gen_max_channels)

            down_layers += [
                conv_layer(in_channels, out_channels, 
                           opts.gen_kernel_size, stride=2,
                           padding=padding, 
                           bias=bias),
                norm_layer(out_channels),
                nn.ReLU(True)]

            in_channels = out_channels

        in_channels = opts.gen_max_channels

        # Residual blocks
        num_res_blocks = opts.gen_num_res_blocks

        for i in range(num_res_blocks - num_res_blocks//2):
            down_layers += [common.ResBlock(in_channels, conv_layer, norm_layer)]

        # Get list of input channels in branches
        input_channels_list = list(map(int, opts.gen_num_input_channels.split(',')))

        # List for downsampling branches
        self.input_branches = nn.ModuleList()

        for current_input_channels in input_channels_list:

            # First layer without normalization
            current_layers = []

            if num_down_blocks == num_up_blocks:
                current_layers += [
                    conv_layer(current_input_channels, 
                        opts.gen_num_channels, 7, 1, 3, bias=False), 
                    nn.ReLU(True)]

            current_layers += copy.deepcopy(down_layers)
            self.input_branches += [nn.Sequential(*current_layers)]

        # Residual decoder blocks
        residual_layers = []

        for i in range(num_res_blocks//2):
            residual_layers += [common.ResBlock(in_channels, conv_layer, norm_layer)]

        # Upsampling decoder blocks
        upsampling_layers = []

        for i in range(num_up_blocks):

            # Decrease the number of channels by 2x
            out_channels = opts.gen_num_channels * 2**(num_up_blocks-i-1)
            out_channels = max(min(out_channels, 
                                   opts.gen_max_channels), 
                               opts.gen_num_channels)

            upsampling_layers += upsampling_layer(
                in_channels, out_channels, opts.gen_kernel_size, 2, 
                bias, conv_type)
            upsampling_layers += [
                norm_layer(out_channels),
                nn.ReLU(True)]

            if opts.light and i == num_up_blocks_light - 1:
                out_channels_light = out_channels

            in_channels = out_channels

        # Get output channels per each branch
        branches_out_channels = opts.gen_num_output_channels.split(';')
        branches_nonlinearities = opts.gen_output_nonlinearities.split(';')

        # Output branches(deep)
        self.branches_residual = nn.ModuleList()
        self.branches_upsampling = nn.ModuleList()

        # Final layers (may be several per branch)
        self.final_layers = nn.ModuleList()

        for branch_out_channels, branch_nonlinearities in zip(branches_out_channels, branches_nonlinearities):

            # Deep copy the main chunk of the branch
            self.branches_residual += [nn.Sequential(*copy.deepcopy(residual_layers))]
            self.branches_upsampling += [nn.Sequential(*copy.deepcopy(upsampling_layers))]
            
            # Each branch has multiple heads
            heads_out_channels = map(int, branch_out_channels.split(','))
            heads_nonlinearity_types = branch_nonlinearities.split(',')

            branch_heads = nn.ModuleList()
            for head_out_channels, head_nonlinearity_type in zip(heads_out_channels, heads_nonlinearity_types): 
                branch_heads += [nn.Sequential(
                    nn.Conv2d(out_channels, head_out_channels, 7, 1, 3, bias=False),
                    common.get_nonlinear_layer(head_nonlinearity_type))]

            self.final_layers += [branch_heads]

        if opts.light:
            light_branch = [nn.Sequential(*copy.deepcopy(residual_layers))]
            light_branch += [nn.Sequential(*copy.deepcopy(upsampling_layers[:4*num_up_blocks_light]))]
            light_branch += [nn.Sequential(nn.Conv2d(out_channels_light, 1, 7, 1, 3, bias=False),
                common.get_nonlinear_layer('tanh'))]
            self.light_branch = nn.Sequential(*light_branch)

        # Initialize weights
        self.apply(common.weights_init)


    def forward(self, inputs):

        coarse_features = 0

        for i, module in enumerate(self.input_branches):
            coarse_features += module(inputs[i])
                        
        fine_features = []

        for module_res, module_up in zip(self.branches_residual, self.branches_upsampling):
            fine_features += [module_up(module_res(coarse_features))]

        final_outputs = []

        for i, modules in enumerate(self.final_layers):
            for module in modules:
                final_outputs += [module(fine_features[i])]

        if self.light:
            low_res_light = self.light_branch(coarse_features)
            final_outputs += [nn.functional.interpolate(low_res_light, 
                size=(final_outputs[0].shape[2], final_outputs[0].shape[3]), 
                mode='bilinear', 
                align_corners=False)]
                  
        return fine_features, final_outputs
