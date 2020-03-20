import torch
import torch.nn as nn
import numpy as np


class NoParam(nn.Module):
    def __init__(self, model):
        super(NoParam, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def parameters(self):
        for param in []:
            yield param

    def named_parameters(self, memo=None, prefix=''):
        for param in []:
            yield param


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def norm(num_features, tp='bn'):
    if tp == 'bn':
        return nn.BatchNorm2d(num_features)
    elif tp == 'in':
        return nn.InstanceNorm2d(num_features)
    elif tp == 'none':
        return Identity()


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection' and to_pad != 0:
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    # print (layers22)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    # print (layers)
    return nn.Sequential(*layers)


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


def get_loss(name):
    
    return torch.nn['name']()


def get_conv_layer(conv_type):
    return {
        'conv': nn.Conv2d,
        'grid_conv': GridConv2d
    }[conv_type]


def get_norm_layer(norm_type):
    return {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
        'none': Identity
    }[norm_type]


def get_nonlinear_layer(nonlinearity_type):
    return {
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(1),
        'none': Identity()
    }[nonlinearity_type]


def get_upsampling_layer(upsampling_type):

    def conv_transpose_layer(in_channels, out_channels, 
        kernel_size, stride, bias, conv_type):

        padding = (kernel_size - 1) // stride
        output_padding = 1 if kernel_size % 2 else 0

        conv = {
            'conv': nn.ConvTranspose2d,
            'grid_conv': GridConvTranspose2d
        }[conv_type]

        return [conv(in_channels, out_channels, kernel_size, stride,
                     padding, output_padding, bias=bias)]

    def pixel_shuffle_layer(in_channels, out_channels, 
        kernel_size, upscale_factor, bias, conv_type):

        kernel_size -= kernel_size % 2 == 0
        padding = kernel_size // 2

        conv = {
            'conv': nn.Conv2d,
            'grid_conv': GridConv2d
        }[conv_type]

        num_channels = out_channels * upscale_factor**2

        return [conv(in_channels, out_channels, kernel_size, 1,
                     padding),
                nn.PixelShuffle(upscale_factor)]

    def upsampling_nearest_layer(in_channels, out_channels, 
        kernel_size, scale_factor, bias, conv_type):

        kernel_size -= kernel_size % 2 == 0
        padding = kernel_size // 2

        conv = {
            'conv': nn.Conv2d,
            'grid_conv': GridConv2d
        }[conv_type]

        return [nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                conv(in_channels, out_channels, kernel_size, 1,
                     padding, bias=bias)]

    return {
        'conv_transpose': conv_transpose_layer,
        'pixel_shuffle': pixel_shuffle_layer,
        'upsampling_nearest': upsampling_nearest_layer
    }[upsampling_type]


class Identity(nn.Module):

    def __init__(self, num_channels=None):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return ('{name}()'.format(name=self.__class__.__name__))


class GridConvTranspose2d(nn.ConvTranspose2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(GridConvTranspose2d, self).__init__(
            in_channels+2, out_channels, kernel_size, stride, 
            padding, output_padding, groups, bias, dilation)
        
        self.grid = None
        
    def forward(self, input, output_size=None):
        
        if self.grid is None or self.grid.size() != input.size():
        
            # Calculate new grid for that input
            b, c, h, w = input.size()
            
            self.grid = torch.meshgrid([
                torch.arange(h)/h, 
                torch.arange(w)/w])
            self.grid = torch.cat([t[None, None] for t in self.grid], 1).expand(b, 2, h, w)
            self.grid = (self.grid.type(input.dtype) - 0.5) * 2

        self.grid = self.grid.cuda(input.get_device())
        
        # Forward input with grid through the layer
        input = torch.cat([input, self.grid], 1)

        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class GridConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(GridConv2d, self).__init__(
            in_channels+2, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias)
        
        self.grid = None
        
    def forward(self, input):
        
        if self.grid is None or self.grid.size() != input.size():
        
            # Calculate new grid for that input
            b, c, h, w = input.size()
            
            self.grid = torch.meshgrid([
                torch.arange(h)/h, 
                torch.arange(w)/w])
            self.grid = torch.cat([t[None, None] for t in self.grid], 1).expand(b, 2, h, w)
            self.grid = (self.grid.type(input.dtype) - 0.5) * 2
        
        self.grid = self.grid.cuda(input.get_device())

        # Forward input with grid through the layer
        input = torch.cat([input, self.grid], 1)
        
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, conv_layer, norm_layer):
        super(ResBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        bias = norm_layer == Identity

        self.block = nn.Sequential(
            conv_layer(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels),
            nn.ReLU(True),
            conv_layer(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels))

    def forward(self, input):

        return input + self.block(input)


class ConcatBlock(nn.Module):

    def __init__(
        self,
        enc_channels,
        out_channels, 
        nonlinear_layer=nn.ReLU,
        norm_layer=None,
        norm_layer_cat=None,
        kernel_size=3):
        super(ConcatBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        norm_layer_cat = Identity if norm_layer_cat is None else norm_layer_cat

        # Get branch from encoder
        layers = get_conv_block(
                enc_channels,
                out_channels,
                nonlinear_layer,
                norm_layer,
                'same', False,
                kernel_size)
        
        layers += [norm_layer_cat(out_channels)]

        self.enc_block = nn.Sequential(*layers)

    def forward(self, input, vgg_input):

        output_enc = self.enc_block(vgg_input)
        output_dis = input

        output = torch.cat([output_enc, output_dis], 1)

        return output

def weights_init(module):

    classname = module.__class__.__name__

    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
