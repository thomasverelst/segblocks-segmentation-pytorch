'''
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys

import torch
import torch.nn as nn
import numpy as np
import segblocks

def flops_to_string(flops, units='GMac', precision=2):
    if units is None:                
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'

def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num

def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)
    net_main_module.compute_total_flops_cost = compute_submodule_average_flops_cost.__get__(net_main_module)
    net_main_module.total_flops_cost_repr = total_flops_cost_repr.__get__(net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module

def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image and number of images

    """

    num_images = self.__batch_counter__
    if num_images == 0:
        return 0, 0

    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / num_images, num_images

def compute_submodule_average_flops_cost(self, submodule_depth: int =0):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns the mean flops consumption per image, for all submodules up to depth 
    'submodule_depth' (>=0)
    """
    from collections import defaultdict
    
    num_images = self.__batch_counter__
    modules_flops = defaultdict(dict)
    if num_images == 0:
        return modules_flops, num_images

    # build module list
    modules_at_depth = defaultdict(list, {0: [('all', self),]})
    for depth in range(1, submodule_depth+1):
        for _, module in modules_at_depth[depth-1]:
            for name, mod_children in module.named_children():
                modules_at_depth[depth].append((name, mod_children))    

    # for every module in the list, get the flops
    for depth in range(0, submodule_depth+1):
        for name, module in  modules_at_depth[depth]:
            cls_name = str(module.__class__.__name__)
            if name not in modules_flops[depth]:
                modules_flops[depth][name] = [0, cls_name]
            for mod in module.modules():
                if is_supported_instance(mod):
                    modules_flops[depth][name][0] += mod.__flops__
            modules_flops[depth][name][0] /= num_images

    return modules_flops, num_images

def total_flops_cost_repr(self, submodule_depth=0, units='GMac', precision=2):
    '''
    Print an overview of the GMACS per module
    up to depth 'submodule_depth'
    '''
    modules_flops, num_images = self.compute_total_flops_cost(submodule_depth)
    s = '\n======= FLOPSCOUNTER =======\n'
    s += f'images: {num_images}\n'
    for depth, modules in modules_flops.items():
        s += f'# depth {depth}: \n'
        for name, (flops, cls_name) in sorted(modules.items()):
            s += f'  {name:20} ({cls_name:10}): {flops_to_string(flops, units=units, precision=precision):>15}\n'
    s += '============================\n'
    return s


def start_flops_count(self, only_conv_and_linear=False):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    if only_conv_and_linear:
        self.apply(add_flops_counter_hook_function_conv_and_linear)
    else:
        self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)

# ---- Internal functions
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    # assert input.dim() == 4
    output_last_dim = output.shape[-1]  # pytorch checks dimensions, so here we don't care much
    module.__flops__ += int(np.prod(input.shape) * output_last_dim)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    assert input.dim() == 4
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    module.affine
    input = input[0]
    assert input.dim() == 4

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def deconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    assert input.dim() == 4

    batch_size = input.shape[0]
    input_height, input_width = input.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

    active_elements_count = batch_size * input_height * input_width
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        output_height, output_width = output.shape[2:]
        bias_flops = out_channels * batch_size * output_height * output_height
    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    assert input.dim() == 4

    batch_size = output.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = np.prod(kernel_dims) * in_channels * filters_per_channel
    active_elements_count = batch_size * np.prod(output_dims)

    output_height, output_width = output.shape[2:]
    active_elements_count = batch_size * output_height * output_width

    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        # print('Warning! No positional inputs found for a module, assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0

MODULES_MAPPING = {
    # convolutions
    torch.nn.Conv1d: conv_flops_counter_hook,
    torch.nn.Conv2d: conv_flops_counter_hook,
    torch.nn.Conv3d: conv_flops_counter_hook,
    # activations
    torch.nn.ReLU: relu_flops_counter_hook,
    torch.nn.PReLU: relu_flops_counter_hook,
    torch.nn.ELU: relu_flops_counter_hook,
    torch.nn.LeakyReLU: relu_flops_counter_hook,
    torch.nn.ReLU6: relu_flops_counter_hook,
    # poolings
    torch.nn.MaxPool1d: pool_flops_counter_hook,
    torch.nn.AvgPool1d: pool_flops_counter_hook,
    torch.nn.AvgPool2d: pool_flops_counter_hook,
    torch.nn.MaxPool2d: pool_flops_counter_hook,
    torch.nn.MaxPool3d: pool_flops_counter_hook,
    torch.nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    torch.nn.BatchNorm1d: bn_flops_counter_hook,
    torch.nn.BatchNorm2d: bn_flops_counter_hook,
    torch.nn.BatchNorm3d: bn_flops_counter_hook,
    # FC
    torch.nn.Linear: linear_flops_counter_hook,
    # Upscale
    torch.nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    torch.nn.ConvTranspose2d: deconv_flops_counter_hook,
}

MODULES_MAPPING_CONV_LINEAR = {
    # convolutions
    torch.nn.Conv1d: conv_flops_counter_hook,
    torch.nn.Conv2d: conv_flops_counter_hook,
    torch.nn.Conv3d: conv_flops_counter_hook,
    # FC
    torch.nn.Linear: linear_flops_counter_hook,
    # Deconvolution
    torch.nn.ConvTranspose2d: deconv_flops_counter_hook,
}

def is_supported_instance(module):
    if type(module) in MODULES_MAPPING:
        return True
    return False

def wrap_segblocks_flops_counter(flops_counter_hook):
    def segblocks_flops_counter_hook(module, input, output):
        if segblocks.is_dualrestensor(input[0]): 
            flops_counter_hook(module, (input[0].highres,), output.highres)
            flops_counter_hook(module, (input[0].lowres,), output.lowres)
        else:
            flops_counter_hook(module, input, output)
    return segblocks_flops_counter_hook


def add_flops_counter_hook_function(module):
    return _add_flops_counter_hook_function(module, only_conv_and_linear=False)

def add_flops_counter_hook_function_conv_and_linear(module):
    return _add_flops_counter_hook_function(module, only_conv_and_linear=True)

def _add_flops_counter_hook_function(module, only_conv_and_linear=False):
    mapping = MODULES_MAPPING_CONV_LINEAR if only_conv_and_linear else MODULES_MAPPING
    if hasattr(module, '__flops_handle__'):
        return
    if type(module) not in mapping:
        return
    function = mapping[type(module)]
    function = wrap_segblocks_flops_counter(function)
    handle = module.register_forward_hook(function)
    module.__flops_handle__ = handle
    module.__flops_function__ = function


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
            del module.__flops_function__

