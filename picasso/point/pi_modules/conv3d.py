import torch
import point_conv3d as module


class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, nn_count, nn_index, bin_index):
        nn_count.requires_grad = False
        nn_index.requires_grad = False
        bin_index.requires_grad = False
        output = module.SPH3D_forward(input, filter, nn_count, nn_index, bin_index)
        variables = [input, filter, nn_count, nn_index, bin_index]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_filter = module.SPH3D_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, grad_filter, None, None, None


def conv3d(input, filter, nn_count, nn_index, bin_index):
        return ConvFunction.apply(input, filter, nn_count, nn_index, bin_index)


class FuzzyConv3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, nn_count, nn_index, bin_index, bin_coeff):
        nn_count.requires_grad = False
        nn_index.requires_grad = False
        bin_index.requires_grad = False
        bin_coeff.requires_grad = False
        output = module.fuzzySPH3D_forward(input, filter, nn_count, nn_index, bin_index, bin_coeff)
        variables = [input, filter, nn_count, nn_index, bin_index, bin_coeff]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_filter = module.fuzzySPH3D_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, grad_filter, None, None, None, None


def fuzzyconv3d(input, filter, nn_count, nn_index, bin_index, bin_coeff):
        return FuzzyConv3dFunction.apply(input, filter, nn_count, nn_index, bin_index, bin_coeff)
