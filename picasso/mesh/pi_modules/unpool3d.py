import torch
import mesh_unpool3d as module


class InterpolateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vt_replace, vt_map):
        vt_replace.requires_grad = False
        vt_map.requires_grad = False
        output = module.forward(input, vt_replace, vt_map)
        variables = [input, vt_replace, vt_map]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = module.backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, None


def interpolate(input, vt_replace, vt_map):
        return InterpolateFunction.apply(input, vt_replace, vt_map)




