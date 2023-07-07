import torch
import mesh_pool3d as module


class MaxPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vt_replace, vt_map, vt_out):
        vt_replace.requires_grad = False
        vt_map.requires_grad = False
        vt_out.requires_grad = False
        outputs = module.max_forward(input, vt_replace, vt_map, vt_out)
        output, max_index = outputs
        variables = [input, max_index]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = module.max_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, None, None


def maxPool(input, vt_replace, vt_map, vt_out):
        return MaxPoolFunction.apply(input, vt_replace, vt_map, vt_out)


class AvgPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vt_replace, vt_map, vt_out):
        vt_replace.requires_grad = False
        vt_map.requires_grad = False
        vt_out.requires_grad = False
        output = module.avg_forward(input, vt_replace, vt_map, vt_out)
        variables = [input, vt_replace, vt_map]
        ctx.save_for_backward(*variables)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = module.avg_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, None, None


def avgPool(input, vt_replace, vt_map, vt_out):
        return AvgPoolFunction.apply(input, vt_replace, vt_map, vt_out)



