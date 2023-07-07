import torch
import point_pool3d as module


class MaxPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_count, nn_index):
        nn_count.requires_grad = False
        nn_index.requires_grad = False
        outputs = module.max_forward(input, nn_count, nn_index)
        output, max_index = outputs
        variables = [input, max_index]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = module.max_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, None

def maxPool(input, nn_count, nn_index):
        return MaxPoolFunction.apply(input, nn_count, nn_index)


class AvgPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_count, nn_index):
        nn_count.requires_grad = False
        nn_index.requires_grad = False
        output = module.avg_forward(input, nn_count, nn_index)
        variables = [input, nn_count, nn_index]
        ctx.save_for_backward(*variables)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = module.avg_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, None


def avgPool(input, nn_count, nn_index):
        return AvgPoolFunction.apply(input, nn_count, nn_index)
