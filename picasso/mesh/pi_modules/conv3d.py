import torch
import mesh_conv3d as module


class f2f_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, textureIn, filter, baryCoeff, numTexture):
        baryCoeff.requires_grad = False
        numTexture.requires_grad = False
        output = module.facet2facet_forward(textureIn, filter, baryCoeff, numTexture)
        variables = [textureIn, filter, baryCoeff, numTexture]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_filter = module.facet2facet_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, grad_filter, None, None


def facet2facet(textureIn, filter, baryCoeff, numTexture):
        '''
        Compute the feature of each triangular face based on the provided interpolated interior features.
        The common input feature for this function is facet textures.
        Input:
            textureIn:   (concat_NiK, in_channels) float32 array, input facet interpolated features
            filter:      (out_channels, in_channels, 3) float32 array, convolution filter
            baryCoeff:   (concat_NiK, 3) float32 array, face interior interpolation weights,
                                                        Barycentric Coordinates
            numTexture:  (concat_NfIn) int32 vector, number of interpolated interior points in each facet
        Output:
            output:      (concat_NfIn, out_channels) float32 array, output facet features
                                                     out_channels = in_channels * multiplier
        '''
        return f2f_Function.apply(textureIn, filter, baryCoeff, numTexture)




class v2f_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, face):
        face.requires_grad = False
        output = module.vertex2facet_forward(input, filter, face)
        variables = [input, filter, face]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_filter = module.vertex2facet_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, grad_filter, None


def vertex2facet(input, filter, face):
        '''
        Compute the feature of each triangular face based on its vertices' features by interpolation
        Input:
            input:  (concat_NvIn, in_channels) float32 array, input vertex/point features
            filter: (3, in_channels, multiplier) float32 array, convolution filter
            face:   (concat_NfIn, 3) int32 array, vertex list of each facet
        Output:
            output: (concat_NfIn, out_channels) float32 array, output facet features
                                                     out_channels = in_channels * multiplier
        '''
        return v2f_Function.apply(input, filter, face)




class f2v_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, coeff, face, nfCount, vtMap):
        coeff.requires_grad = False
        face.requires_grad = False
        nfCount.requires_grad = False
        vtMap.requires_grad = False
        output = module.facet2vertex_forward(input, filter, coeff, face, nfCount, vtMap)
        variables = [input, filter, coeff, face, nfCount, vtMap]
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_filter = module.facet2vertex_backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, grad_filter, None, None, None, None


def facet2vertex(input, filter, coeff, face, nfCount, vtMap):
        '''
        Input:
            input:   (concat_NfIn, in_channels) float32 array, input facet features
            filter:  (clusterSize, in_channels, multiplier) float32 array, convolution filter
            coeff:   (concat_NfIn, modelSize) float32 array, coefficients for each model
            face:    (concat_NfIn, 3) int32 array, vertex list of each facet
            nfCount: (concat_NvIn) int32 vector, number of adjacent faces of each vertex
            vtMap:   (concat_NvIn) int32 vector, input to output vertex index mapping
        Output:
            output:  (concat_NvOut, out_channels) float32 array, output vertex/point features,
                                                  out_channels = in_channels*multiplier
        '''
        return f2v_Function.apply(input, filter, coeff, face, nfCount, vtMap)