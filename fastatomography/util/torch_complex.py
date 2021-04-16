import torch as th
import numpy as np

re = np.s_[..., 0]
im = np.s_[..., 1]


def cangle(x: th.Tensor, deg=False) -> th.Tensor:
    real = th.atan2(x[im], x[re])
    if deg:
        real *= 180 / np.pi
    return real


def complex_numpy(x: th.Tensor) -> np.array:
    a = x.detach().numpy()
    return a[re] + 1j * a[im]


def cx_from_numpy(x: np.array) -> th.Tensor:
    if 'complex' in str(x.dtype):
        out = th.zeros(x.shape + (2,))
        out[re] = th.from_numpy(x.real)
        out[im] = th.from_numpy(x.imag)
    else:
        if x.shape[-1] != 2:
            out = th.zeros(x.shape + (2,))
            out[re] = th.from_numpy(x.real)
        else:
            out = th.zeros(x.shape + (2,))
            out[re] = th.from_numpy(x[re])
            out[re] = th.from_numpy(x[im])
    return out


def make_real(x: th.Tensor) -> th.Tensor:
    out_shape = x.shape + (2,)
    out = th.zeros(out_shape)
    out[re] = x
    return out


def make_imag(x: th.Tensor) -> th.Tensor:
    out_shape = x.shape + (2,)
    out = th.zeros(out_shape)
    out[im] = x
    return out


def complex_polar(r: th.Tensor, angle: th.Tensor) -> th.Tensor:
    real = r * th.cos(angle)
    imag = r * th.sin(angle)
    return th.stack([real, imag], -1)


def complex_expi(x: th.Tensor) -> th.Tensor:
    real = th.cos(x)
    imag = th.sin(x)
    return th.stack([real, imag], -1)


def complex_exp(x: th.Tensor) -> th.Tensor:
    if x.shape[-1] != 2:
        raise RuntimeWarning('taking exp of non-complex tensor!')
    real = th.exp(x[re]) * th.cos(x[im])
    imag = th.exp(x[re]) * th.sin(x[im])
    return th.stack([real, imag], -1)


def complex_mul(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2 or b.shape[-1] != 2:
        raise RuntimeWarning(
            'taking complex_mul of non-complex tensor! a.shape ' + str(a.shape) + 'b.shape ' + str(b.shape))
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = b[im]
    real = are * bre - aim * bim
    imag = are * bim + aim * bre
    return th.stack([real, imag], -1)


def complex_mul_conj(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2 or b.shape[-1] != 2:
        raise RuntimeWarning(
            'taking complex_mul of non-complex tensor! a.shape ' + str(a.shape) + 'b.shape ' + str(b.shape))
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = -b[im]
    real = are * bre - aim * bim
    imag = are * bim + aim * bre
    return th.stack([real, imag], -1)


def complex_mul_real(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning(
            'taking complex_mul of non-complex tensor! a.shape ' + str(a.shape) + 'b.shape ' + str(b.shape))
    are = a[re]
    aim = a[im]
    return th.stack([are * b, aim * b], -1)


def complex_div(complex_tensor1, complex_tensor2):
    '''Compute element-wise division between complex tensors'''
    denominator = (complex_tensor2 ** 2).sum(-1)
    complex_tensor_mul_real = (complex_tensor1[..., 0] * complex_tensor2[..., 0] + complex_tensor1[..., 1] *
                               complex_tensor2[..., 1]) / denominator
    complex_tensor_mul_imag = (complex_tensor1[..., 1] * complex_tensor2[..., 0] - complex_tensor1[..., 0] *
                               complex_tensor2[..., 1]) / denominator
    return th.stack((complex_tensor_mul_real, complex_tensor_mul_imag), dim=-1)


def make_real(x: th.Tensor) -> th.Tensor:
    out_shape = x.shape + (2,)
    out = th.zeros(out_shape, device=x.device)
    out[re] = x
    return out


def complex_expi(x: th.Tensor) -> th.Tensor:
    real = th.cos(x)
    imag = th.sin(x)
    return th.stack([real, imag], -1)


def complex_mul_real(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning(
            'taking complex_mul of non-complex tensor! a.shape ' + str(a.shape) + 'b.shape ' + str(b.shape))
    are = a[re]
    aim = a[im]
    return th.stack([are * b, aim * b], -1)


def complex_numpy(x: th.Tensor) -> np.array:
    a = x.detach().numpy()
    return a[re] + 1j * a[im]


def complex_mul(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2 or b.shape[-1] != 2:
        raise RuntimeWarning(
            'taking complex_mul of non-complex tensor! a.shape ' + str(a.shape) + 'b.shape ' + str(b.shape))
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = b[im]
    real = are * bre - aim * bim
    imag = are * bim + aim * bre
    return th.stack([real, imag], -1)


def conj(a: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning('taking conj of non-complex tensor!')
    real = a[re] * 1
    imag = -1 * a[im]
    return th.stack([real, imag], -1)


class ComplexMul(th.autograd.Function):
    '''Complex multiplication class for autograd'''

    @staticmethod
    def forward(ctx, input1, input2):
        assert input1.shape[-1] == 2, "Complex tensor should have real and imaginary parts."
        assert input2.shape[-1] == 2, "Complex tensor should have real and imaginary parts."
        output = complex_mul(input1, input2)

        ctx.save_for_backward(input1, input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = complex_mul(conj(input2), grad_output)
        grad_input2 = complex_mul(conj(input1), grad_output)
        if len(input1.shape) > len(input2.shape):
            grad_input2 = grad_input2.sum(0)
        elif len(input1.shape) < len(input2.shape):
            grad_input1 = grad_input1.sum(0)

        return grad_input1, grad_input2


class ComplexAbs(th.autograd.Function):
    '''Absolute value class for autograd'''

    @staticmethod
    def forward(ctx, input):
        assert input.shape[-1] == 2, "Complex tensor should have real and imaginary parts."
        output = ((input ** 2).sum(-1)) ** 0.5

        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = th.stack((grad_output, th.zeros_like(grad_output)), dim=len(grad_output.shape))
        phase_input = cangle(input)
        phase_input = th.stack((th.cos(phase_input), th.sin(phase_input)), dim=len(grad_output.shape))
        grad_input = complex_mul(phase_input, grad_input)

        return 0.5 * grad_input


class ComplexAbs2(th.autograd.Function):
    '''Absolute value squared class for autograd'''

    @staticmethod
    def forward(ctx, input):
        assert input.shape[-1] == 2, "Complex tensor should have real and imaginary parts."
        output = complex_mul(conj(input), input)

        ctx.save_for_backward(input)
        return output[..., 0]

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output_c = th.stack((grad_output, th.zeros_like(grad_output)), dim=len(grad_output.shape))
        grad_input = complex_mul(input, grad_output_c)

        return grad_input


class ComplexExp(th.autograd.Function):
    '''Complex exponential class for autograd'''

    @staticmethod
    def forward(ctx, input):
        assert input.shape[-1] == 2, "Complex tensor should have real and imaginary parts."
        output = input.clone()
        amplitude = th.exp(input[..., 0])
        output[..., 0] = amplitude * th.cos(input[..., 1])
        output[..., 1] = amplitude * th.sin(input[..., 1])

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = complex_mul(conj(output), grad_output)

        return grad_input


cexp = ComplexExp.apply
cabs = ComplexAbs.apply
cabs2 = ComplexAbs2.apply
cmul = ComplexMul.apply
