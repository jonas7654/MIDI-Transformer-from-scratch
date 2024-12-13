# Transformer-specific layers adapted from numpy-based implementation
# by Rustam Akimov
#
# Implementation available under
# https://github.com/AkiRusProd/numpy-nn-model
#
# and nanoGPT as implemented by Andrej Karpathy
#
# Implementation available under
# https://github.com/karpathy/nanoGPT

import sys
import math
import copy
from types import NoneType
from typing import Union, Callable


import numpy as np
from numpy.typing import ArrayLike
from icecream import ic

sys.path.append('.')

from utils import compress_numpy_array, decompress_numpy_array

class Linear():
    def __init__(self,
                    in_features: int,
                    out_features: int,
                    batch_size: int,
                    lr: float = 0.1,
                    bias: bool = True,
                    weight_init_func: Union[Callable, None] = None,
                    bias_init_func: Union[Callable, None] = None) -> None:

        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.lr = lr

        self.use_bias = bias

        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        if self.weight_init_func:
            self.weight = np.asanyarray(self.weight_init_func((in_features, out_features)))
        else:
            self.weight = np.random.normal(size=(in_features, out_features))*np.sqrt(1./in_features)

        if self.bias_init_func:
            self.bias = np.asanyarray(self.bias_init_func((in_features, out_features)))
        else:
            self.weight = np.random.normal(size=(in_features, out_features))*np.sqrt(1./in_features)

        if self.use_bias:
            if self.bias_init_func:
                self.bias = np.asanyarray(self.bias_init_func((out_features,)))
            else:
                self.bias = np.random.normal(size=(out_features,))*np.sqrt(1./in_features)

        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros(out_features)

        self.input = np.zeros((batch_size, in_features))


    def _multi_dim_matmul(self,
                            mat_a: np.ndarray,
                            mat_b: np.ndarray,
                            transpose_a: bool = False,
                            transpose_b: bool = False,
                            reshape_output: bool = True) -> np.ndarray:

        """
        Replicate torch behavior of flattening all but the
        last dimension of an input of the matrix multiplication
        in linear layers. We implement this for both the first
        and the last matrix in the matrix multiplication to
        provide a unified operation for both the forward and
        the backward pass.
        """
        
        if (len(mat_a.shape) > 2) or (len(mat_b.shape) > 2):
            # Dimension handling.
            # We should refactor this if we find the time.

            dims_internal_mat_a = mat_a.shape if len(mat_a.shape) <= 2 else\
                                    (np.prod(mat_a.shape[:-1]), mat_a.shape[-1])

            dims_internal_mat_b = mat_b.shape if len(mat_b.shape) <= 2 else\
                                    (np.prod(mat_b.shape[:-1]), mat_b.shape[-1])

            mat_a_shape = mat_a.shape[::-1] if transpose_a else mat_a.shape
            mat_b_shape = mat_b.shape[::-1] if transpose_b else mat_b.shape

            dims_out_first = mat_a.shape[:-1] if reshape_output else\
                    (dims_internal_mat_a[1] if transpose_a else dims_internal_mat_a[0],)

            dims_out = (*dims_out_first, mat_b_shape[-1])

            def mat_a_transform():
                if transpose_a:
                    return mat_a.reshape(dims_internal_mat_a).T
                else:
                    return mat_a.reshape(dims_internal_mat_a)

            def mat_b_transform():
                if transpose_b:
                    return mat_b.reshape(dims_internal_mat_b).T
                else:
                    return mat_b.reshape(dims_internal_mat_b)

            return np.matmul(mat_a_transform(),
                                mat_b_transform()).reshape(dims_out)

        else:
            return np.matmul(mat_a, mat_b.T) if transpose_b else np.matmul(mat_a, mat_b)


    def forward(self, input: ArrayLike) -> np.ndarray:

        self.input = np.asanyarray(input)

        output = self._multi_dim_matmul(self.input, self.weight)
        if self.use_bias:
            output += self.bias
        return output


    def backward(self, grad_output: ArrayLike) -> np.ndarray:

        grad_output = np.asanyarray(grad_output)

        grad_input = self._multi_dim_matmul(grad_output,
                                                self.weight,
                                                transpose_b=True)

        flattened_input_shape = (np.prod(self.input.shape[:-1]), self.input.shape[-1])
        flattened_grad_output_shape = (np.prod(grad_output.shape[:-1]), grad_output.shape[-1])

        self.grad_weight = (1. /self.batch_size)*self._multi_dim_matmul(self.input,
                                                                            grad_output,
                                                                            transpose_a=True,
                                                                            reshape_output=False)

        if self.use_bias:
            self.grad_bias = (1. /self.batch_size)*grad_output.sum(0)

        return grad_input


    def update(self) -> None:
        self.weight = self.weight - self.lr*self.grad_weight
        if self.use_bias:
            self.bias = self.bias - self.lr*self.grad_bias


    @property
    def weight_transposed(self):
        return self.weight.T


    @weight_transposed.setter
    def weight_transposed(self, value):
        self.weight = value.T


class Sigmoid():
    def __init__(self, in_features: int, batch_size: int):
        super(Sigmoid, self).__init__()
        self.input = np.zeros(batch_size)


    def forward(self, input):
        input = np.asanyarray(input)
        self.input = input
        return 1./(1.+np.exp(-input))


    def backward(self, grad_output):
        grad_output = np.asanyarray(grad_output)
        grad_input = grad_output*np.exp(-self.input)/np.power(1. + np.exp(-self.input), 2)
        return grad_input


class Softmax():
    def __init__(self, axis: int=1):
        self.input = None
        self.output = None
        self.axis = axis

    def forward(self, input: ArrayLike):
        self.input = np.asanyarray(input)
        shifted_inp = input - np.max(input, axis=self.axis, keepdims=True)
        exp_res = np.exp(shifted_inp)
        output = exp_res/np.sum(exp_res, axis=self.axis, keepdims=True)
        self.output = output
        return output

    def backward(self, grad: ArrayLike):
        grad = np.asanyarray(grad)
        f_x = self.output
        grad = (grad - (grad * f_x).sum(self.axis, keepdims=True)) * f_x
        return grad


class Dropout():
    def __init__(self, p: float=0.2):

        self.p = p
        self.scale = 1/(1 - p)

        self.rng = np.random.default_rng()

        self.mask = None
        self.input = None


    def forward(self,
                    input: ArrayLike,
                    train: bool = False) -> np.ndarray:

        input = np.asanyarray(input)

        self.input = input

        if train:
            self.mask = self.rng.binomial(1, 1 - self.p, size=input.shape).astype(input.dtype)*self.scale
        else:
            self.mask = 1

        return input*self.mask

    def backward(self, grad):
        grad_out = self.mask * grad
        return grad_out

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

class LayerNorm():
    def __init__(self,
                    normalized_shape: Union[int, tuple[int]],
                    eps: float=1e-05,
                    lr: float=1e-3,
                    weight_init_func: Union[Callable, None] = None,
                    bias_init_func: Union[Callable, None] = None) -> None:

        self.normalized_shape = ((normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape)

        self.eps = eps
        self.lr = lr

        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        if self.weight_init_func:
            self.weight = np.asanyarray(self.weight_init_func((normalized_shape)))
        else:
            self.weight = np.ones((normalized_shape), dtype=np.float32)
        if self.bias_init_func:
            self.bias = np.asanyarray(self.bias_init_func((normalized_shape)))
        else:
            self.bias = np.zeros((normalized_shape), dtype=np.float32)

        self.axis = None

        self.input = None

        self.grad_weight = None
        self.grad_bias = None

        self.x_centered = None
        self.stddev_inv = None


    def forward(self, input: ArrayLike) -> np.ndarray:
        
        input = np.asanyarray(input)

        self.input = input

        self.axis = tuple(range(-len(self.normalized_shape), 0))

        mean = np.mean(input, axis=self.axis, keepdims=True)
        var = np.var(input, axis=self.axis, keepdims=True)

        self.x_centered = input - mean
        self.stddev_inv = 1/np.sqrt(var + self.eps)

        output = self.x_centered*self.stddev_inv

        return self.weight*output + self.bias


    def backward(self, doutput: ArrayLike) -> np.ndarray:
        """
        Perform backpropagation to compute gradients of input, weight, and bias.
        """
        doutput = np.asanyarray(doutput)
        output_forward = self.weight * self.x_centered*self.stddev_inv
        # Calculate gradient of bias and weight
        self.grad_bias = np.sum(doutput, axis=(0,1)) 
        self.grad_weight = np.sum(doutput * output_forward, axis=(0,1))
        # Calculate gradient of the normalized output (dhat_x)
        dx_hat = doutput * self.weight  # Backpropagate through the scaling
        dvar = np.sum(dx_hat * self.x_centered, axis=self.axis, keepdims=True) * (-0.5) * (self.stddev_inv ** 3)
        dmean = np.sum(dx_hat * -self.stddev_inv, axis=self.axis, keepdims=True) + dvar * np.mean(-2.0 * self.x_centered, axis=self.axis, keepdims=True)

        # Calculate gradient of input
        dinput = dx_hat * self.stddev_inv + (dvar * 2.0 * self.x_centered) / self.input.size + dmean / self.input.size
    
        return dinput


    def update(self):
        self.weight -= self.lr * self.grad_weight
        self.bias -= self.lr * self.grad_bias
        return None


class GELU():
    def __init__(self) -> None:
        self._sqrt_of_2_by_pi = np.sqrt(2/np.pi)
        self.input = None


    def forward(self, input: ArrayLike) -> np.ndarray:
        self.input = np.asanyarray(input)
        return (0.5*input*(1 + np.tanh(self._sqrt_of_2_by_pi*(input + 0.044715*np.power(input, 3)))))

    def gelu_grad(self, x):
        sqrt_2_pi = np.sqrt(2 / np.pi)
        a = sqrt_2_pi * (x + 0.044715 * x**3)
        tanh_a = np.tanh(a)
        b = sqrt_2_pi * (1 + 3 * 0.044715 * x**2)
    
        grad = 0.5 * (1 + tanh_a + x * (1 - tanh_a**2) * b)
        return grad

    def backward(self, grad_output: ArrayLike) -> np.ndarray:
        grad_output = np.asarray(grad_output)
        input_grad = self.gelu_grad(self.input)
        return grad_output * input_grad


class MLP():
    def __init__(self,
                    d_model,
                    batch_size,
                    lr,
                    dropout,
                    c_fc_init_func,
                    c_proj_init_func,
                    bias_init_func):

        self.d_model = d_model
        self.batch_size = batch_size
        self.lr = lr

        self.c_fc = Linear(d_model,
                                4*d_model,
                                batch_size,
                                lr,
                                weight_init_func=c_fc_init_func,
                                bias_init_func=bias_init_func)
        self.gelu = GELU()

        self.c_proj = Linear(4*d_model,
                                d_model,
                                batch_size,
                                lr,
                                weight_init_func=c_proj_init_func,
                                bias_init_func=bias_init_func)

        self.dropout = Dropout(dropout)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.c_fc.forward(x)
        x = self.gelu.forward(x)
        x = self.c_proj.forward(x)
        x = self.dropout.forward(x)
        return x

    def backward(self, x: np.ndarray) -> np.ndarray:
        x = self.dropout.backward(x)
        x = self.c_proj.backward(x)
        x = self.gelu.backward(x)
        x = self.c_fc.backward(x)
        return x

    def update(self) -> None:
        # TODO
        #self.dropout.update()
        self.c_proj.update()
        self.c_fc.update()

    def get_params(self) -> dict:
        return {'c_fc': [compress_numpy_array(self.c_fc.weight), compress_numpy_array(self.c_fc.bias)],\
                'c_proj': [compress_numpy_array(self.c_proj.weight), compress_numpy_array(self.c_proj.bias)]}


    def load_params(self, state_dict: dict) -> None:
        self.c_fc.weight = decompress_numpy_array(state_dict['c_fc'][0])
        self.c_fc.bias = decompress_numpy_array(state_dict['c_fc'][1])
        self.c_proj.weight = decompress_numpy_array(state_dict['c_proj'][0])
        self.c_proj.bias = decompress_numpy_array(state_dict['c_proj'][1])


class MultiHeadAttention():
    def __init__(self, d_model: int,
                        context_size: int,
                        n_heads: int,
                        batch_size: int,
                        lr: float=0.1,
                        dropout: float=0.1,
                        c_attn_weight_init_func: Union[Callable, None] = None,
                        c_proj_weight_init_func: Union[Callable, None] = None,
                        bias_init_func: Union[Callable, None] = None) -> None:

        self.d_model = d_model
        self.context_size = context_size
        self.n_heads = n_heads
        self.scale = math.sqrt(d_model)
        self.batch_size = batch_size

        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)
        self.softmax_attn = Softmax(axis=-1)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.depth = d_model//n_heads

        self.c_attn = Linear(d_model,
                                3*d_model,
                                batch_size,
                                lr,
                                weight_init_func=c_attn_weight_init_func,
                                bias_init_func=bias_init_func)

        self.c_proj = Linear(d_model,
                                d_model,
                                batch_size,
                                lr,
                                weight_init_func=c_proj_weight_init_func,
                                bias_init_func=bias_init_func)

        self.mask = np.tril(np.ones((context_size, context_size), dtype=np.float32)).reshape(1, 1, context_size, context_size)

        self.input = None
        self.v = None
        self.q = None
        self.k = None
        self.attn = None


    def forward(self, input: ArrayLike) -> tuple:
        
        self.input = np.asanyarray(input)
        # C = d_model : default n_embd: int=384
        # B = batch dim
        # T = seq length
        B, T, C = self.input.shape

        """
        c_attn has shape d_model * (3 * d_model)
        when splitting at axis = 2 (the seq length axis) we split it into three equal parts with the correct d_model dimension
        for each q, k, v
        
        
        Note that q, k, v are calculated in one run.
        This is why the c_attn linear object has dimensions (d_model * (3 * d_model))
        """
        q, k, v  = np.split(self.c_attn.forward(self.input), 3, axis=2)

        """
        the output of self.c_attn.forward(self.input) has shape (B, seq_len, 3 * d_model)
        
        
        C = d_model
        B = batch dim
        T = seq length
        
        The original shape of k (before transpose) is (B, T, n_heads, C // n_heads)
        
        After transposing, the shape becomes (B, n_heads, T, C // n_heads)
        
        """
        
        k = k.reshape((B, T, self.n_heads, C//self.n_heads)).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        q = q.reshape((B, T, self.n_heads, C//self.n_heads)).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        v = v.reshape((B, T, self.n_heads, C//self.n_heads)).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        self.k = k
        self.q = q
        self.v = v

        """
        q dim = (B, n_heads, T, C // n_heads)
        k.tranpose dim = (B, n_heads, C // n_heads, T)
        
        we want to multiply along the sequence axis (T) and (C // n_heads) axis
        
        Note that k.shape[-1] = C // n_heads
        
        And the resulting shape of attn is (B, n_heads, T, T) [Dot product between tokens]
        """
        
        attn = (q @ k.transpose(0, 1, 3, 2))*(1.0/math.sqrt(k.shape[-1]))

        attn = np.where(self.mask == 0, -1e9, attn)
        attn = self.softmax_attn.forward(attn)
        attn = self.attn_dropout.forward(attn)

        self.attn = attn
        """
        Now x has shape (B, n_heads, T, C // n_heads)
        """
        x = attn @ v
        
        """
        We reshape x to the original Input shape which is (B, T, C) => concat heads
        -1 tells python to calculate the remaining dimension to fit (i.e returning T in this case)
        
        I don't know why they wrote: self.n_heads*self.depth 
        This is just a non readible way of using C from the beginning
        """
        x = np.ascontiguousarray(x).transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.n_heads*self.depth)
        x = self.c_proj.forward(x)
        x = self.resid_dropout.forward(x)

        return x, attn

    def backward(self, grad: ArrayLike) -> np.ndarray:
        grad = np.asanyarray(grad)

        B, T, C = self.input.shape

        grad = self.resid_dropout.backward(grad)
        grad = self.c_proj.backward(grad)

        #grad = np.ascontiguousarray(grad)
        # OLD
        #grad = grad.reshape((B, T, self.n_heads, -1))
        #grad = grad.transpose(0,2,1,3) 

        # NEW
        grad = np.ascontiguousarray(grad).reshape(B, self.n_heads, T, -1) # B, n_heads, T, C // n_heads
        
        grad_v = self.attn.transpose(0,1,3,2) @ grad
        grad = grad @ self.v.transpose(0,1,3,2)
        

        grad = self.attn_dropout.backward(grad)
        grad = self.softmax_attn.backward(grad)

        grad = np.where(self.mask == 0, -1e9, grad)

        grad_ktrans = ((1.0/math.sqrt(self.k.shape[-1])) * (self.q.transpose(0, 1, 3, 2) @ grad))
        grad_k = grad_ktrans.transpose(0, 1, 3, 2)
        grad_q = (1.0/math.sqrt(self.k.shape[-1]))  * (grad @ self.k)

        grad_q = grad_q.reshape(self.batch_size, T, self.n_heads * self.depth) 
        grad_v = grad_v.reshape(self.batch_size, T, self.n_heads * self.depth)
        grad_k = grad_k.reshape(self.batch_size, T, self.n_heads * self.depth)
        
        # We need to tranpose to the shape (batch_size, seq_len, 3* d_model)
        grad = np.concatenate([grad_q, grad_k, grad_v], axis=2)

        grad_downstream = self.c_attn.backward(grad)

        return grad_downstream


    def update(self) -> None:
        #raise NotImplementedError("Implement the MultiHeadAttention update path")

        # Jonas Version

        self.c_attn.update()
        self.c_proj.update()
    


    def get_params(self) -> dict:
        return {'c_attn': [compress_numpy_array(self.c_attn.weight), compress_numpy_array(self.c_attn.bias)],
                    'c_proj': [compress_numpy_array(self.c_proj.weight), compress_numpy_array(self.c_proj.bias)]}


    def load_params(self, state_dict: dict) -> None:
        self.c_attn.weight = decompress_numpy_array(state_dict['c_attn'][0])
        self.c_attn.bias = decompress_numpy_array(state_dict['c_attn'][1])
        self.c_proj.weight = decompress_numpy_array(state_dict['c_proj'][0])
        self.c_proj.bias = decompress_numpy_array(state_dict['c_proj'][1])

class Embedding():
    def __init__(self, num_embeddings: int,
                        embedding_dim: int,
                        batch_size: int,
                        lr: float,
                        init_func: Union[Callable, None] = None,
                        weight_external = None):

        self.rng = np.random.default_rng()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.batch_size = batch_size
        self.lr = lr

        self.init_func = init_func

        # If we get external weights passed, use them
        # instead of allocating ones on our own.
        # This is used for implementing weight tying.
        #
        # https://paperswithcode.com/method/weight-tying

        if not isinstance(type(weight_external), NoneType):
            if self.init_func:
                self.weight = np.asanyarray(self.init_func((num_embeddings, embedding_dim)))
            else:
                self.weight = self.rng.standard_normal((num_embeddings, embedding_dim), dtype=np.float32)

        else:
            self.weight = weight_external

        self.gradient_projection_mask = np.eye(num_embeddings, dtype=np.uint8)

        self.input = None
        self.grad_weight = None


    def forward(self, input: ArrayLike) -> np.ndarray:
        self.input = np.asanyarray(input) # (Batch size, seq len)
        return self.weight[self.input.astype(np.int32), :]


    def backward(self, grad_output: np.ndarray):
        """
        Backward pass to compute gradients for the embedding weights.

        Args:
        grad_output (np.ndarray): Gradient of the loss w.r.t. the output embeddings
                                  (shape: batch_size x seq_length x embedding_dim).
        """
        # Flatten input indices and corresponding gradients
        #input_flat = self.input.ravel()  # Flattened input indices (batch_size * seq_length,)
        #grad_output_flat = grad_output.reshape(-1, self.embedding_dim)  # Flattened gradients (batch_size * seq_length, embedding_dim)

        # Initialize grad_weight
        
        #self.grad_weight = np.zeros_like(self.weight)

        # Accumulate gradients for the corresponding rows
        #grad = np.add.at(self.grad_weight, input_flat, grad_output_flat)A
        
        
        # Jonas Version
        # Ensure `self.input` has shape (batch_size, seq_length)
        batch_size, seq_length = self.input.shape

        # Flatten input indices (batch_size * seq_length,)
        input_flat = self.input.ravel()

        # Flatten the gradients (batch_size * seq_length, embedding_dim)
        grad_output_flat = grad_output.reshape(-1, self.embedding_dim)

        # Initialize grad_weight to zero (same shape as weight matrix)
        self.grad_weight = np.zeros_like(self.weight)

        # Accumulate gradients for the corresponding rows
        np.add.at(self.grad_weight, input_flat, grad_output_flat)
        # Return None since there's no gradient with respect to the input indices



    def update(self) -> None:
        self.weight -= self.lr *  self.grad_weight
        return None


class Block():

    def __init__(self,
                    d_model: int,
                    context_size: int,
                    n_heads: int,
                    batch_size: int,
                    lr: float,
                    dropout: float,
                    weight_init_func: Union[Callable, None],
                    c_proj_init_func: Union[Callable, None],
                    bias_init_func: Union[Callable, None]) -> None:

        self.d_model = d_model
        self.context_size = context_size
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout

        self.ln_1 = LayerNorm(d_model,
                                weight_init_func=weight_init_func)

        self.attn = MultiHeadAttention(d_model,
                                            context_size,
                                            n_heads,
                                            batch_size,
                                            lr,
                                            dropout,
                                            c_attn_weight_init_func=weight_init_func,
                                            c_proj_weight_init_func=c_proj_init_func,
                                            bias_init_func=bias_init_func)

        self.ln_2 = LayerNorm(d_model,
                                weight_init_func=weight_init_func)

        self.mlp = MLP(d_model,
                            batch_size,
                            lr,
                            dropout,
                            c_fc_init_func=weight_init_func,
                            c_proj_init_func=c_proj_init_func,
                            bias_init_func=bias_init_func)


    def forward(self, input: ArrayLike) -> np.ndarray:

        input = np.asanyarray(input)

        x = self.ln_1.forward(input)
        x = self.attn.forward(x)[0]

        x = input + x

        residual = copy.deepcopy(x)

        x = self.ln_2.forward(x)
        x = self.mlp.forward(x)
        x = residual + x

        return x


    def backward(self, grad_output: ArrayLike) -> np.ndarray:
        grad_output = np.asanyarray(grad_output)

        grad_resid1 = copy.deepcopy(grad_output)

        grad = self.mlp.backward(grad_output)
        grad = self.ln_2.backward(grad)

        grad = grad + grad_resid1

        grad_resid2 = copy.deepcopy(grad)

        grad = self.attn.backward(grad)[0]
        grad = self.ln_1.backward(grad)

        grad = grad + grad_resid2
        
        return grad


    def update(self) -> None:
        #raise NotImplementedError("Implement the Block update")
    
    
        # Jonas Version
        self.mlp.update()
        self.ln_2.update()
        self.attn.update()
        self.ln_1.update()


    def state_dict(self) -> dict:
        return {'ln_1': [compress_numpy_array(self.ln_1.weight), compress_numpy_array(self.ln_1.bias)],
                            'ln_2': [compress_numpy_array(self.ln_2.weight), compress_numpy_array(self.ln_2.bias)],
                            'mlp': self.mlp.get_params(),
                            'attn': self.attn.get_params()}


    def load_params(self, state_dict: dict) -> None:
        self.ln_1.weight = decompress_numpy_array(state_dict['ln_1'][0])
        self.ln_1.bias = decompress_numpy_array(state_dict['ln_1'][1])
        self.ln_2.weight = decompress_numpy_array(state_dict['ln_2'][0])
        self.ln_2.bias = decompress_numpy_array(state_dict['ln_2'][1])

        self.mlp.load_params(state_dict['mlp'])
        self.attn.load_params(state_dict['attn'])
