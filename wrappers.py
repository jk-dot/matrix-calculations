import torch
from torch import func

import functools
from math import factorial



def jordan_form_func(function):
    def split_jordan_blocks(matrix):
        """
        Split a Jordan normal form matrix into its individual Jordan blocks.

        Parameters:
            matrix (torch.Tensor): The input square Jordan normal form matrix.

        Returns:
            list: A list of tensors, each representing an individual Jordan block.
        """
        n = matrix.size(0)
        blocks = []
        start_idx = 0

        for i in range(n - 1):
            if matrix[i, i + 1] != 1:  # End of a block
                # Extract the block
                blocks.append(matrix[start_idx:i + 1, start_idx:i + 1])
                start_idx = i + 1

        # Add the last block
        if start_idx < n:
            blocks.append(matrix[start_idx:n, start_idx:n])

        return blocks

    def grad(f, n=0):
        return func.vmap(functools.reduce(lambda f, _: torch.func.grad(f), range(n), f))

    def wrapper(input):

        input = input.reshape((1, 1)) if not input.dim() else input
        assert input.size(0) == input.size(1), "Input must be a square matrix"

        eigs = input.diagonal()

        output = function(eigs).diag()

        block_start_idx = 0
        for block in split_jordan_blocks(input):
            block_size = len(block)
            eig = block.diagonal()
            block_slice = slice(block_start_idx, block_start_idx+block_size)

            for i in range(1, block_size):
                output[block_slice, block_slice] += grad(function, i)(eig[:-i]).diag(diagonal=i) / factorial(i)

            block_start_idx += block_size

        return output

    return wrapper


def taylor_extension(n_terms = 100, loc = 0.):
    def compute_taylor(input, function):

        input = input.reshape((1, 1)) if not input.dim() else input
        assert input.size(0) == input.size(1), "Input must be a square matrix"

        x_loc = loc * torch.ones((1, 1), requires_grad=True, dtype=torch.float64)
        matrix_loc = x_loc * torch.eye(input.size(0), dtype=torch.float64)

        factorial = torch.tensor([1.], dtype=torch.float64)
        grad_fn = function(x_loc)
        matrix_power = torch.eye(input.size(0), dtype=torch.float64)

        output = grad_fn / factorial * matrix_power

        for term in range(1, n_terms):
            
            grad_fn = torch.autograd.grad(grad_fn.sum(), x_loc, create_graph=True)[0]
            factorial *= term
            # for log of scalar, doesn't make much sense for matrices (- matrix_loc)
            matrix_power @= input - matrix_loc

            output += grad_fn / factorial * matrix_power

        return output

    def decorator(function):
        @functools.wraps(function)
        def wrapper(input):
            return compute_taylor(input=input, function=function)
        return wrapper
    
    # Handle the case where the decorator is used without parentheses
    if callable(n_terms):
        function, n_terms = n_terms, 100
        return decorator(function)

    return decorator