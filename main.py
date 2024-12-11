import torch
import wrappers


@wrappers.taylor_extension(200)
def f(x):
    return torch.exp(x)


@wrappers.taylor_extension(n_terms=5, loc=1)
def g(x):
    return torch.log(x)


@wrappers.taylor_extension(10)
def h(x):
    return torch.sin(x)**2 + torch.cos(x)**2


'''
exhibits the initial worsing of the approximation
thus needs to be run with a high number of terms,
however, the computation of gradient of a high number
of terms takes way too much time and memory for
composite function like the one below
'''
@wrappers.taylor_extension(n_terms=10)
def u(x):
    return torch.cos(x) * torch.cos(x)


from timeit import timeit


if __name__ == "__main__":

    A = torch.tensor([
        [10, 1, 0],
        [0, -10, 1],
        [0, 0, 10]
    ], dtype=torch.float64)

    B = torch.rand((50, 50), dtype=torch.float64)

    x = torch.tensor([[1.]], dtype=torch.float64)

    # print((torch.log(x) - g(x)).norm())
    # print((torch.linalg.matrix_exp(A) - f(A)).norm())
    # print((torch.linalg.matrix_exp(B) - f(B)).norm())

    # print(u(A))

    function = f
    n_runs = 100

    function(x)
    for input in [x, A, B]:

        exec_time = timeit(lambda: function(input), number=n_runs)

        print(f"Execution time for {input.shape}: {exec_time / 10:.6f} seconds (average over {n_runs} runs)")
        print(f"accuracy: {(function(input) - torch.linalg.matrix_exp(input)).norm().item()}")
        # print(f"Should be the dim of the corresponding vec space: {function(input).sum().item()}")