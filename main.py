import torch
import wrappers


@wrappers.taylor_extension
def f(x):
    return torch.exp(x)


@wrappers.taylor_extension(n_terms=5, loc=1)
def g(x):
    return torch.log(x)


@wrappers.taylor_extension(10)
def h(x):
    return torch.sin(x)**2 + torch.cos(x)**2


@wrappers.taylor_extension(n_terms=20)
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

    x = torch.tensor(1., dtype=torch.float64)

    # print((torch.log(x) - g(x)).norm())
    # print((torch.linalg.matrix_exp(A) - f(A)).norm())
    # print((torch.linalg.matrix_exp(B) - f(B)).norm())

    print(u(A))

    # function = h
    # n_runs = 100

    # function(x)
    # for input in [x, A, B]:

    #     exec_time = timeit(lambda: function(x), number=n_runs)

    #     print(f"Execution time for {input.shape}: {exec_time / 10:.6f} seconds (average over {n_runs} runs)")
    #     print(f"Should be the dim of the corresponding vec space: {function(input).sum().item()}")