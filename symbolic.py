import math
import functools


class Function:
    def __init__(self, function, string: str, gradient=None):
        self.function = function
        self.string = string
        self._gradient_func = gradient  # Internal gradient function

    def __str__(self):
        terms = [term.strip() if not '*0' in term else None for term in self.string.split('+')]

        return ' + '.join(filter(lambda term: term is not None, terms))

    def __call__(self, input):
        if isinstance(input, Function):
            return Function(
                function=lambda x: self(input(x)),  # Evaluate input(x) before applying self
                string=f"{self.string}({input.string})",
                gradient=lambda: input.gradient * Function(
                    function=lambda x: self.gradient(input(x)),  # Apply self.gradient to evaluated input
                    string=f"{self.string}'({input.string})"
                )
            )
        return self.function(input)

    def __add__(self, other):
        if isinstance(other, Function):
            return Function(
                function=lambda x: self(x) + other(x),
                string=f"{self.string} + {other.string}",
                gradient=lambda: self.gradient + other.gradient
            )

        return Function(
            function=lambda x: self(x) + other,
            string=f"({self.string} + {other})",
            gradient=lambda: self.gradient
        )

    def __sub__(self, other):
        if isinstance(other, Function):
            return Function(
                function=lambda x: self(x) - other(x),
                string=f"({self.string} - {other.string})",
                gradient=lambda: self.gradient - other.gradient
            )

        return Function(
            function=lambda x: self(x) - other,
            string=f"({self.string} - {other})",
            gradient=lambda: self.gradient
        )

    def __neg__(self):
        return Function(
            function=lambda x: -self(x),
            string=f"(-{self.string})",
            gradient=lambda: -self.gradient
        )

    def __mul__(self, other):
        if isinstance(other, Function):
            return Function(
                function=lambda x: self(x) * other(x),
                string=f"{self.string} * {other.string}",
                gradient=lambda: self.gradient * other + self * other.gradient
            )

        return Function(
            function=lambda x: self(x) * other,
            string=f"{other} * {self.string}",
            gradient=lambda: self.gradient * other
        )

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        return Function(
            function=lambda x: self(x) ** power,
            string=f"{self.string}^{power}",
            gradient=lambda: power * self ** (power - 1) * self.gradient
        )

    @property
    def gradient(self):
        if self._gradient_func is None:
            raise NotImplementedError(f"Gradient not defined for {self.string}")
        return self._gradient_func()  # Evaluate the lazy gradient


class Sin(Function):
    def __init__(self):
        super().__init__(
            function=math.sin,
            string='sin',
            gradient=lambda: Cos()
        )


class Cos(Function):
    def __init__(self):
        super().__init__(
            function=math.cos,
            string='cos',
            gradient=lambda: -Sin()
        )


class Variable(Function):
    def __init__(self, name):
        super().__init__(
            function=lambda x: x,
            string=name,
            gradient=lambda: 1
        )



def grad(function, n=1):
    return functools.reduce(lambda f, _: f.gradient, range(n), function)


sin = Sin()
cos = Cos()

x = Variable("x")

if __name__ == "__main__":

    function = 3 * sin**2 + 3*cos**2

    x_val = 3.0

    print(f"D[{str(function)}]({x_val}) = {str(grad(function))}|({x_val}) = {grad(function)(x_val):.6f}")

    print(f"D[{str(function)}, {{x, 5}}]({x_val}) = {str(grad(function, 5))}|({x_val}) = {grad(function, 5)(x_val):.6f}")
