class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        def derive(f):
            return 2 * f

        for _ in range(iterations):
            res = init - (learning_rate * derive(init))
            init = res
        return round(init, 5)
