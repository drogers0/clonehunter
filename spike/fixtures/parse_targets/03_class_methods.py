class Calculator:
    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y

    def reset(self):
        """Reset the calculator state."""
        self._state = 0


class InheritedCalc(Calculator):
    def multiply(self, x, y):
        return x * y

    def power(self, base, exp):
        """Raise base to exp."""
        return base**exp
