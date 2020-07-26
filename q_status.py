import time


class QStatus:
    def __init__(self, value, delta, max, type):
        self.value = value
        self.delta = delta
        self.max = max
        self.type = type
        self.action_space = ['Add', 'Sub']
        self.n_actions = len(self.action_space)

    def step(self, action, is_change = True):
        # [0, 1] -> [-1, 1]
        sign = action * 2 - 1
        delta = sign * self.delta

        if self.type == 0:
            # value belongs to [1, inf]
            value = self.value + delta
        else:
            # value belongs to [1, max]
            value = min(self.value + delta, self.max)

        if value <= 0:
            value = self.value

        if is_change:
            self.value = value
        return value

    def render(self):
        # time.sleep(0.0001)
        pass
