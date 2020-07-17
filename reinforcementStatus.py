import time


class ReinforcementStatus:
    def __init__(self, initValue, changeValue, MAXValue, valueID):
        self.value = initValue
        self.changeValue = changeValue
        self.MAXValue = MAXValue
        self.valueID = valueID
        self.action_space = ['Add', 'Sub']
        self.n_actions = len(self.action_space)

    def step(self, action):
        if self.valueID == 0:
            if action == 0:  # ADD
                self.value += self.changeValue
            if action == 1:  # Sub
                self.value -= self.changeValue
                if self.value <= 0:
                    self.value += self.changeValue
            return self.value
        else:
            if action == 0:  # ADD
                self.value += self.changeValue
                if self.value >= self.MAXValue:
                    self.value -= self.changeValue
            if action == 1:  # Sub
                self.value -= self.changeValue
                if self.value <= 0:
                    self.value += self.changeValue
            return self.value

    def render(self):
        time.sleep(0.0001)
