class Perceptron:

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward_pass(self, input_matrix):
        return input_matrix.dot(self.w) + self.b > 0

    def train_on_simple_example(self, example, y):
        predict = (self.w.T.dot(example) + self.b) > 0
        error = y - predict
        delta_w = error * example
        self.w += delta_w
        self.b += error

        return error

