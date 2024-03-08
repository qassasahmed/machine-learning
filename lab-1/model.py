import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, x, y):
        self.beta_note = None
        self.beta_one = None
        self.x_input = x
        self.x_square = [i ** 2 for i in x]
        self.y_observed = y
        self.x_bar = np.mean(x)
        self.y_bar = np.mean(y)
        self.x_y = [i * j for i, j in zip(x, y)]
        self.n = len(x)

    def linear_regression(self):
        self.beta_one = (sum(self.x_y) - self.n * self.x_bar * self.y_bar) / (sum(
            self.x_square) - self.n * self.x_bar ** 2)
        self.beta_note = self.y_bar - self.beta_one * self.x_bar
        return {"beta_note": self.beta_note, "beta_one": self.beta_one}


    def plot_line(self):
        intercept = self.linear_regression()["beta_note"]
        slope = self.linear_regression()["beta_one"]
        x = np.linspace(1, 16, 100)
        y = [slope * i + intercept for i in x]
        plt.scatter(self.x_input, self.y_observed, color="orange", label="Original date")
        plt.plot(x, y, color="green", label="Fitting line")
        plt.legend(loc='upper center')
        plt.show()




