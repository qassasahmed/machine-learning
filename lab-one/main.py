import model
x = [1, 0, 2]
y = [0, 1, 2]


m = model.Model(x, y)
print(m.linear_regression())
m.plot_line()
