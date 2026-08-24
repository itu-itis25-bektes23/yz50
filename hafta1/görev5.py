list1 = [0.6, 0.76, 0.83, 1.2, 1.34]
list2 = [0.5, 0.7, 0.9, 1.15, 1,46]
def loss_calc (w, b):
  pred = []
  for x in X:
    pred.append(x * w + b)
  return loss_mse (pred, list2)

def gradient_descent(w, b, lr, steps):
  h = 0.000001
  for i in range(steps):
    slope_w = (loss_calc(w + h, b) - loss_calc(w, b)) / h
    slope_b = (loss_calc(w, b + h) - loss_calc(w, b)) / h
    print("Step: ", i, "loss=", loss_mse(w, b))
    w = w -lr * slope_w
    b = b -lr * slope_b
  return w, b
