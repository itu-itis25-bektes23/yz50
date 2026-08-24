list = [0.1, 0.2, 0.3, 0.4, 0.5]
list2 = [0.15, 0.23, 0.38, 0.44, 0.59]


def visualize (list)
  for w in list: 
    pred = []
    for x in X:
      pred.append(neuron([X], [W], 1.0))
    loss = loss_mse(pred, lis2)
    print("w=", w, "loss =", loss, "*" * int(loss))
