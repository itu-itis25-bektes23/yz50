import math
def loss_log(p)
  loss = - math.log(p)
  return loss

def loss_mse(predictions, targets)
  total = 0
  for predictions, targets in zip(predictions, targets):
    diff = (predictions - targets)
    total += diff * diff
  mean = total / len(predictions)
  return mean
