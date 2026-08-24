outputs []
def forward_pass(inputs, weights, biases)
  for range i in range(len(weights))
    final = neuron (inputs, weights, biases)
    outputs.append(final) //Listenin sonuna ekle.
return  outputs
