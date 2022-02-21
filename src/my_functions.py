from tensorflow.keras import layers
#from tensorflow.keras import Model

def freezing_layers(model, block: str):
  '''
  for transfer-learning
  it freezes all layers until the selected block
  '''
  for layer in model.layers:
    if layer.name == block:
      break

    layer.trainable = False
    print("layer " + layer.name + " frozen")