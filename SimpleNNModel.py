from keras.models import Model
from keras.layers import Input, Dense




class SimpleNN(Model):
    def __init__(self, num_classes, input_shape):
        super(SimpleNN, self).__init__()

        # Define the layers of your model in the constructor
        self.my_layers = []
        self.my_layers.append(Dense(250, activation='LeakyReLU'))
        self.my_layers.append(Dense(125, activation='LeakyReLU'))
        self.my_layers.append(Dense(64, activation='LeakyReLU'))
        self.my_layers.append(Dense(num_classes, activation='softmax'))
        return

    def call(self, inputs):
        x = inputs
        # Define the forward pass of your model in the call method
        for my_layer in self.my_layers:
            x = my_layer(x)
        return x
