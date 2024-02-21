from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D



class SimpleCNN(Model):
    def __init__(self, num_classes, input_shape):
        super(SimpleCNN, self).__init__()

        # Define the layers of your model in the constructor
        self.my_layers = []
        self.my_layers.append(Conv2D(32, kernel_size=(3, 3), activation='LeakyReLU', input_shape=input_shape))
        self.my_layers.append(MaxPooling2D(pool_size=(2, 2)))
        self.my_layers.append(Conv2D(64, kernel_size=(3, 3), activation='LeakyReLU', input_shape=input_shape))
        self.my_layers.append(MaxPooling2D(pool_size=(2, 2)))
        self.my_layers.append(Conv2D(128, kernel_size=(3, 3), activation='LeakyReLU', input_shape=input_shape))
        self.my_layers.append(MaxPooling2D(pool_size=(2, 2)))
        self.my_layers.append(Flatten())
        self.my_layers.append(Dense(128, activation='LeakyReLU'))
        self.my_layers.append(Dense(64, activation='LeakyReLU'))
        self.my_layers.append(Dense(num_classes, activation='softmax'))
        return

    def call(self, inputs):
        x = inputs
        # Define the forward pass of your model in the call method
        for my_layer in self.my_layers:
            x = my_layer(x)
        return x
