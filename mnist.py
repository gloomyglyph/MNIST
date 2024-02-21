
from SimpleNNModel import SimpleNN
from SimpleCNNModel import SimpleCNN
from data_handler import DataLoader
import pandas as pd




if __name__ == "__main__":
    root = './data/'
    train_data_name = 'train.csv'
    test_data_name = 'test.csv'
    data = DataLoader(root, train_data_name, test_data_name)
    data.load_data()

    num_classes = 10

    #Simple NNModel
    #my_model = SimpleNN(num_classes, data.get_data_shape())



    # Simple CNNModel
    data.reshape_data_to_image_shape()
    data.preprocess_data()
    my_model = SimpleCNN(num_classes, data.get_data_real_shape())



    # Compile the model with an optimizer, loss function, and metrics
    my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define a custom callback to print or log metrics after each epoch

    # Train the model using the fit method
    my_model.fit(data.train_data, data.train_labels, epochs=20, batch_size=32, validation_split=0.2)

    # Print the model summary
    my_model.summary()

    #Predict test data
    test_predictions = my_model.predict(data.test_data)
    test_predictions = test_predictions.argmax(axis=-1)

    #Save Outputs
    output_labels = pd.DataFrame({'ImageId': list(range(1, len(test_predictions) + 1)),
                        'Label': test_predictions})

    output_labels.to_csv('output.csv')