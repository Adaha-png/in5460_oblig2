from model import SimpleRNN, LSTM
import tensorflow as tf
import tensorflow_federated as tff
import argparse
import pandas as pd
import numpy as np

def main():
    file_path = "Dataset for assignment 2.xlsx"
    df = pd.read_excel(file_path)
    total = df.sum(axis='columns').to_numpy()

    print(total)
    rnn_model = SimpleRNN()
    rnn_model.build_default((7*96,1), 96, activation = None)

    lstm_model = LSTM()
    lstm_model.build_default((7*96,1), 96, activation = None)

    trainer = tff.learning.algorithms.build_unweighted_fed_avg(model_fn(rnn_model), client_optimizer_fn = lambda: tf.keras.optimizers.SGD(0.1))
    state = trainer.initialize()

    for _ in range(300):
        state, metrics = trainer.next(state,train_data)
        print(metrics['train']['loss'])
    
def model_fn (model, training = True):
    return tff.learning.from_keras_model(model, input_spec = train_data[0].element_spec, loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics =[tf.keras.metrics.SparseCategoricalAccuracy()])

# Simulate a few rounds of training with the selected client devices .

if __name__=="__main__":
    main()
