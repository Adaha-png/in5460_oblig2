import pytablereader as ptr
import pytablewriter as ptw
from model import Model, LSTM
import tensorflow as tf
import tensorflow_federated as tff
import argparse

def main():
    file_path = "sample_data.csv"
    with open(file_path, "w") as f:
        f.write(csv_text)

    loader = ptr.CsvTableFileLoader(file_path)
    for table_data in loader.load():
        print ("\n".join([
        "load from file",
        "==============",
        "{:s}".format(ptw.dumps_tabledata(table_data)),
        ]))

# Load simulation data .
source, _ = tff.simulation.datasets.emnist.load_data()
def client_data(n):
    return source.create_tf_dataset_for_client(source.client_ids[n]).map( lambda e:(tf.reshape(e[’pixels’], [-1]), e[’label’])).repeat(10).batch(20)
# Pick a subset of client devices to participate in training .
train_data = [client_data(n) for n in range(3)]
# Wrap a Keras model for use with TFF.

def model_fn (args, training = True):
    model = Model(args, training)
    return tff.learning.from_keras_model(model, input_spec = train_data[0].element_spec, loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics =[tf.keras.metrics.SparseCategoricalAccuracy()])

args = argparse.ArgumentParser(
        model = 'rnn',
        num_layers = 3,
        rnn_size = 4,
        output_keep_prob = 1,
        input_keep_prob = 1,
        batch_size = 64,
        seq_length = 129438,
        vocab_size = 214,
        grad_clip = 1,
        )

# Simulate a few rounds of training with the selected client devices .
trainer = tff.learning.build_federated_averaging_process(model_fn(args), client_optimizer_fn = lambda: tf.keras.optimizers.SGD(0.1))
state = trainer.initialize()

for _ in range(5):
    state, metrics = trainer.next(state,train_data)
    print(metrics[’train’][’loss’])

if __name__=="__main__":
    main()
