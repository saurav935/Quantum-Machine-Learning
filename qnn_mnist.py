# Importing necessary packages

%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuitss

import importlib, pkg_resources
importlib.reload(pkg_resources)

import collections
import seaborn as sns
import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq


# Loading the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing the data
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0


# Function to keep only digit 3 and digit 6
def keep_only_3_and_6(a, b):
    keep = (b == 3) | (b == 6)
    a, b = a[keep], b[keep]
    b = b == 3
    return a,b
  
# Applying the above function to the dataset
x_train, y_train = keep_only_3_and_6(x_train, y_train)
x_test, y_test = keep_only_3_and_6(x_test, y_test)


# Function to resize
def resize_to_4_x_4(dataset, size):
  return tf.image.resize(dataset, size).numpy()

# Resizing the dataset
x_train_resized = resize_to_4_x_4(x_train, (4,4))
x_test_resized = resize_to_4_x_4(x_test, (4,4))


def remove_contradicting_images(xs, ys):
  # In the end "mapping" will hold the number of unique images
  mapping = collections.defaultdict(set)
  orig_x = {}

  # Establish the labels for each individual image.:
  for x,y in zip(xs,ys):
    orig_x[tuple(x.flatten())] = x
    mapping[tuple(x.flatten())].add(y)
  
  new_x = []
  new_y = []
  
  for flatten_x in mapping:
    x = orig_x[flatten_x]
    labels = mapping[flatten_x]

    if len(labels) == 1:
        new_x.append(x)
        new_y.append(next(iter(labels)))
    else:
        # Images that match multiple labels are discarded
        pass
  
  # Number of unique images of digit 3
  unique_images_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
  
  # Number of unique images of digit 6
  unique_images_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    
  return np.array(new_x), np.array(new_y)


# Removing contradictory examples

x_train_no_contradicting, y_train_no_contradicting = remove_contradicting_images(x_train_resized, y_train)

# Convert images to quantum circuit

# Binary encoding
threshold = 0.5

x_train_binary = np.array(x_train_no_contradicting > threshold, dtype=np.float32)
x_test_binary = np.array(x_test_resized > threshold, dtype=np.float32)

def convert_image_to_circuit(image):
  image_values = np.ndarray.flatten(image)
  qubits = cirq.GridQubit.rect(4, 4)
  quantum_circuit = cirq.Circuit()
  
  for i, value in enumerate(image_values):
      if value:
          quantum_circuit.append(cirq.X(qubits[i]))
  return quantum_circuit


x_train_circuit = [convert_image_to_circuit(x) for x in x_train_binary]
x_test_circuit = [convert_image_to_circuit(x) for x in x_test_binary]


# Converting data to tensors
x_train_tensors = tfq.convert_to_tensor(x_train_circuit)
x_test_tensors = tfq.convert_to_tensor(x_test_circuit)


# QUANTUM NEURAL NETWORK MODEL 

# Class to build the circuit,layer by layer
class CircuitLayerBuilder():
  def __init__(self, data_qubits, readout_qubit):
    self.data_qubits = data_qubits
    self.readout_qubit = readout_qubit
  
  def add_layer(self, circuit, gate, prefix):
    for i, qubit in enumerate(self.data_qubits):
      symbol = sympy.Symbol(prefix + '-' + str(i))
      circuit.append(gate(qubit, self.readout_qubit)**symbol)
      

      
      
# Creating sample circuit

sample_circuit = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(2,2), readout_qubit=cirq.GridQubit(-2,-1))

quantum_circuit = cirq.Circuit()

sample_circuit.add_layer(quantum_circuit, gate = cirq.XX, prefix='xx')


# Creating QNN

def create_quantum_model():
  """This function creates a QNN model circuit and the necessary operations."""
  
  # A 4x4 grid
  data_qubits = cirq.GridQubit.rect(4, 4)
  
  # A single qubit at [-1, -1]
  readout_qubit = cirq.GridQubit(-1, -1)
  quantum_circuit = cirq.Circuit()
  
  # Preparing the readout qubit
  quantum_circuit.append(cirq.X(readout_qubit))
  quantum_circuit.append(cirq.H(readout_qubit))
  builder = CircuitLayerBuilder(data_qubits = data_qubits, readout_qubit=readout_qubit)
  
  # Adding layers
  builder.add_layer(quantum_circuit, cirq.XX, "xx1")
  builder.add_layer(quantum_circuit, cirq.ZZ, "zz1")
  
  # Finally prepare the readout qubit
  quantum_circuit.append(cirq.H(readout_qubit))
  return quantum_circuit, cirq.Z(readout_qubit)

model_circuit, model_readout = create_quantum_model()


# Building the Keras model. Input to the model will be the data-circuit, encoded in a tf.string format
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(), dtype=tf.string),
                             # PQC layer returns the expected value of the readout gate, ranging between [-1,1]
                             tfq.layers.PQC(model_circuit, model_readout)])


# Steps before compile the model

# Step-1
y_train_hinge = 2.0*y_train_no_contradicting-1.0
y_test_hinge = 2.0*y_test-1.0

# Step-2
def hinge_accuracy(y_true, y_pred):
  y_true = tf.squeeze(y_true) > 0.0
  y_pred = tf.squeeze(y_pred) > 0.0
  result = tf.cast(y_true == y_pred, tf.float32)
  return tf.reduce_mean(result)

# Compile the model 
model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy])

# Model summary
print(model.summary())


# Starting the training process
qnn_history = model.fit(
      x_train_tensors, y_train_hinge,
      batch_size=32,
      epochs=10,
      verbose=1,
      validation_data=(x_test_tensors, y_test_hinge))


# Evaluation
qnn_results = model.evaluate(x_test_tensors, y_test)
accuracy = qnn_results[1] * 100

print("The accuracy of the model is: ", str(accuracy) + ' %')

