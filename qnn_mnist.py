#!/usr/bin/env python
# coding: utf-8

# ### Installing the necessary packages

# In[31]:


get_ipython().system('pip install --quiet tensorflow==2.7.0')


# In[32]:


get_ipython().system('pip install --quiet tensorflow-quantum==0.7.2')


# In[33]:


get_ipython().system('pip install --quiet cirq')


# ### Importing the necessary packages

# In[34]:


import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections

# visualization tools
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


# # **Data preprocessing**

# In[35]:


# Loading the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[36]:


# Normalizing the data
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0


# ### Function to keep only digit 3 and digit 6

# In[37]:


def keep_only_3_and_6(a, b):
    keep = (b == 3) | (b == 6)
    a, b = a[keep], b[keep]
    b = b == 3
    return a,b


# In[38]:


x_train, y_train = keep_only_3_and_6(x_train, y_train)
x_test, y_test = keep_only_3_and_6(x_test, y_test)


# ### Seeing an example

# In[39]:


plt.imshow(x_train[25, :, :, 0])


# ### Function to resize

# In[40]:


def resize_to_4_x_4(dataset, size):
    return tf.image.resize(dataset, size).numpy()


# In[41]:


x_train_resized = resize_to_4_x_4(x_train, (4,4))
x_test_resized = resize_to_4_x_4(x_test, (4,4))


# In[42]:


print(y_train[0])

plt.imshow(x_train_resized[25,:,:,0], vmin=0, vmax=1)
plt.savefig('sixx.png')


# ### Remove conradictory examples

# In[43]:


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


# In[44]:


x_train_no_contradicting, y_train_no_contradicting = remove_contradicting_images(x_train_resized, y_train)


# # **Converting images to quantum circuits**

# ### Applying threshold

# In[45]:


# As this is a binary classifier, thereshold is 0.5
threshold = 0.5

x_train_binary = np.array(x_train_no_contradicting > threshold, dtype=np.float32)
x_test_binary = np.array(x_test_resized > threshold, dtype=np.float32)


# In[46]:


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


# ### Training data's quantum circuit

# In[47]:


print("Training data:")
SVGCircuit(x_train_circuit[1])


# ### Testing data's quantum circuit

# In[48]:


print("Test data:")
SVGCircuit(x_test_circuit[1])


# ### Convert to tensors

# In[49]:


x_train_tensors = tfq.convert_to_tensor(x_train_circuit)
x_test_tensors = tfq.convert_to_tensor(x_test_circuit)


# # **Quantum Neural Network Model**

# In[50]:


class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout_qubit):
        self.data_qubits = data_qubits
        self.readout_qubit = readout_qubit

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout_qubit)**symbol)


# ### Create a sample circuit

# In[51]:


sample_circuit = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(2,2),
                                   readout_qubit=cirq.GridQubit(-2,-1))

quantum_circuit = cirq.Circuit()
sample_circuit.add_layer(quantum_circuit, gate = cirq.XX, prefix='xx')
SVGCircuit(quantum_circuit)


# ### Create circuit for QNN

# In[52]:


def create_quantum_model():
    data_qubits = cirq.GridQubit.rect(4, 4)        # a 4x4 grid.
    readout_qubit = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    quantum_circuit = cirq.Circuit()

    # Preparing the readout qubit.
    quantum_circuit.append(cirq.X(readout_qubit))
    quantum_circuit.append(cirq.H(readout_qubit))

    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout_qubit=readout_qubit)

    # Adding layers
    builder.add_layer(quantum_circuit, cirq.XX, "xx1")
    builder.add_layer(quantum_circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    quantum_circuit.append(cirq.H(readout_qubit))

    return quantum_circuit, cirq.Z(readout_qubit)


# In[53]:


model_circuit, model_readout = create_quantum_model()


# In[54]:


SVGCircuit(model_circuit)


# ### Build the Keras model

# In[55]:


model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(model_circuit, model_readout),
])


# In[56]:


model.summary()


# ### Converting from True/False to [-1, 1]

# In[57]:


y_train_hinge = 2.0*y_train_no_contradicting-1.0
y_test_hinge = 2.0*y_test-1.0


# ### Hinge accuracy

# In[58]:


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


# ### Compiling the model

# In[59]:


model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy])


# # **Training the model**

# In[60]:


qnn_history = model.fit(
      x_train_tensors, y_train_hinge,
      batch_size=32,
      epochs=10,
      verbose=1,
      validation_data=(x_test_tensors, y_test_hinge))


# ### Evaluating the model

# In[61]:


qnn_results = model.evaluate(x_test_tensors, y_test)

print(qnn_results)


# ### Accuracy

# In[62]:


accuracy = qnn_results[1] * 100
print("The accuracy of the model is: ", str(accuracy) + ' %')


# # **Inference on an encrypted image**

# ### Importing the necessary packages

# In[63]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cryptography.fernet import Fernet
import time
import cv2


# ### Constructing necessary functions

# In[64]:


# Function to display image
def show_image(image):
    img = mpimg.imread(image)
    print('Shape of the image is: ', img.shape)
    plt.axis('off')
    imgplot = plt.imshow(img)
    plt.show()


# Function to encrypt an image
def encrypt(image):
    key = Fernet.generate_key()
    with open('mykey.key', 'wb') as mykey:
        mykey.write(key)
    with open('mykey.key', 'rb') as mykey:
        key = mykey.read()
    print("Your secret key is: ", key)
    f = Fernet(key)
    with open(image, 'rb') as original_file:
        original = original_file.read()
    print("\nEncrypting the image with the secret key....")
    time.sleep(5)
    print("\nEncryption successful!")
    encrypted = f.encrypt(original)
    with open ('encrypted_image.png', 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

        
# Function to decrypt an image
def decrypt(enc_img, key):
    f = Fernet(key)
    with open('encrypted_image.png', 'rb') as encrypted_file:
        encrypted = encrypted_file.read()
    print("Decrypting the given image....")
    time.sleep(5)
    decrypted = f.decrypt(encrypted)
    with open('decrypted_image.png', 'wb') as decrypted_file:
        decrypted_file.write(decrypted)
    print("\nDecryption successful!")


# Function to process an image and perform the prediction
def image_processing_and_prediction(image):
    img = image[..., np.newaxis]/255.0
    img = tf.image.resize(img, (4,4)).numpy()
    # print("Shape of the image is: ",img.shape)
    threshold = 0.5
    img = np.array(img > threshold, dtype=np.float32)
    img_circ = [convert_image_to_circuit(img)]
    processed_img = tfq.convert_to_tensor(img_circ)

    # Prediction
    prediction = model.predict(processed_img)
    # print('Prediction is: ', prediction)
    if prediction[0][0] < 0:
        print("The image is of digit 3")
    else:
        print("The image is of digit 6")


# ### Sample Image

# In[65]:


digit_img = './three.png'
show_image(digit_img)


# ### Encrypting the image

# In[66]:


encrypt(digit_img)


# ### Checking whether the key and encrypted image is generated

# In[73]:


get_ipython().system('ls')


# ### Decrypting the image

# In[68]:


with open('./mykey.key', mode='rb') as file:
    secret_key = file.read()

enc_img = './encrypted_image.png'
decrypt(enc_img, secret_key)


# ### Checking whether the decrypted image is generated

# In[72]:


get_ipython().system('ls')


# ### Seeing the decrypted image

# In[74]:


show_image('./decrypted_image.png')


# ### Loading the decrypted image and resizing

# In[75]:


# Load decrypted image
file = r'./decrypted_image.png'
test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
before_resizing = test_image.shape

# Resizing the image to make it suitable for prediction
img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)
after_resizing = img_resized.shape

print("The size of the image has been resized from " + str(before_resizing) + " to " + str(after_resizing))


# ### Inference (prediction)

# In[76]:


image_processing_and_prediction(img_resized)


# In[ ]:




