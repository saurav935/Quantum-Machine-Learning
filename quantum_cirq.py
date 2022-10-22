import cirq

# Creating qubits one by one
qubit_0 = cirq.LineQubit(0)           # Qubit number 1
qubit_1 = cirq.LineQubit(1)           # Qubit number 2
qubit_2 = cirq.LineQubit(2)           # Qubit number 3
qubit_3 = cirq.LineQubit(3)           # Qubit number 4
qubit_4 = cirq.LineQubit(4)           # Qubit number 5
qubit_5 = cirq.LineQubit(5)           # Qubit number 6


# Creating qubits using 'range' function
total_qubits = cirq.LineQubit.range(6)

# Creating grid qubits
grid_qubit = cirq.GridQubit(1,2)

# Creating qubits in a rectangular grid shape
rectangular_grid_qubits = cirq.GridQubit.rect(rows=4, cols=3)

# Creating qubits in a square grid shape
square_grid_qubits = cirq.GridQubit.square(2)   # 2 is the size of the square

# Initializing last qubit with a state of 1, applying Hadamard gate to every qubit and appending them to an empty circuit
circuit = cirq.Circuit(cirq.X(total_qubits[5]))
circuit.append(cirq.H(total_qubits[0]))
circuit.append(cirq.H(total_qubits[1]))
circuit.append(cirq.H(total_qubits[2]))
circuit.append(cirq.H(total_qubits[3]))
circuit.append(cirq.H(total_qubits[4]))
circuit.append(cirq.H(total_qubits[5]))

# Doing the same as above, but in an efficient way
circuit = cirq.Circuit(cirq.X(total_qubits[5]))

for qubit in total_qubits:
    circuit.append(cirq.H(qubit))


# Applying CNOT gates to qubits that go from 4-5, 1-5, and 0-5
circuit.append(cirq.CNOT(total_qubits[4], total_qubits[5]))
circuit.append(cirq.CNOT(total_qubits[1], total_qubits[5]))
circuit.append(cirq.CNOT(total_qubits[0], total_qubits[5]))

# Applying Hadamard gates
for qubit in total_qubits:
    circuit.append(cirq.H(qubit))
    

##########################################################################


# CREATING CIRCUIT USING MOMENTS

# Initializing 6 qubits
total_qubits = cirq.LineQubit.range(6)

# Changing the state of the last qubit from 0 to 1
first_moment = cirq.Moment(cirq.X(total_qubits[5]))

# Applying the Hadamard gates on each qubits
second_moment = cirq.Moment(cirq.H.on_each(total_qubits))

# Creating an empty circuit
circuit = cirq.Circuit()

# Appending first moment to the circuit
circuit.append(first_moment)

# Appending second moment to the circuit
circuit.append(second_moment)

# Appending CNOT gates
circuit.append(cirq.CNOT(total_qubits[4], total_qubits[5]))
circuit.append(cirq.CNOT(total_qubits[1], total_qubits[5]))
circuit.append(cirq.CNOT(total_qubits[0], total_qubits[5]))

# Applying the final Hadamard gates
circuit.append(second_moment)


# SIMULATION:

# Before starting the simulation, we need to insert measurements at the end of our quantum circuits
measurement = cirq.Moment([cirq.measure(qubit) for qubit in total_qubits])

# Append it to the circuit
circuit.append(measurement_gates)

# Initializing the quantum simulator
quantum_simulator = cirq.Simulator()

# Running the circuit on the simulator
results = quantum_simulator.run(circuit)

