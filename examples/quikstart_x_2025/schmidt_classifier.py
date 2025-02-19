import numpy as np

import pandas as pd
import kagglehub
from IPython.display import Markdown, display
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_experiments.library import StateTomography

from qclib.state_preparation.util.baa import _split_combinations
from qdna.quantum_info import correlation
from qdna.compression import SchmidtCompressor

from torch import optim, tensor, float32, no_grad

# Define torch NN module
from torch.nn import (
    Module,
    Linear,
    Sequential,
    CrossEntropyLoss,
    Sigmoid
)

class NeuralNet(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_stack = Sequential(
            Linear(input_size, output_size, dtype=float32),
            Sigmoid()
        )

    def forward(self, x):
        x = self.linear_stack(x)
        return x.reshape(-1)

def complete_circuit(initializer, opt_params, n_qubits, state, compressor):
    # Typical state initializer.
    if opt_params is not None:
        init_gate = initializer(state, opt_params=opt_params)
    else:
        init_gate = initializer(state)

    # Creates the quantum circuit.
    circuit = QuantumCircuit(n_qubits)

    # Circuit on Alice's side.
    circuit.append(init_gate.definition, range(n_qubits))
    circuit.append(compressor.definition, range(n_qubits))

    return circuit

def trash_data(initializer, opt_params, n_qubits, input_data, compressor, backend, shots, optimization_level=2, verbose=1):
    # Applies the compression process to each of the test samples.

    rhos = []

    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)

    # Iterates through all test samples.
    for i, test_sample in enumerate(input_data):
        circuit = complete_circuit(initializer, opt_params, n_qubits, test_sample, compressor)
        circuit = pm.run(circuit)

        if i == 0 and verbose:
            print('\t\tcircuit depth:', circuit.depth())
            print('\t\tcircuit cnots:', circuit.count_ops().get('cx', 0)+circuit.count_ops().get("cz", 0))

        # trash_qubits = set(compressor.trash_qubits)
        trash_state = []
        # Tomography of individual trash qubits.
        for trash_qubit in compressor.trash_qubits:
            # trace_out = compressor.latent_qubits + list(trash_qubits - set([trash_qubit]))
            # temp_state = partial_trace(DensityMatrix(circuit), trace_out).data
            temp_state = tomography(backend=backend, circuit=circuit, qubit=trash_qubit, shots=shots)
            trash_state.append(temp_state)

        trash_state = np.array(trash_state).reshape(-1)
        # trash_state = np.concatenate([[np.real(e), np.imag(e)] for e in trash_state])
        trash_state = np.array([np.real(e) for e in trash_state])

        # Stores and prints the results.
        rhos.append(trash_state)

    return rhos

def tomography(backend, circuit, qubit, shots):
    qst = StateTomography(circuit, backend=backend, measurement_indices=[qubit])
    qstdata = qst.run(backend, seed_simulation=42, shots=shots).block_for_results()

    return qstdata.analysis_results("state").value

def load_fraud_dataset(size=None, seed=42, remove_categorical=False, padding=True, normalize=True):
    """
    Reads the Bank Account Fraud Dataset, encodes categorical columns, rescales all columns,
    balances the dataset, randomly selects pairs of labels and features, ensures the dataset remains balanced,
    and limits the number of samples if specified.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
        size (int, optional): Maximum number of samples to return. Defaults to None.
        remove_categorical (bool, optional): Whether to remove categorical columns from the features. Defaults to False.

    Returns:
        features (np.ndarray): Numpy array containing the rescaled and balanced feature columns.
        labels (np.ndarray): Numpy array containing the balanced "fraud_bool" column.
    """

    # Download latest version
    path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
    file_path = path + '/Base.csv'

    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure the "fraud_bool" column exists
    if "fraud_bool" not in data.columns:
        raise ValueError("The dataset does not contain a column named 'fraud_bool'.")

    # Separate features and labels
    labels = data["fraud_bool"]
    features = data.drop(columns=["fraud_bool"])

    if remove_categorical:
        categorical_columns = features.select_dtypes(include=['object']).columns
        features = features.drop(columns=categorical_columns)
    else:
        # Encode categorical columns
        label_encoders = {}
        for column in features.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            features[column] = le.fit_transform(features[column])
            label_encoders[column] = le

    # Rescale all columns to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)

    # Balance the dataset
    ros = RandomOverSampler(random_state=42)
    features, labels = ros.fit_resample(features, labels)

    # Combine features and labels into a single dataset for shuffling
    if size is None:
        size = len(labels)
    target_size_per_class = size // 2

    indices_class_0 = np.where(labels == 0)[0]
    indices_class_1 = np.where(labels == 1)[0]

    np.random.seed(seed)

    selected_class_0 = np.random.choice(indices_class_0, size=target_size_per_class, replace=False)
    selected_class_1 = np.random.choice(indices_class_1, size=target_size_per_class, replace=False)

    selected_indices = np.concatenate((selected_class_0, selected_class_1))
    np.random.shuffle(selected_indices)

    features = features[selected_indices]
    labels = labels[selected_indices]

    # Additional pre-processing
    feature_dim = len(features[0])
    n_qubits = int(np.ceil(np.log2(feature_dim)))

    # Padding
    if padding and feature_dim < 2**n_qubits:
        pad = np.zeros((target_size_per_class*2, 2**n_qubits-feature_dim))
        features = np.hstack((features, pad))

    # Normalize
    if normalize:
        features = features / np.linalg.norm(features, axis=1).reshape(
            (len(features), 1)
        )

    return features, labels.to_numpy()

def load_train_test_data(target_class, features, labels, seed=42, training_size=0.5):

    selected_indices = np.where(labels == target_class)[0]

    _features = features[selected_indices]
    _labels = labels[selected_indices]

    sample_train, sample_test, label_train, label_test = train_test_split(
        _features, _labels, train_size=training_size, random_state=seed
    )

    # Pick training and test size number of samples for each class label
    training_input = sample_train[label_train == target_class, :][:]
    test_input = sample_test[label_test == target_class, :][:]

    return training_input, test_input

def find_partitioning(n_qubits, typical_state, n_trash_partition, verbose=0):
    n_latent_partition = n_qubits - n_trash_partition
    min_entropy = np.iinfo(np.int32).max # 2147483648
    trash_partition = None
    for partition in _split_combinations(range(n_qubits), n_latent_partition):
        set_a = set(partition)
        set_b = set(range(n_qubits)).difference(set_a)
        entropy = correlation(typical_state, set_a, set_b)

        if verbose > 0:
            print('latent', set_a, 'trash', set_b, 'entropy', entropy)

        if entropy <= min_entropy:
            min_entropy = entropy
            latent_partition = sorted(partition)
            trash_partition = sorted(set(range(n_qubits)).difference(set(latent_partition)))

    return latent_partition, trash_partition

# Estimate the centroid.
# Simply the average of the training samples (or a random selection of samples).
def calculate_typical_state(n_qubits, training_input):
    centroid = np.zeros(2**n_qubits)
    for train_sample in training_input:
        centroid += train_sample

    typical_state = centroid / np.linalg.norm(centroid)

    return typical_state

def display_results(results):
    def calculate_metrics(result, index):
        """Calculate mean and standard deviation for a given metric index."""
        avg = np.mean(result[index])
        std = np.std(result[index])
        return avg, std

    def format_row(max_fidelity_loss, metrics):
        """Format a single row for the table."""
        return (
            f'| {round(max_fidelity_loss, 4)} | {round(metrics["avg_tp"], 4)} | {round(metrics["std_tp"], 4)} '
            f'| {round(metrics["avg_tn"], 4)} | {round(metrics["std_tn"], 4)} '
            f'| {round(metrics["avg_fp"], 4)} | {round(metrics["std_fp"], 4)} '
            f'| {round(metrics["avg_fn"], 4)} | {round(metrics["std_fn"], 4)} '
            f'| {round(metrics["avg_acc"], 4)} | {round(metrics["std_acc"], 4)} '
            f'| {round(metrics["avg_f1"], 4)} | {round(metrics["std_f1"], 4)} '
            f'| {round(metrics["avg_mcc"], 4)} | {round(metrics["std_mcc"], 4)} '
            f'| {round(metrics["avg_auc"], 4)} | {round(metrics["std_auc"], 4)} |\n'
        )

    # Initialize table header
    header = (
        '| max loss | avg. TP | std. TP | avg. TN | std. TN | avg. FP | std. FP | avg. FN | std. FN | '
        'avg. acc | std. acc | avg. F1 | std. F1 | avg. MCC | std. MCC | avg. AUC | std. AUC |\n'
        '|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n'
    )

    # Generate table rows
    rows = ""
    for max_fidelity_loss, result in results.items():
        metrics = {
            "avg_tp": calculate_metrics(result, 1)[0], "std_tp": calculate_metrics(result, 1)[1],
            "avg_tn": calculate_metrics(result, 2)[0], "std_tn": calculate_metrics(result, 2)[1],
            "avg_fp": calculate_metrics(result, 4)[0], "std_fp": calculate_metrics(result, 4)[1],
            "avg_fn": calculate_metrics(result, 5)[0], "std_fn": calculate_metrics(result, 5)[1],
            "avg_acc": calculate_metrics(result, 3)[0], "std_acc": calculate_metrics(result, 3)[1],
            "avg_f1": calculate_metrics(result, 6)[0], "std_f1": calculate_metrics(result, 6)[1],
            "avg_mcc": calculate_metrics(result, 7)[0], "std_mcc": calculate_metrics(result, 7)[1],
            "avg_auc": calculate_metrics(result, 10)[0], "std_auc": calculate_metrics(result, 10)[1],
        }
        rows += format_row(max_fidelity_loss, metrics)

    # Display the complete table
    table = header + rows
    display(Markdown(table))

def configure_compressor(features, labels, target_class, n_qubits, n_trash_partition, training_size, backend, seed=42, verbose=1):

    # Load training data
    training_input, _ = load_train_test_data(
        target_class, features, labels, seed=seed, training_size=training_size
    )

    # Calculate the typical state
    typical_state = calculate_typical_state(n_qubits, training_input)

    # Search for the best partitioning
    _, trash_partition = find_partitioning(
        n_qubits, typical_state, n_trash_partition, verbose=0
    )

    # Build the compressor
    compressor = SchmidtCompressor(
        typical_state, opt_params={'partition': trash_partition, 'lr': 0}
    )

    # Print compressor details
    if verbose:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
        isa_circuit = pm.run(compressor.definition)

        print(f'\ttrash qubits: {n_trash_partition}')
        print(f'\ttrash_partition: {trash_partition}')
        print(f'\tcompressor depth: {isa_circuit.depth()}')
        print(f'\tcompressor cnots: {isa_circuit.count_ops().get("cx", 0)+isa_circuit.count_ops().get("cz", 0)}')

    return compressor

def extract_features(features, labels, target_class, compressor, n_qubits, training_size, backend, shots, initializer, opt_params, seed=42, verbose=1):
    anomaly_class = [j for j in range(2) if j != target_class][0]
    train_data, train_targets = [], []
    test_data, test_targets = [], []

    for i, label in enumerate([target_class, anomaly_class]):
        _training_input, _test_input = load_train_test_data(
            label, features, labels, seed=seed, training_size=training_size
        )

        train_data.extend(
            trash_data(
                initializer, opt_params, n_qubits, _training_input, compressor, backend, shots, verbose=verbose
            )
        )
        train_targets.extend([i] * len(_training_input))

        test_data.extend(
            trash_data(
                initializer, opt_params, n_qubits, _test_input, compressor, backend, shots, verbose=verbose
            )
        )
        test_targets.extend([i] * len(_test_input))

    if verbose:
        rho_size = len(train_data[0])
        print(f'\trho_size: {rho_size}')

    return train_data, train_targets, test_data, test_targets

def new_data(batch_size, x, y):
    x_new, y_new = [], []
    for _ in range(batch_size):
        n = np.random.randint(len(x))
        x_new.append(x[n])
        y_new.append(y[n])

    return (
        tensor(np.array(x_new), dtype=float32),
        tensor(np.array(y_new), dtype=float32)
    )

def training(train_data, train_targets, iters, batch_size, verbose=1):
    rho_size = len(train_data[0])

    model = NeuralNet(rho_size, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_func = CrossEntropyLoss()

    loss_list = []
    model.train()

    for i in range(iters):
        x_batch, y_batch = new_data(batch_size, train_data, train_targets)
        optimizer.zero_grad(set_to_none=True)
        output = model(x_batch)
        loss = loss_func(output, y_batch)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    if verbose:
        print(
            f"Training [{100.0 * (i + 1) / iters:.0f}%]\t"
            f"Loss: {loss_list[-1]:.4f}\t"
            f"Avg. Loss: {sum(loss_list) / len(loss_list):.4f}"
        )

    return model

def testing(model, test_data, test_targets, verbose=1):
    total_loss = []
    scores = []
    tp = tn = total_p = total_n = acc = fp = fn = f1 = mcc = 0

    loss_func = CrossEntropyLoss()

    model.eval()
    with no_grad():
        for data, target in zip(test_data, test_targets):
            data = tensor(data, dtype=float32)
            target = tensor([target], dtype=float32)
            output = model(data)
            pred = np.round(output, 0)

            scores.append(output)

            if pred == target:
                tp += int(pred == 0)
                tn += int(pred != 0)
            else:
                fp += int(pred == 0)
                fn += int(pred != 0)

            total_p += int(target == 0)
            total_n += int(target != 0)

            loss = loss_func(output, target)
            total_loss.append(loss.item())

        acc = (tp + tn) / (total_p + total_n)
        f1 = (2 * tp) / (2 * tp + fp + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        auc = roc_auc_score(test_targets, scores)

        if verbose:
            print(
                f"\tPerformance on test data:\n\tLoss: {sum(total_loss) / (total_p + total_n):.4f}\n\t"
                f"Accuracy: {acc * 100:.1f}%\n\t"
                f"MCC: {mcc:.2f}\n\t"
                f"Accuracy 0: {tp / total_p * 100:.1f}%\n\t"
                f"Accuracy 1: {tn / total_n * 100:.1f}%"
            )

    return total_loss, tp, tn, acc, fp, fn, f1, mcc, scores, auc
