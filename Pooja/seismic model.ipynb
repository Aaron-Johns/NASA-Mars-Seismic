!pip install obspy

import numpy as np
from obspy import read
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


from google.colab import drive
drive.mount('/content/drive')

def load_data(data_dir):
    """Loads seismic data from miniseed files within the specified directory."""
    data_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mseed"):
                file_path = os.path.join(root, file)
                st = read(file_path)
                data_list.append(st)
    return data_list

def preprocess_data(data_list):
    """Preprocesses seismic data, including filtering and normalization."""
    preprocessed_data = []
    for st in data_list:
        # Apply filtering (e.g., bandpass filter)
        st.filter("bandpass", freqmin=1, freqmax=10)

        # Normalize data
        st.normalize()

        # Extract waveform data
        waveform = st[0].data

        preprocessed_data.append(waveform)

    return preprocessed_data

def create_cnn_model(input_shape):
    """Creates a CNN model for feature extraction."""
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (signal or noise)
    return model


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Trains and evaluates the CNN-SVM model."""
    # Train CNN
    cnn_model = create_cnn_model(input_shape=(X_train.shape[1], 1))
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Extract features from CNN
    X_train_features = cnn_model.predict(X_train)
    X_test_features = cnn_model.predict(X_test)

    # Train SVM
    svm_model = SVC()
    svm_model.fit(X_train_features, y_train)

    # Evaluate SVM
    y_pred = svm_model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Main execution
import os 
data_dir = "/content/drive/MyDrive/zenodo/data_figures/events"  # Replace with the actual path
data_list = load_data(data_dir)


preprocessed_data = preprocess_data(data_list)

features = extract_features(preprocessed_data)
  # Assuming labels are in file names

labels = np.array([1 if "signal" in file else 0 for file in data_list])

def identify_noise(features):
    """Identifies potential noise segments using unsupervised learning."""
    features = features.real 
    # Clustering
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(features)

    # Anomaly detection
    clf = IsolationForest(contamination=0.1)  # Adjust contamination parameter
    predictions = clf.fit_predict(features)

    # Combine results
    noise_mask = (labels == kmeans.cluster_centers_.argmax()) | (predictions == -1)
    return noise_mask

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
noise_mask = identify_noise(features)


print(data_list)

def extract_features(data_list, window_size=100):
    """Extracts features from seismic data using sliding windows."""
    features = []
    for waveform in data_list:
        num_windows = len(waveform) // window_size
        for i in range(0, len(waveform) - window_size + 1, window_size):
            window = waveform[i:i+window_size]
            feature = np.array([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                np.fft.fft(window)[0]  # First frequency component
            ])
            features.append(feature)
    return np.array(features)

num_samples_features = len(features)
num_samples_labels = len(labels)
print("Number of samples in features:", num_samples_features)
print("Number of samples in labels:", num_samples_labels)
print(features[:5])
print(labels[:5])

num_zeros = np.count_nonzero(labels == 0)
num_ones = np.count_nonzero(labels == 1)

print(f"Number of zeros: {num_zeros}")
print(f"Number of ones: {num_ones}")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)



train_and_evaluate_model(X_train, y_train, X_test, y_test)
