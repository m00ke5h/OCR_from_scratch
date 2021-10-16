#!/usr/bin/python3


# Constant values
TEST_DIR = "test/"
TEST_DATA = "data/t10k-images-idx3-ubyte"
TEST_LABELS = "data/t10k-labels-idx1-ubyte"
TRAINING_DATA = "data/train-images-idx3-ubyte"
TRAINING_LABELS = "data/train-labels-idx1-ubyte"
DEBUG = False  # Set Debug True/False
k = 5  # Hyperparameter
NUM_TRAINING = 10000  # Number of images to train
NUM_TEST = 30  # Number of images to test


if DEBUG:
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert("L"))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), "L")
        img.save(path)


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, "big")


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, "rb") as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_index in range(n_images):
            image = []
            for row_index in range(n_rows):
                row = []
                for col_index in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, "rb") as f:
        _ = f.read(4)
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_index in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def euclidean_distance(x, y):
    return sum(
        [(bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 for x_i, y_i in zip(x, y)]
    ) ** (0.5)


def get_distances_for_test_sample(X_train, test_sample):
    return [euclidean_distance(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, Y_train, X_test, k):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        training_distances = get_distances_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(enumerate(training_distances), key=lambda x: x[1])
        ]
        candidates = [Y_train[idx] for idx in sorted_distance_indices[:k]]
        top_canditate = get_most_frequent_element(candidates)
        y_pred.append(top_canditate)
    return y_pred


def main():
    X_train = read_images(TRAINING_DATA, NUM_TRAINING)
    Y_train = read_labels(TRAINING_LABELS, NUM_TRAINING)
    X_test = read_images(TEST_DATA, NUM_TEST)
    Y_test = read_labels(TEST_LABELS, NUM_TEST)

    if DEBUG:
        for idx, test_sample in enumerate(X_test):
            write_image(test_sample, f"{TEST_DIR}{idx}.png")
        X_test = [read_image(f"{TEST_DIR}our_test.png")]
        Y_test = [5]

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train, Y_train, X_test, k)
    print(f"Predicted labels: \n{y_pred}")
    accuracy = sum(
        [int(y_pred_i == y_test_i) for y_pred_i, y_test_i in zip(y_pred, Y_test)]
    ) / len(Y_test)
    print(f"\n\nAccuracy: {accuracy*100}%")


if __name__ == "__main__":
    main()
