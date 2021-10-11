#!/usr/bin/python3

from math import dist

TEST_DATA = 'data/t10k-images-idx3-ubyte'
TEST_LABELS = 'data/t10k-labels-idx1-ubyte'
TRAINING_DATA = 'data/train-images-idx3-ubyte'
TRAINING_LABELS = 'data/train-labels-idx1-ubyte'


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
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
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_index in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def euclidean_distance(x, y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i))**2 for x_i, y_i in zip(x, y)]
        )**(0.5)


def get_distances_for_test_sample(X_train, test_sample):
    return [euclidean_distance(train_sample, test_sample) for train_sample in X_train]


def knn(X_train, Y_train, X_test, Y_test, k=3):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        training_distances = get_distances_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances), 
                key=lambda x:x[1]
            )
        ]
        candidates = [
            bytes_to_int(Y_train[idx])
            for idx in sorted_distance_indices[:k]
        ]    
        print(f'Point is {bytes_to_int(Y_test[test_sample_idx])} and we guessed {candidates}')
        y_sample = 5
        y_pred.append(y_sample)
    return y_pred    


def main():
    X_train = read_images(TRAINING_DATA, 1000)
    Y_train = read_labels(TRAINING_LABELS)
    X_test = read_images(TEST_DATA, 5)
    Y_test = read_labels(TEST_LABELS)
    # print(len(X_train), len(Y_train), len(X_test), len(Y_test), sep='\n')

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    # print(len(X_train[0]))
    # print(len(X_test[0]))

    knn(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    main()