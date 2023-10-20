from abc import ABC, abstractmethod

import numpy as np


class Transform(ABC):
    @abstractmethod
    def transform(x):
        ...

    @abstractmethod
    def transform(x, y):
        ...

    def __call__(self, x):
        return self.transform(x)


class Sequence(Transform):
    def __init__(self, *transforms):
        self.transforms = transforms

    def transform(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def transform(self, x, y):
        for transform in self.transforms:
            x, y = transform.transform(x, y)
        return x, y


class Binarizer(Transform):
    def __init__(self, target):
        self.target = target

    def transform(self, x):
        return np.where(x == self.target, 1, 0)


class OneHotEncoder(Transform):
    def __init__(self, num_categories):
        self.num_categories = num_categories

        # Create the identity matrix
        self.identity_matrix = np.eye(self.num_categories)

    def transform(self, y):
        # One-hot encode the numbers
        y = y.astype(int)
        one_hot_encoded = self.identity_matrix[y]
        return one_hot_encoded


class MinoritySampler(Transform):
    def transform(self, x, y, min_occurences=0):
        # Get the number of samples for each category
        unique_items, counts = np.unique(y, axis=0, return_counts=True)

        # Get the minimum number of samples
        min_count = np.min(counts)

        # Get the indices of the samples to keep
        X_balanced, y_balanced = [], []
        for label in unique_items:
            indices = np.where(np.all(y == label, axis=1))[0]

            # Shuffle the indices
            np.random.shuffle(indices)

            # Randomly select a subset of samples
            indices_subset = np.random.choice(indices, size=min_count, replace=False)

            # Append the samples to the balanced dataset
            X_balanced.append(x[indices_subset])
            y_balanced.append(y[indices_subset])

        # Concatenate the subsets of each class into a new dataset
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)

        return X_balanced, y_balanced


class DoubleIdle(Transform):
    def transform(self, x, y, min_occurences=0):
        # Get the number of samples for each category
        unique_items, counts = np.unique(y, axis=0, return_counts=True)

        # Get the minimum number of samples
        min_count = np.min(counts)

        # Get the indices of the samples to keep
        X_balanced, y_balanced = [], []
        for label in unique_items:
            indices = np.where(np.all(y == label, axis=1))[0]

            # Shuffle the indices
            np.random.shuffle(indices)

            # Randomly select a subset of samples
            if np.argmax(label) != 0:
                indices_subset = np.random.choice(
                    indices, size=min_count, replace=False
                )
            else:
                indices_subset = np.random.choice(
                    indices, size=2 * min_count, replace=False
                )

            # Append the samples to the balanced dataset
            X_balanced.append(x[indices_subset])
            y_balanced.append(y[indices_subset])

        # Concatenate the subsets of each class into a new dataset
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)

        return X_balanced, y_balanced


class TrimIdle(Transform):
    def transform(self, x, y):
        # Get the number of samples for each category
        unique_items, counts = np.unique(y, axis=0, return_counts=True)

        idle_idx = counts.argmax()
        idle_label = unique_items[idle_idx]

        # remove idle from counts and unique items
        new_counts = np.delete(counts, idle_idx, axis=0)

        second_most_item_idx = new_counts.argmax()
        second_most_item_count = new_counts[second_most_item_idx]

        # Get the indices of the samples to keep
        X_balanced, y_balanced = [], []
        for label in unique_items:
            indices = np.where(np.all(y == label, axis=1))[0]

            # Shuffle the indices
            np.random.shuffle(indices)

            if np.all(label == idle_label):
                # Randomly select a subset of samples
                indices = np.random.choice(
                    indices, size=second_most_item_count, replace=False
                )

            # Append the samples to the balanced dataset
            X_balanced.append(x[indices])
            y_balanced.append(y[indices])

        # Concatenate the subsets of each class into a new dataset
        X_balanced = np.concatenate(X_balanced)
        y_balanced = np.concatenate(y_balanced)

        return X_balanced, y_balanced


class Standardizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def save_parameters(self, file_name):
        import json

        parameter_dict = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

        with open(file_name, "w") as f:
            json.dump(parameter_dict, f)

    def load_parameters(self, filename):
        import json

        with open(filename) as f:
            jsonString = f.read()
        parameter_dict = json.loads(jsonString)
        self.mean = np.array(parameter_dict["mean"])
        self.std = np.array(parameter_dict["std"])

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-9)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class MinMaxScaler:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def save_parameters(self):
        import json

        parameter_dict = {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

        with open("standardizer_parameters", "w") as f:
            json.dump(parameter_dict, f)

    def fit(self, x):
        self.min = np.min(x, axis=0)
        self.max = np.max(x, axis=0)

    def transform(self, x):
        # return (x - self.min) / (self.max - self.min + 1e-9)
        return (x - self.min) / (self.max - self.min)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
