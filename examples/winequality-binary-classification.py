import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import pandas as pd
import numpy as np

from core import Dataloader, Sequential, BinaryCrossEntropyLoss, LinearLayer, ReLU, Sigmoid, Trainer
from core.dataset import NumpyDataset
from core.optimizer import GradientDescent


def get_train_test_numpy(df, split_ration):
    X = df.values[:, :-1]
    y = df.values[:, -1]

    train_indices = np.random.choice(len(X), round(len(X) * split_ration), replace=False)

    X_train = X[train_indices]
    y_train = y[train_indices]

    test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    df = pd.read_csv("./data/red_wine_quality/winequality-red.csv")
    df["quality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

    X_train, y_train, X_test, y_test = get_train_test_numpy(df, 0.8)

    train_dataset = NumpyDataset(X_train, y_train)
    test_dataset = NumpyDataset(X_test, y_test)

    train_dataloader = Dataloader(train_dataset)
    test_dataloader = Dataloader(test_dataset)

    loss_fn = BinaryCrossEntropyLoss()

    model = Sequential(
        loss_fn=loss_fn,
        layers=[
            LinearLayer(in_dimension=11, out_dimension=5, activation_fn=ReLU),
            LinearLayer(in_dimension=5, out_dimension=1, activation_fn=Sigmoid),
        ],
    )

    trainer = Trainer(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        epochs=2,
        optimizer=GradientDescent(model, learning_rate=0.01),
    )

    trainer.train()

    __, acc, __, __= trainer.test()
    print("Accuracy: ", acc)


