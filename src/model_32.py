from comet_ml import Experiment
import os
import pickle
import numpy as np
from tensorflow import keras

from utils import *


train_model = False
load_model = True
model_name = "model_32"

data_path = os.path.abspath("../data")
model_path = os.path.abspath("../artifacts")
if not os.path.exists(model_path):
    os.makedirs(model_path)

data = Data(
    os.path.join(data_path, "data_train_test.npz"),
    use_transpose=True,
    valid_ratio=0.05,
    test_ratio=0.05,
)

n_epochs = 20
batch_size = 64
lr_initial = 5e-3
lr_decay_factor = 0.8
steps_per_epoch = len(data.train.X) // batch_size


# %%


if load_model:
    model = keras.models.load_model(os.path.join(model_path, model_name + ".h5"))
else:

    def conv_block(x, n_layers, n_filters):
        for _ in range(n_layers):
            x = keras.layers.Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)

        return x

    def create_model(data):
        x_input = keras.layers.Input(shape=data.input_shape)

        x = conv_block(x_input, 4, 16)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = conv_block(x, 4, 32)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = conv_block(x, 4, 64)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        output = keras.layers.Dense(data.n_classes, activation="softmax")(x)

        return keras.Model([x_input], output)

    model = create_model(data)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        lr_initial, steps_per_epoch, lr_decay_factor, staircase=True
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-3, amsgrad=True)

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"],
    )

model.summary()


# %%


if train_model:
    gen_train = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True,
    )
    gen_train.fit(data.train.X)

    fit = model.fit(
        gen_train.flow(data.train.X, data.train.Y, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        validation_data=(data.valid.X, data.valid.Y),
        verbose=2,
    )

    model.save(os.path.join(model_path, model_name + ".h5"))
    with open("fit_history.pkl", "wb") as f:
        pickle.dump(fit.history, f)

    plot_fit_history(fit.history, save_as="fit_history.png")


# %%


data.analyze_predictions("train", model, plot_misclassified=True)
data.analyze_predictions("valid", model, plot_misclassified=True)
data.analyze_predictions("test", model, plot_misclassified=True)
