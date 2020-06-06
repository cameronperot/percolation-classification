import os
import numpy as np
from tensorflow import keras

from utils import *


data_path = os.path.abspath("../data")
model_path = os.path.abspath("../artifacts")

model_name = "model_32"

data = Data(os.path.join(data_path, "data_transition.npz"), test_ratio=1)
model = keras.models.load_model(os.path.join(model_path, model_name + ".h5"))


# %%


score = model.evaluate(data.test.X, data.test.Y, verbose=0)
print("Loss:     %.3f" % score[0])
print("Accuracy: %.3f" % (score[1] * 100))


Y_hat = model.predict(data.test.X)
plot_p_output_avgs(Y_hat, data.test.p, save_as="transition.png")
