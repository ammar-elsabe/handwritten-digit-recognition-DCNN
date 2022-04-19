import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# load the mnist dataset
dstest, dsinfo = tfds.load(
    'mnist',
    split=['test'],  # only need the test set
    data_dir='../dataset/',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

dstest = dstest[0]
batch_size = 128

# Evaluation pipleine
dstest = dstest.batch(batch_size)
dstest = dstest.cache()
dstest = dstest.prefetch(tf.data.AUTOTUNE)

# load the model
model = tf.keras.models.load_model('./model.h5')

class_names = dsinfo.features['label'].names

model_probabilities = model.predict(dstest)

predictions = [np.argmax(x) for x in model_probabilities]

labels = np.concatenate([y for x, y in dstest], axis=0)

confusion_matrix = tf.math.confusion_matrix(
    labels=labels,
    predictions=predictions,
)

sns.heatmap(confusion_matrix,
            annot=True,
            xticklabels=class_names,
            yticklabels=class_names,
            fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('../paper/figs/confusion_matrix.svg', format='svg')
plt.show()
