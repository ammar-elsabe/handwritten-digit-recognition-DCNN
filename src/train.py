import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

batch_size = 128

# Import the dataset mnist is 60k images of 28x28 pixels
# And 10k images for testing
(dstrain, dstest), dsinfo = tfds.load(
    'mnist',
    split=['train', 'test'],
    data_dir='../dataset/',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
# Summarize loaded datasets
print('\nDataset info:')
print('Image shape:')
print(dsinfo.features['image'].shape)
print('Class Names')
print(dsinfo.features['label'].names)


# Visualize a single image
def visualize_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title('Label: {}'.format(label))
    plt.show()


# Now use that function
mnist_example = dstrain.take(1)
for sample in mnist_example:
    image, label = sample[0], sample[1]
    visualize_image(image, label)
    break


# Preprocessing

# Normalization function


# mapping
dstrain = dstrain.batch(batch_size)
dstrain = dstrain.cache()
dstrain = dstrain.shuffle(dsinfo.splits['train'].num_examples)
dstrain = dstrain.prefetch(tf.data.AUTOTUNE)


# Evaluation pipleine
dstest = dstest.batch(batch_size)
dstest = dstest.cache()
dstest = dstest.prefetch(tf.data.AUTOTUNE)

# Create the model

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(28, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(28, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# build the model
model.build()
# print the model summary
model.summary()

history = model.fit(
    dstrain,
    epochs=30,
    validation_data=dstest,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1)
    ]
)


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(accuracy))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('../paper/figs/accuracy_loss.svg', format='svg')
plt.show()
