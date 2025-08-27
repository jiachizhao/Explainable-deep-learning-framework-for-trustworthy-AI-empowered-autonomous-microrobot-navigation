import tensorflow as tf
import os
from tensorflow.keras import layers, models


data_dir = 'D:/dataset/dataset_for_GMP'  
def get_label_from_filename(filename):
    label = int(filename.split('_')[-1].split('.')[0])
    return label

def load_data(data_dir):
    images = []
    labels = []
    for file in os.listdir(data_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            filepath = os.path.join(data_dir, file)
            label = get_label_from_filename(file)
            image = tf.keras.preprocessing.image.load_img(filepath, target_size=(32, 32))
            image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
            image = image[:, :, :3]
            images.append(image)
            labels.append(label)
    return images, labels

images, labels = load_data(data_dir)
images = tf.convert_to_tensor(images, dtype=tf.float32)
labels = tf.convert_to_tensor(labels, dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size=len(images))
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")

train_size = int(0.8 * len(images))
val_size = int(0.1 * len(images))
test_size = len(images) - train_size - val_size
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

batch_size = 32
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(8, activation='softplus')
    ])
    return model

model = create_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=200, validation_data=val_dataset)

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f"Test accuracy: {test_acc}")

model.save('saved_model/global_model')