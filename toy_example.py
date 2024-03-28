from __future__ import absolute_import, division, print_function, unicode_literals

import os
from tempfile import gettempdir

import tensorflow as tf
from clearml import Task
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

task = Task.init(project_name='toy_example', task_name='tf_training_with_logging')
# output_model = OutputModel(task=task, framework='tensorflow')

# parser = argparse.ArgumentParser()
# parser.add_argument('-lr', default=0.001, help='Init lr')
# parser.add_argument('-epochs', default=3, help='Total epochs')
# args = parser.parse_args()

# extra_params = {'divider': 255.0,
#                 'units': 128
#                 }
# task.connect(extra_params)

config_file_yaml = task.connect_configuration(name='my yaml file', configuration='config.yaml')

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(1024)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1024)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', dtype=tf.float32)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu', dtype=tf.float32)
        self.d2 = Dense(10, activation='softmax', dtype=tf.float32)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = MyModel()

# Choose an optimizer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model.
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# Test the model
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


train_log_dir = os.path.join(gettempdir(), 'logs', 'gradient_tape', 'train')
test_log_dir = os.path.join(gettempdir(), 'logs', 'gradient_tape', 'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
manager = tf.train.CheckpointManager(ckpt, os.path.join(gettempdir(), 'tf_ckpts'), max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    print(f'Restored from {manager.latest_checkpoint}')
else:
    print('Initializing from scratch.')

EPOCHS = 5
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    ckpt.step.assign_add(1)
    if int(ckpt.step) % 1 == 0:
        save_path = manager.save()
        print(f'Saved chekcpoint for step {int(ckpt.step)}: {save_path}')

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          round(train_loss.result().numpy(), 4),
                          round(train_accuracy.result().numpy() * 100, 1),
                          round(test_loss.result().numpy(), 4),
                          round(test_accuracy.result().numpy() * 100, 1)))

    try:
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
    except AttributeError:
        train_loss.reset_state()
        train_accuracy.reset_state()
        test_loss.reset_state()
        test_accuracy.reset_state()
