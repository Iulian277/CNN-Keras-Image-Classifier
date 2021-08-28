from imports import *
from data_processing import train_batches, valid_batches

# We want to classify into 2 classes: 'cat' and 'dog'
# The vgg16 model has the last layer with 1000 neurons
# So, he will classify the input in one of those 1000 classes

# We will create a new model with the same layers as the vgg16 model,
# but we will remove the last layer from vgg16 and add our own one

# Download (/Load) the vgg16 model
vgg16_model = tf.keras.applications.vgg16.VGG16()

# Create a new sequential model
model = Sequential()

# For each layer excepting the last one
# We don't include the predictions dense layer
for layer in vgg16_model.layers[: -1]:
    model.add(layer)

# Freeze the base
# Set the layers to not be trainable, because the weights and biases are already trained
for layer in model.layers:
    layer.trainable = False

# Add our predictions layer (output layer) with 2 classes ('cat' and 'dog')
model.add(Dense(units = 2, activation = 'softmax'))

# Prepare the model for training
model.compile(optimizer = Adam(learning_rate = 0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model
model.fit(x = train_batches,
          steps_per_epoch = len(train_batches),
          validation_data = valid_batches,
          validation_steps = len(valid_batches),
          epochs = 5,
          verbose = 2)

# Save the model
if os.path.isfile('../models/vgg16_dogs_vs_cats_model.h5') is False:
    model.save('../models/vgg16_dogs_vs_cats_model.h5')
else:
    print('The model is already saved in the "models" folder')
