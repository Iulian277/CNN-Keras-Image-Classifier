from imports import *
from data_processing import train_batches, valid_batches

# Declare the model
model = Sequential([
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (224, 224, 3)),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Flatten(),
    Dense(units = 2, activation = 'softmax')])

model.summary()

# Prepare the model for training
model.compile(optimizer = Adam(learning_rate = 0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model
model.fit(x = train_batches,
          validation_data = valid_batches,
          epochs = 10,
          steps_per_epoch = len(train_batches),
          validation_steps = len(valid_batches),
          verbose = 2)

# Save the model
if os.path.isfile('../models/dogs_vs_cats_model.h5') is False:
    model.save('../models/dogs_vs_cats_model.h5')
