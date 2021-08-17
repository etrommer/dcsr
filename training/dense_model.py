import os

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses

'''
Definitions of DS-CNN architectures for Keyword Spotting application
'''

def ds_cnn(input_shape, num_labels, dropout, input_stride=(2,2), first_dsc_stride=(1,1), channels=64, dsconv_blocks=4):
    model_layers = [
        layers.InputLayer(input_shape=input_shape),
        layers.BatchNormalization(),
    ]

    # Input Conv layer with distinct kernel shape
    model_layers.extend([
        layers.Conv2D(channels, (10,4), strides=input_stride, padding='same', name='input_conv'),
        layers.BatchNormalization(),
        layers.ReLU(),
    ])

    if dropout < 1.0 and dropout > 0.0:
        model_layers.append(layers.Dropout(dropout))

    if first_dsc_stride != (1,1):
        dsconv_blocks -= 1
        model_layers.extend([
            layers.DepthwiseConv2D(3, strides=first_dsc_stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, 1, 1),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        if dropout < 1.0 and dropout > 0.0:
            model_layers.append(layers.Dropout(dropout))

    for _ in range(dsconv_blocks):
        # The builtin Keras DSConv Operator does not have BatchNorm and ReLU
        # between the DS and PW Conv operations, so we build the DSConv block
        # by hand instead
        model_layers.extend([
            layers.DepthwiseConv2D(3, 1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, 1, 1),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        if dropout < 1.0 and dropout > 0.0:
            model_layers.append(layers.Dropout(dropout))

    model_layers.extend([
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dense(num_labels),
    ])
    return models.Sequential(model_layers)

'''
Pre-defined architectures from the 'Hello Edge' paper
'''
def ds_cnn_model(size, input_shape, num_labels, dropout=0.1):
    if size == 's':
        return ds_cnn(input_shape, num_labels, dropout)
    elif size == 'm':
        return ds_cnn(input_shape, num_labels, dropout, input_stride=(2,1), first_dsc_stride=(2,2), channels=172)
    elif size == 'l':
        return ds_cnn(input_shape, num_labels, dropout, input_stride=(2,1), first_dsc_stride=(2,2), channels=276, dsconv_blocks=5)
    raise ValueError("Unknown DS-CNN size. Supported size parameters: s,m,l")

def train(model, train_ds, val_ds, weights_path, epochs=50):
    epoch_size = train_ds.cardinality().numpy()
    lr = optimizers.schedules.PolynomialDecay(5e-3, epoch_size * epochs)

    save_checkpoint = callbacks.ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        mode='auto',
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[save_checkpoint],
    )
    return history
