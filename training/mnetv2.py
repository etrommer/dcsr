import os
import re
import tempfile

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import applications


# C&P from StackOverflow to insert Dropout in the model
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer._name = insert_layer_name
            else:
                new_layer._name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return models.Model(inputs=model.inputs, outputs=model_outputs)

def model(width, dropout, num_classes, all_trainable=True, im_size=96):

    # Get pre-trained model
    model = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(im_size, im_size, 3),
        pooling='max',
        alpha=width,
    )

    for layer in model.layers:
        layer.trainable = all_trainable

    # Insert Dropout
    def dropout_layer_factory():
        return layers.Dropout(rate=dropout, name='dropout')
    # Save and load to resolve potential model issues
    model = insert_layer_nonseq(model, '.*_add', dropout_layer_factory)
    tmp_path = os.path.join(tempfile.gettempdir(), 'temp.h5')
    model.save(tmp_path)
    model = models.load_model(tmp_path)

    # Add FC layer
    x = model.layers[-1].output
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(num_classes)(x)
    model = models.Model(inputs = model.input, outputs = predictions) 

    return model

def train(model, train_ds, val_ds, epochs):
    epoch_size = train_ds.cardinality().numpy()
    lr = optimizers.schedules.PolynomialDecay(1e-4, epoch_size * epochs, end_learning_rate=1e-6)

    early_stopping = callbacks.EarlyStopping(
            verbose=2,
            patience=5,
            monitor='val_accuracy',
            restore_best_weights=True)

    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr),
        loss = losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping])

    return history
