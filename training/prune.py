import tensorflow_model_optimization as tfmot

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import models

def prunable_layer(layer, pruning_params, retrain_dense=False, blacklist=[]):
    if layer.name in blacklist:
        layer.trainable = retrain_dense
        return layer
    if isinstance(layer, layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, layers.Conv2D) and layer.kernel_size == (1,1):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    layer.trainable = retrain_dense
    return layer

def prunable_model(dense_model, pruning_params):
    cf = lambda layer: prunable_layer(layer, pruning_params)
    model_for_pruning = models.clone_model(
        dense_model,
        clone_function=cf,
    )
    return model_for_pruning

def prune(dense_model, train_ds, val_ds, pruning_epochs=40, retraining_epochs=20, target_sparsity=0.75):
    epoch_size = train_ds.cardinality().numpy()
    lr = optimizers.schedules.PolynomialDecay(5e-3, epoch_size * (pruning_epochs + retraining_epochs))

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=target_sparsity/2,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=pruning_epochs*epoch_size
        )
    }

    pm = prunable_model(dense_model, pruning_params)
    pm.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    pm.summary()

    history = pm.fit(
        train_ds,
        validation_data=val_ds,
        epochs=pruning_epochs+retraining_epochs,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep()
        ],
    )
    return tfmot.sparsity.keras.strip_pruning(pm), history
