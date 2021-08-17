import matplotlib.pyplot as plt
import mlflow

'''
MLFlow logging helpers
'''

# Plot loss/accuracy to file using matplotlib
def plot_history(history, filename='training_history.pdf'):
    plt.figure(figsize=(10,4))
    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Training', 'Validation'])

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Training', 'Validation'])

    plt.tight_layout()
    plt.savefig(filename)

def log_history(history):
    mlflow.log_metric('Training Loss', history.history['loss'][-1])
    mlflow.log_metric('Training Accuracy', history.history['accuracy'][-1])
    mlflow.log_metric('Validation Loss', history.history['val_loss'][-1])
    mlflow.log_metric('Validation Accuracy', history.history['val_accuracy'][-1])

    plot_history(history)
    mlflow.log_artifact('training_history.pdf')
