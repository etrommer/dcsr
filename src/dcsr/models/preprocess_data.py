import tensorflow as tf


def get_spectrogram(waveform, num_mel_bins=40, num_coeffs=10):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=640, frame_step=320)

    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.square(spectrogram)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz = 20.0, 4000.0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz, upper_edge_hertz
    )
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 10.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_coeffs]
    return tf.expand_dims(mfccs, -1)


def cifar10(img, im_size, augment=False):
    img = tf.cast(img, tf.float32)
    img /= 255.0

    if augment is True:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, 0.1)
        img = tf.image.random_brightness(img, 0.2)

    img = tf.image.resize(img, (im_size, im_size))

    return img
