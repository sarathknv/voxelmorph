import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import grid_sample_3d, regular_grid_3d


class VoxelMorph1(object):
    def __init__(self, input_shape=(32, 32, 1), optimizer='adam', loss=None,
                 metrics=None, loss_weights=None):
        in_channels = 1
        out_channels = 3
        input_shape = input_shape + (in_channels,)

        moving = layers.Input(shape=input_shape, name='moving')
        static = layers.Input(shape=input_shape, name='static')

        x_in = layers.concatenate([static, moving], axis=-1)

        # encoder
        x1 = layers.Conv3D(16, kernel_size=3, strides=2, padding='same',
                           kernel_initializer='he_normal')(x_in)
        x1 = layers.LeakyReLU(alpha=0.2)(x1)  # 16

        x2 = layers.Conv3D(32, kernel_size=3, strides=2, padding='same',
                           kernel_initializer='he_normal')(x1)
        x2 = layers.LeakyReLU(alpha=0.2)(x2)  # 8

        x3 = layers.Conv3D(32, kernel_size=3, strides=2, padding='same',
                           kernel_initializer='he_normal')(x2)
        x3 = layers.LeakyReLU(alpha=0.2)(x3)  # 4

        x4 = layers.Conv3D(32, kernel_size=3, strides=2, padding='same',
                           kernel_initializer='he_normal')(x3)
        x4 = layers.LeakyReLU(alpha=0.2)(x4)  # 2

        # decoder [32, 32, 32, 32, 8, 8]
        x = layers.Conv3D(32, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_normal')(x4)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling3D(size=2)(x)  # 4
        x = layers.concatenate([x, x3], axis=-1)  # 4

        x = layers.Conv3D(32, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling3D(size=2)(x)  # 8
        x = layers.concatenate([x, x2], axis=-1)  # 8

        x = layers.Conv3D(32, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.UpSampling3D(size=2)(x)  # 16
        x = layers.concatenate([x, x1], axis=-1)  # 16

        x = layers.Conv3D(32, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv3D(8, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)  # 16

        x = layers.UpSampling3D(size=2)(x)  # 32
        x = layers.concatenate([x, x_in], axis=-1)
        x = layers.Conv3D(8, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_normal')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)  # 32

        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                                stddev=1e-5)
        deformation = layers.Conv3D(out_channels, kernel_size=3, strides=1,
                                    padding='same',
                                    kernel_initializer=kernel_initializer,
                                    name='deformation')(x)

        nb, nd, nh, nw, nc = tf.shape(deformation)

        # Regular grid.
        grid = regular_grid_3d(nd, nh, nw)  # shape (D, H, W, 2)
        grid = tf.expand_dims(grid, axis=0)  # shape (1, D, H, W, 2)
        multiples = tf.stack([nb, 1, 1, 1, 1])
        grid = tf.tile(grid, multiples)

        # Compute the new sampling grid.
        grid_new = grid + deformation
        grid_new = tf.clip_by_value(grid_new, -1, 1)

        # Sample the moving image using the new sampling grid.
        moved = grid_sample_3d(moving, grid_new, name='moved')

        model = tf.keras.Model(inputs={'moving': moving, 'static': static},
                               outputs={'moved': moved,
                                        'deformation': deformation},
                               name='voxelmorph1')
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                      loss_weights=loss_weights)

        self.model = model

    def compile(self, optimizer='adam', loss=None, metrics=None,
                loss_weights=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                           loss_weights=loss_weights)

    def summary(self):
        return self.model.summary()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None,
            validation_freq=1, max_queue_size=10, workers=1,
            use_multiprocessing=False):
        return self.model.fit(x=x, y=y, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data, shuffle=shuffle,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_batch_size=validation_batch_size,
                              validation_freq=validation_freq,
                              max_queue_size=max_queue_size, workers=workers,
                              use_multiprocessing=use_multiprocessing)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1,
                 steps=None, callbacks=None, max_queue_size=10, workers=1,
                 use_multiprocessing=False, return_dict=False):
        return self.model.evaluate(x=x, y=y, batch_size=batch_size,
                                   verbose=verbose, steps=steps,
                                   callbacks=callbacks,
                                   max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=return_dict)

    def predict(self, x, batch_size=None, verbose=0,
                steps=None, callbacks=None, max_queue_size=10, workers=1,
                use_multiprocessing=False):
        return self.model.predict(x=x, batch_size=batch_size,
                                  verbose=verbose, steps=steps,
                                  callbacks=callbacks,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def save_weights(self, filepath, overwrite=True):
        self.model.save_weights(filepath=filepath, overwrite=overwrite,
                                save_format=None)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
