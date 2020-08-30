import tensorflow as tf
import numpy as np
from models import VoxelMorph1
from dataloader import RegistrationDataLoader
from metrics import local_normalized_cross_correlation_loss, gradient_loss
from viz import overlay_slices
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', required=False, default=60,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', required=False, default=1,
                        help='Batch size')
    parser.add_argument('--lr', required=False, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lamda', required=False, default=1.0,
                        help='Regularization parameter (lambda)')
    args = parser.parse_args()

    # Load data
    data = np.load('data/t1_moving_128.npy')
    static = np.load('data/t1_static_128.npy')
    input_shape = (128, 128, 128)

    x_train = data[:125, ...][..., None]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = data[125:-1, ...][..., None]
    x_test = x_test.astype(np.float32) / 255.0
    x_sample = x_test[:5, ...].copy()  # some images to visualize results

    static = static[None, ..., None]
    static = static.astype(np.float32) / 255.0

    print('Train: ', x_train.shape)
    print('Test: ', x_test.shape)
    print('Sample: ', x_sample.shape)
    print('Static: ', static.shape)

    # Create the data loader objects
    train_loader = RegistrationDataLoader(x_train, static,
                                          batch_size=args.batch_size,
                                          shuffle=True)
    test_loader = RegistrationDataLoader(x_test, static,
                                         batch_size=args.batch_size,
                                         shuffle=True)
    sample_loader = RegistrationDataLoader(x_sample, static, shuffle=True)

    # Instantiate and compile the model with the loss and the optimizer.
    ncc_loss = local_normalized_cross_correlation_loss()
    grad_loss = gradient_loss()

    # Map model outputs to the loss functions and loss weights
    loss_weights = {'moved': 1.0, 'deformation': args.lamda}
    losses = {'moved': ncc_loss, 'deformation': grad_loss}

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model = VoxelMorph1(input_shape=input_shape, optimizer=optimizer,
                        loss=losses, loss_weights=loss_weights)

    # Training
    hist = model.fit(train_loader, epochs=args.epochs,
                     validation_data=test_loader)

    # Compute the moved images for the sample set
    output = model.predict(sample_loader)
    moved = output['moved']
    deformation = output['deformation']

    # Visualize the transformed image together with the static
    # and the moving images.
    moved = moved.squeeze(axis=-1)  # Remove the channel dim.
    moved = moved * 255.0  # Rescale to [0, 255].
    moved = moved.astype(np.uint8)  # Convert back to 8-bit images.

    moving = x_sample.copy().squeeze(axis=-1)  # shape (num_samples, 32, 32)
    moving = moving * 255.0
    moving = moving.astype(np.uint8)

    static_ = static.squeeze(axis=-1)  # shape (1, 32, 32)
    static_ = np.repeat(static_, repeats=moving.shape[0], axis=0)
    static_ = static_*255.0
    static_ = static_.astype(np.uint8)

    # The images are flipped, so flipping them to the usual style
    moving = moving.transpose(0, 1, 3, 2)
    moving = np.flip(moving, (1, 3))
    moved = moved.transpose(0, 1, 3, 2)
    moved = np.flip(moved, (1, 3))
    static_ = static_.transpose(0, 1, 3, 2)
    static_ = np.flip(static_, (1, 3))

    # Plot images.
    for i in range(moving.shape[0]):
        overlay_slices(moving[i], static_[i], moved[i], None,
                       'Moving', "Static", "Moved", "%d.png" % (i))


