import os
import re
import argparse

import cv2
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import *

z_dims = 64
h_dims = 256
n_filters = 128
image_shape = (64, 64, 3)

class Dataset(object):
    def __init__(self, face_dir, data_dir, image_shape):
        self.face_dir = face_dir
        self.data_dir = data_dir
        self.image_shape = image_shape
        self.size = -1

    def load_data(self):
        face_files = sorted([os.path.join(self.face_dir, f) for f in os.listdir(self.face_dir) if f.endswith('.jpg')])
        norm_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('_normal.png')])
        albd_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('_albedo.png')])
        mask_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('_mask.png')])
        sh_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('_sh.txt')])

        pat = re.compile('[0-9]+')
        valid_indices = [0] * len(norm_files)
        for i, f in enumerate(norm_files):
            mat = pat.search(f)
            if mat is None:
                raise Exception('Invalid file name:', f)

            valid_indices[i] = int(mat.group(0)) - 1

        face_files = [face_files[i] for i in valid_indices]

        print('Loading face files...')
        self.faces = [cv2.imread(f, cv2.IMREAD_COLOR) for f in face_files]
        self.faces = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.faces]
        self.faces = [(img / 255.0).astype(np.float32) for img in self.faces]
        self.faces = [cv2.resize(img, self.image_shape[:-1]) for img in self.faces]
        self.faces = np.stack(self.faces, axis=0)

        print('Loading normal files...')
        self.norms = [cv2.imread(f, cv2.IMREAD_COLOR) for f in norm_files]
        self.norms = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.norms]
        self.norms = [(img / 255.0).astype(np.float32) * 2.0 - 1.0 for img in self.norms]
        self.norms = [cv2.resize(img, self.image_shape[:-1]) for img in self.norms]
        self.norms = np.stack(self.norms, axis=0)

        print('Loading mask files...')
        self.masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]
        self.masks = [(img / 255.0).astype(np.float32) for img in self.masks]
        self.masks = [cv2.resize(img, self.image_shape[:-1]) for img in self.masks]
        self.masks = [img[:,:,np.newaxis] for img in self.masks]
        self.masks = np.stack(self.masks, axis=0)

        print('Loading albedo files...')
        self.albds = [cv2.imread(f, cv2.IMREAD_COLOR) for f in albd_files]
        self.albds = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.albds]
        self.albds = [(img / 255.0).astype(np.float32) for img in self.albds]
        self.albds = [cv2.resize(img, self.image_shape[:-1]) for img in self.albds]
        self.albds = np.stack(self.albds, axis=0)

        print('Making background files...')
        self.bg = [face * np.repeat(1.0 - mask, repeats=3, axis=-1) for face, mask in zip(self.faces, self.masks)]
        self.bg = np.stack(self.bg, axis=0)

        print('Loading SH files...')
        self.sh = np.stack([self.load_sh(f) for f in sh_files], axis=0)

        if len(self.faces) != len(self.norms):
            raise Exception('Data size does not match!!')

        if len(self.faces) != len(self.masks):
            raise Exception('Data size does not match!!')

        if len(self.faces) != len(self.bg):
            raise Exception('Data size does not match!!')

        if len(self.faces) != len(self.sh):
            raise Exception('Data size does not match!!')

        self.size = len(self.faces)

    def load_sh(self, filename):
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        L = np.zeros((9), dtype=np.float32)
        with open(filename, 'r') as f:
            lines = list(f)
            for i, l in enumerate(lines):
                L[i] = float(l.strip())

        sh = np.asarray([[c1 * L[8],  c1 * L[4], c1 * L[7], c2 * L[3]],
                         [c1 * L[4], -c1 * L[8], c1 * L[5], c2 * L[1]],
                         [c1 * L[7],  c1 * L[5], c3 * L[6], c2 * L[2]],
                         [c2 * L[3],  c2 * L[1], c2 * L[2], c4 * L[0] - c5 * L[6]]], dtype=np.float32)

        return sh

    def __len__(self):
        return self.size

class TestDataset(object):
    def __init__(self, test_dir, image_shape, num=10):
        self.test_dir = test_dir
        self.image_shape = image_shape
        self.num = num

    def load_data(self):
        test_files = sorted([os.path.join(self.test_dir, f) for f in os.listdir(self.test_dir) if f.endswith('.jpg')])
        test_files = test_files[:min(self.num, len(test_files))]
        self.images = [cv2.imread(f, cv2.IMREAD_COLOR) for f in test_files]
        self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images]
        self.images = [(img / 255.0).astype(np.float32) for img in self.images]
        self.images = [cv2.resize(img, self.image_shape[:-1]) for img in self.images]
        self.images = np.stack(self.images, axis=0)

def image_of_sh(sh):
    size = 64
    ix, iy = np.meshgrid(range(size), range(size))
    ix = ix / size * 2.0 - 1.0
    iy = iy / size * 2.0 - 1.0
    iz = np.sqrt(1.0 - np.minimum(1.0, ix * ix + iy * iy))

    norm = np.stack((ix, iy, iz, np.ones((size, size))), axis=2)
    norm = norm.reshape((size * size, 4))

    res = np.zeros((size * size))
    temp = np.dot(norm, sh[:, :])
    res = np.sum(norm * temp, axis=1)

    res = res.reshape((size, size))
    res = np.maximum(0.0, np.minimum(res, 1.0))

    return res

def save_images(outfile, gen, images):
    preds, norms, masks, bg, albedo, sh = gen.predict(images)

    n_images = len(images)
    fig = plt.figure(figsize=(8, 6))
    grid = gridspec.GridSpec(7, n_images, wspace=0.1, hspace=0.1)
    for i in range(n_images):
        # images
        ax = plt.Subplot(fig, grid[0 * n_images + i])
        ax.imshow(images[i], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

        # predictions
        ax = plt.Subplot(fig, grid[1 * n_images + i])
        ax.imshow(preds[i], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)


        # normals
        ax = plt.Subplot(fig, grid[2 * n_images + i])
        ax.imshow(norms[i] * 0.5 + 0.5, interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

        # masks
        ax = plt.Subplot(fig, grid[3 * n_images + i])
        ax.imshow(np.squeeze(masks[i], -1), cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

        # backgrounds
        ax = plt.Subplot(fig, grid[4 * n_images + i])
        ax.imshow(bg[i], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

        # albedos
        ax = plt.Subplot(fig, grid[5 * n_images + i])
        ax.imshow(albedo[i], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

        # SH
        ax = plt.Subplot(fig, grid[6 * n_images + i])
        ax.imshow(image_of_sh(sh[i]), cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)


    print('Saved to:', outfile)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def progress_bar(x, maxval, width=40):
    tick = int(x / maxval * width)
    tick = min(tick, width)

    if tick == width:
        return '=' * tick

    return '=' * tick + '>' + ' ' * (width - tick - 1)

def main():
    parser = argparse.ArgumentParser(description='Keras face intrinsic')
    parser.add_argument('--faces', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', default='result')
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--batchsize', '-B', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--memlim', type=float, default=1.0)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print('Use GPU #%d' % (args.gpu))

    if args.memlim < 1.0:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.memlim
        set_session(tf.Session(config=config))

    model = FaceIntrinGAN(
        z_dims=z_dims,
        h_dims=h_dims,
        n_filters=n_filters,
        image_shape=image_shape
    )
    dataset = Dataset(args.faces, args.data, image_shape)
    dataset.load_data()

    test_data = TestDataset(args.test, image_shape)
    test_data.load_data()

    data_size = len(dataset)
    for e in range(args.epoch):
        perm = np.random.permutation(data_size)

        print('Epoch #%d' % (e))
        for b in range(0, data_size, args.batchsize):
            # Batch data
            batchsize = min(args.batchsize, data_size - b)
            indx = perm[b:b+batchsize]

            face_batch = dataset.faces[indx]
            norm_batch = dataset.norms[indx]
            mask_batch = dataset.masks[indx]
            albd_batch = dataset.albds[indx]
            bg_batch = dataset.bg[indx]
            sh_batch = dataset.sh[indx]

            batch = [face_batch, norm_batch, mask_batch, albd_batch, sh_batch, bg_batch]

            # Train AE
            ae_loss = model.ae_trainer.train_on_batch(batch, face_batch)
            face_fake_batch = model.autoencoder.predict(face_batch)

            # Train generator
            true_batch = np.zeros((batchsize), dtype='int32')
            true_batch = keras.utils.to_categorical(true_batch, 2)
            fake_batch = np.ones((batchsize), dtype='int32')
            fake_batch = keras.utils.to_categorical(fake_batch, 2)
            gen_loss = model.gen_trainer.train_on_batch(face_batch, true_batch)

            # Train discriminator
            dis_loss_true = model.dis_trainer.train_on_batch(face_batch, true_batch)
            dis_loss_fake = model.dis_trainer.train_on_batch(face_fake_batch, fake_batch)
            dis_loss = (dis_loss_true + dis_loss_fake) * 0.5

            # Progress
            ratio = 100.0 * (b + batchsize) / data_size
            print('Epoch #%d | %6.2f %% [%6d / %6d] | ae_loss: %.6f | g_loss: %.6f | d_loss: %.6f' % \
                  (e + 1, ratio, b + batchsize, data_size, ae_loss, gen_loss, dis_loss), end='\r')

            # Save image
            if (b + batchsize) == data_size or (b != 0 and b % 10000 == 0):
                outfile = os.path.join(args.output, 'epoch_%03d_%06d' % (e, b))
                save_images(outfile, model, test_data.images)

        print('')

if __name__ == '__main__':
    main()
