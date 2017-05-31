import keras
from keras.models import *
from keras.layers import *
from keras import backend as K

def extract_dim(d):
    def _extract_dim(x):
        return x[:, :, d]

    return _extract_dim

def dot_prod(x0):
    def _dot_prod(x1):
        return K.batch_dot(x0, x1)

    return _dot_prod

def sum_channel(x):
    return K.sum(x, axis=-1)

def set_trainable(model, train):
    model.trainable = train
    for l in model.layers:
        l.trainable = train

def image_recon_layer(image_shape):
    # Input
    norm = Input(image_shape)
    mask = Input(image_shape[:-1])
    albedo = Input(image_shape)
    sh = Input((4, 4))
    bg = Input(image_shape)

    # Forwarding
    image_w = image_shape[1]
    image_h = image_shape[0]

    norm_reshape = Reshape((image_w * image_h, 3))(norm)

    ones_vect = Lambda(lambda x: K.ones_like(x))(norm_reshape)
    ones_vect = Lambda(extract_dim(0))(ones_vect)
    ones_vect = Reshape((-1, 1))(ones_vect)
    norm_reshape = Concatenate(axis=-1)([norm_reshape, ones_vect])

    shade = Lambda(lambda x: K.batch_dot(x[0], x[1]))([norm_reshape, sh])
    shade = Multiply()([shade, norm_reshape])
    shade = Lambda(lambda x: K.sum(x, axis=-1))(shade)
    shade = Reshape((image_w * image_h, 1))(shade)

    mask_reshape = Reshape((image_w * image_h, 1))(mask)
    mask_reshape = Concatenate(axis=-1)([mask_reshape, mask_reshape, mask_reshape])

    shade = Concatenate(axis=-1)([shade, shade, shade])
    shade = Multiply()([shade, mask_reshape])
    shade = Reshape((image_h, image_w, 3))(shade)
    fg = Multiply()([shade, albedo])

    rev_mask_reshape = Lambda(lambda x: K.ones_like(x) - x)(mask_reshape)
    bg_masked = Reshape((image_w * image_h, 3))(bg)
    bg_masked = Multiply()([bg_masked, rev_mask_reshape])
    bg_masked = Reshape((image_h, image_w, 3))(bg_masked)

    recon = Add()([fg, bg_masked])

    return Model(inputs=[norm, mask, bg, albedo, sh], outputs=recon)

class AutoencoderLoss(object):
    __name__ = 'autoencoder_loss'

    def __init__(self, in_img, in_norm, in_mask, in_albd, in_sh, in_bg,
                       out_img, out_norm, out_mask, out_albd, out_sh, out_bg):
        self.in_img = in_img
        self.in_norm = in_norm
        self.in_mask = in_mask
        self.in_albd = in_albd
        self.in_sh = in_sh
        self.in_bg = in_bg

        self.out_img = out_img
        self.out_norm = out_norm
        self.out_mask = out_mask
        self.out_albd = out_albd
        self.out_sh = out_sh
        self.out_bg = out_bg

    def __call__(self, x_true, x_pred):
        norm_loss = K.mean(K.abs(self.in_norm - self.out_norm), axis=[1, 2, 3])
        mask_loss = K.mean(K.abs(self.in_mask - self.out_mask), axis=[1, 2, 3])
        albd_loss = K.mean(K.abs(self.in_albd - self.out_albd), axis=[1, 2, 3])
        sh_loss = K.mean(K.abs(self.in_sh - self.out_sh), axis=[1, 2]) * (16 * 16) * 0.1
        bg_loss = K.mean(K.abs(self.in_bg - self.out_bg), axis=[1, 2, 3])
        img_loss = K.mean(K.square(self.in_img - self.out_img), axis=[1, 2, 3]) * 20.0
        loss = norm_loss + mask_loss + albd_loss + sh_loss + bg_loss + img_loss
        return loss

class FaceIntrinGAN(object):
    def __init__(
        self,
        z_dims=50,
        h_dims=1024,
        n_filters=128,
        image_shape=(96, 96, 3)):

        self.z_dims = z_dims
        self.h_dims = h_dims
        self.n_filters = n_filters
        self.image_shape = image_shape
        self._build_model()

    def _build_model(self):
        self._build_generator()
        self._build_discriminator()

        inputs = Input(self.image_shape)

        in_norm = Input(self.image_shape)
        in_mask = Input(self.image_shape[:-1] + (1,))
        in_albd = Input(self.image_shape)
        in_sh = Input((4, 4))
        in_bg = Input(self.image_shape)

        recon, out_norm, out_mask, out_bg, out_albd, out_sh = self.generator(inputs)

        self.ae_loss = AutoencoderLoss(inputs, in_norm, in_mask, in_albd, in_sh, in_bg,
                                       recon, out_norm, out_mask, out_albd, out_sh, out_bg)

        self.ae_trainer = Model(inputs=[inputs, in_norm, in_mask, in_albd, in_sh, in_bg],
                                outputs=[recon])

        self.ae_optim = keras.optimizers.Adam(lr=2.0e-4, beta_1=0.5)
        self.ae_trainer.compile(loss=self.ae_loss, optimizer=self.ae_optim)

        y_fake = self.discriminator(recon)
        set_trainable(self.discriminator, False)
        set_trainable(self.generator, True)
        self.gen_trainer = Model(inputs, y_fake)
        self.gen_optim = keras.optimizers.Adam(lr=2.0e-4, beta_1=0.5)
        self.gen_trainer.compile(loss=keras.losses.binary_crossentropy, optimizer=self.gen_optim)

        y_true = self.discriminator(inputs)
        set_trainable(self.discriminator, True)
        set_trainable(self.generator, False)
        self.dis_trainer = Model(inputs, y_true)
        self.dis_optim = keras.optimizers.Adam(lr=2.0e-4, beta_1=0.5)
        self.dis_trainer.compile(loss=keras.losses.binary_crossentropy, optimizer=self.dis_optim)

    def predict(self, image):
        norm = self.norm_gen.predict(image)
        mask = self.mask_gen.predict(image)
        bg = self.bg_gen.predict(image)
        albedo = self.albedo_gen.predict(image)
        sh = self.sh_gen.predict(image)
        image = self.autoencoder.predict(image)
        return image, norm, mask, bg, albedo, sh

    def _build_discriminator(self):
        self.discriminator = self.build_encoder(self.image_shape, 2)

    def _build_generator(self):
        self.enc = self.build_encoder(self.image_shape, self.h_dims * 5)
        self.norm_dec = self.build_decoder(self.h_dims, 3, last_activation='tanh')
        self.mask_dec = self.build_decoder(self.h_dims, 1)
        self.bg_dec = self.build_decoder(self.h_dims, 3)
        self.albedo_dec = self.build_decoder(self.h_dims, 3)
        self.sh_dec = self.build_sh_decoder(self.h_dims)

        inputs = Input(self.image_shape)
        h1_all = self.enc(inputs)
        h1_all = Reshape((self.h_dims, 5))(h1_all)

        h1_norm = Lambda(extract_dim(0))(h1_all)
        h1_mask = Lambda(extract_dim(1))(h1_all)
        h1_bg = Lambda(extract_dim(2))(h1_all)
        h1_albd = Lambda(extract_dim(3))(h1_all)
        h1_sh = Lambda(extract_dim(4))(h1_all)

        h2_norm = self.norm_dec(h1_norm)
        h2_mask = self.mask_dec(h1_mask)
        h2_bg = self.bg_dec(h1_bg)
        h2_albd = self.albedo_dec(h1_albd)
        h2_sh = self.sh_dec(h1_sh)

        self.norm_gen = Model(inputs, h2_norm)
        self.mask_gen = Model(inputs, h2_mask)
        self.bg_gen = Model(inputs, h2_bg)
        self.albedo_gen = Model(inputs, h2_albd)
        self.sh_gen = Model(inputs, h2_sh)

        h3_recon = image_recon_layer(self.image_shape)([h2_norm, h2_mask, h2_bg, h2_albd, h2_sh])
        self.autoencoder = Model(inputs, h3_recon)

        self.generator = Model(inputs=[inputs], outputs=[h3_recon, h2_norm, h2_mask, h2_bg, h2_albd, h2_sh])
        self.generator.summary()

    def build_decoder(self, input_dims, output_dims, last_activation='sigmoid'):
        inputs = Input(shape=(input_dims,))

        # Fully connected layer
        x = Dense(8 * 8 * self.n_filters)(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Reshape((8, 8, self.n_filters))(x)

        activation = 'relu'
        x = self.basic_decode_layer(x, activation)
        x = self.basic_decode_layer(x, activation)
        x = self.basic_decode_layer(x, activation)
        x = self.basic_decode_layer(x, activation, upsample=False)

        x = Convolution2D(filters=output_dims, kernel_size=(5, 5), padding='same')(x)
        x = Activation(last_activation)(x)
        return Model(inputs, x)

    def build_encoder(self, input_shape, output_dims):
        inputs = Input(shape=input_shape)

        # Basic layers
        activation = 'leaky_relu'
        x = self.basic_encode_layer(inputs, activation)
        x = self.basic_encode_layer(x, activation)
        x = self.basic_encode_layer(x, activation)
        x = self.basic_encode_layer(x, activation)

        # Fully connected layer
        x = Flatten()(x)
        x = Dense(output_dims * 2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(output_dims)(x)

        return Model(inputs, x)

    def build_sh_decoder(self, input_dims):
        inputs = Input(shape=(input_dims,))
        x = Dense(4 * 4)(inputs)
        x = Reshape((4, 4))(x)

        return Model(inputs, x)

    def basic_decode_layer(self, x, activation='relu', upsample=True):
        x = Conv2D(filters=self.n_filters, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        x = Conv2D(filters=self.n_filters, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        if upsample:
            x = UpSampling2D(size=(2, 2))(x)

        return x

    def basic_encode_layer(self, x, activation='elu'):
        x = Conv2D(filters=self.n_filters, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU()(x)
        else:
            x = Activation(activation)(x)

        x = Convolution2D(filters=self.n_filters, kernel_size=(5, 5), padding='same')(x)
        x = AveragePooling2D()(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU()(x)
        else:
            x = Activation(activation)(x)

        return x
