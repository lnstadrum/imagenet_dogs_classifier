from sigmoid_like import sigmoid_like
import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.initializers import GlorotUniform
from beatmup_keras import Shuffle

BATCH_NORM_MOMENTUM = 0.75


def conv2d(x, filters, size, name, groups=1, strides=1, padding='valid', activation=True, batch_norm=True):
    """ Conv2D + Batch Normalization + Activation pattern
    """
    assert size * size * x.shape[3] // 4 // groups < 64, "Kernel too big to fit Pi GPU"
    assert (filters // groups) % 4 == 0, "Grouping is not aligned with 4-channel blocks"
    x = Conv2D(filters,
               size,
               groups=groups,
               name=name,
               use_bias=not batch_norm,
               strides=strides,
               padding=padding,
               kernel_initializer=GlorotUniform(),
               bias_initializer=GlorotUniform())(x)
    if batch_norm:
        x = BatchNormalization(name=name+'-bn', momentum=BATCH_NORM_MOMENTUM)(x)
    if activation:
        return Activation(sigmoid_like, name=name+'-act')(x)
    return x


def regular_unit(input, name, shuffle=False):
    """ Regular unit: optional Shuffle + grouped separable convolution + residual connection
    """
    depth = input.shape[3]
    
    # add shuffle
    x = Shuffle()(input) if shuffle else input

    # add pointwise convolution
    x = conv2d(x,
               filters=depth,
               size=1,
               groups=depth // 32 if depth > 32 else 1,
               name=name+'-pw',
               padding='same')

    # add depthwise convolution
    x = conv2d(x,
               filters=depth,
               size=3,
               groups=depth//4,
               name=name+'-dw',
               padding='same',
               activation=False)

    # add residual connection
    x = Add(name=name+'-add')([x, input])

    # add activation function
    return Activation(sigmoid_like, name=name+'-act')(x)


def downsmp_unit(x, name, depth, size=3):
    """ Downsampling unit: 2D group convolution
    """
    return conv2d(x,
                  filters=depth,
                  size=size,
                  groups=x.shape[3]//8,
                  strides=2,
                  name=name,
                  padding='valid')


def make_model(input_size, num_classes=120):
    input = Input(shape=(input_size, input_size, 3), name='input')
    x = input
    
    # stage 1
    x = conv2d(x, 32, 5, name='b0-conv', strides=2)    # 191x191 from now on
    x = regular_unit(x, 'b1-stage1')
    x = regular_unit(x, 'b1-stage2')
    x = downsmp_unit(x, 'b1-scale', 32)    # 95x95

    # stage 2
    x = regular_unit(x, 'b2-stage1', True)
    x = regular_unit(x, 'b2-stage2', True)
    x = regular_unit(x, 'b2-stage3', True)
    x = downsmp_unit(x, 'b2-scale', 64)    # 47x47

    # stage 3
    x = regular_unit(x, 'b3-stage1', True)
    x = regular_unit(x, 'b3-stage2', True)
    x = regular_unit(x, 'b3-stage3', True)
    x = downsmp_unit(x, 'b3-scale', 128)    # 23x23

    # stage 4
    x = regular_unit(x, 'b4-stage1', True)
    x = regular_unit(x, 'b4-stage2', True)
    x = regular_unit(x, 'b4-stage3', True)
    x = regular_unit(x, 'b4-stage4', True)
    x = downsmp_unit(x, 'b4-scale', 192)    # 11x11

    # stage 5
    x = regular_unit(x, 'b5-stage1', True)
    x = regular_unit(x, 'b5-stage2', True)
    x = regular_unit(x, 'b5-stage3', True)
    x = regular_unit(x, 'b5-stage4', True)
    x = regular_unit(x, 'b5-stage5', True)
    x = downsmp_unit(x, 'b5-scale', 192)    # 5x5

    # final stage
    x = conv2d(x, 192, 3, groups=192 // 8, name='b6-conv')
    x = GlobalAveragePooling2D(name='b7-pool')(x)
    x = Flatten(name='flatten')(x)
    output = Dense(num_classes,
                   name='Dense',
                   use_bias=True)(x)

    # build the model
    model = tf.keras.models.Model(inputs=input, outputs=output)

    # compile
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    return model
