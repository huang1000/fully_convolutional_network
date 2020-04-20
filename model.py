import tensorflow as tf

def conv_blk(x, filters, dropout_rate=0.2):
    """
    filters: [#_filter, kernel_size, stride]
    """
    
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=filters[1], strides=filters[2])(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def FCN_model(len_classes=5, dropout_rate=0.2):
    
    input = tf.keras.layers.Input(shape=(None, None, 3))

    x = conv_blk(input, [32, 3, 1], dropout_rate)

    # x = tf.keras.layers.MaxPooling2D()(x)

    x = conv_blk(x, [64, 3, 1], dropout_rate)

    # x = tf.keras.layers.MaxPooling2D()(x)

    x = conv_blk(x, [128, 3, 2], dropout_rate)

    # x = tf.keras.layers.MaxPooling2D()(x)

    x = conv_blk(x, [256, 3, 2], dropout_rate)

    # x = tf.keras.layers.MaxPooling2D()(x)

    x = conv_blk(x, [512, 3, 2], dropout_rate)

    # Uncomment the below line if you're using dense layers
    # x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Fully connected layer 1
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(units=64)(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 1
    x = conv_blk(x, [64, 1, 1], dropout_rate)

    # Fully connected layer 2
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(units=len_classes)(x)
    # predictions = tf.keras.layers.Activation('softmax')(x)

    # Fully connected layer 2
    x = tf.keras.layers.Conv2D(filters=len_classes, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    predictions = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=predictions)
    
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')

    return model

if __name__ == "__main__":
    FCN_model(len_classes=5, dropout_rate=0.2)
    
