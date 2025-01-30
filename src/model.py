import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_pretrained_model(
    base_model_name='EfficientNetB0',
    input_shape=(224, 224, 3),
    learning_rate=1e-5,  # Reduced learning rate for transfer learning
    freeze_backbone=True
):
    """
    Builds a transfer-learning model using a pretrained backbone (e.g. EfficientNetB0).
    """
    # 1) Pick a pretrained model from Keras applications
    if base_model_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    # You could add more elif blocks for ResNet, MobileNet, etc.

    # 2) Decide if you want to freeze the base CNN
    base_model.trainable = not freeze_backbone

    # 3) Build a new "head" with stronger regularization
    inputs = tf.keras.Input(shape=input_shape, name='image_input')
    x = base_model(inputs, training=False)        # pass input through pretrained
    x = layers.GlobalAveragePooling2D()(x)        # condense spatial dims
    x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)  # Added intermediate layer
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)                    # Increased dropout
    outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

    
def build_cnn_model(input_shape=(224, 224, 1), learning_rate=0.001,
               conv_filters=[32, 64, 128], dense_units=64,
               dropout_rates=[0.3, 0.4, 0.5], l2_reg=0.03):
    """
    Build and compile a CNN model optimized for small datasets with strong regularization.
    Uses multiple techniques to prevent overfitting:
    - L2 regularization on all layers
    - Increased dropout rates
    - Batch normalization
    - Reduced model complexity
    - Skip connections for better gradient flow
    
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_positions (int): Number of position categories.
        learning_rate (float): Learning rate for the Adam optimizer.
        conv_filters (list): List of filter sizes for conv layers.
        dense_units (int): Number of units in dense layers.
        dropout_rates (list): Dropout rates for different layers.
        l2_reg (float): L2 regularization factor.
    
    Returns:
        tf.keras.Model: Compiled CNN model optimized for binary classification.
    """
    # Single input for images
    inputs = tf.keras.Input(shape=input_shape, name='image_input')
    
    # Initial normalization and input preprocessing
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.GaussianNoise(0.01)(x)  # Add slight noise for regularization
    
    # Simplified Convolutional Blocks with residual connections
    for filters, dropout_rate in zip(conv_filters, dropout_rates):
        # Store input for residual connection
        block_input = x
        
        # Conv block
        x = Conv2D(filters, (3, 3), padding='same', activation='relu',
                  kernel_regularizer=l2(l2_reg),
                  kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu',
                  kernel_regularizer=l2(l2_reg),
                  kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        
        # Add residual connection if shapes match
        if block_input.shape[-1] == filters:
            x = tf.keras.layers.Add()([x, block_input])
        
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout_rate)(x)
    
    # Global Average Pooling instead of Flatten
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Single Dense layer with strong regularization
    x = Dense(dense_units, activation='relu',
              kernel_regularizer=l2(l2_reg),
              kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[-1])(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid',
                   kernel_regularizer=l2(l2_reg))(x)
    
    # Create and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use AMSGrad variant of Adam optimizer for better convergence
    optimizer = Adam(learning_rate=learning_rate, amsgrad=True)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model