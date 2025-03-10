import tensorflow as tf

def pneumonia_detection_model(train_generator, val_generator):
    
    resnet50 = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3))
    
    model = tf.keras.models.Sequential([
        resnet50,
        tf.keras.layers.Dense(256), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128), 
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.0001, first_decay_steps=5*326, t_mul=2.0, m_mul=0.8, alpha=1e-6)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint('best_model_1.h5', monitor='val_loss', save_best_only=True)]
    
    model.fit(train_generator, validation_data=val_generator, epochs=100, callbacks=callbacks)
    
    # model.save('best_model_1.keras')
    
    return model