# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

def build_model():
    resnet50 = ResNet50V2(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3))
    model = Sequential([
        resnet50,
        Dense(256), BatchNormalization(), Activation('relu'), Dropout(0.5),
        Dense(128), Activation('relu'),
        Dense(1, activation="sigmoid")
    ])
    lr_schedule = CosineDecayRestarts(initial_learning_rate=0.0001, first_decay_steps=5*326, t_mul=2.0, m_mul=0.8, alpha=1e-6)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model