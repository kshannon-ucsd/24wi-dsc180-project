from data_loader import download_and_preprocess_data
from model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import src.etl.python.subset_preproc
import src.models.model_2

import pandas as pd
import pickle

def main():
    local_raw_path = "./data/raw"
    local_processed_path = "./data/processed"
    datasets = download_and_preprocess_data(local_raw_path, local_processed_path)
    train_df = pd.DataFrame(datasets['train'], columns=['filename', 'binary_label'])
    val_df = pd.DataFrame(datasets['val'], columns=['filename', 'binary_label'])
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_dataframe(train_df, directory=f'{local_processed_path}/train', x_col='filename', y_col='binary_label', target_size=(224, 224), batch_size=16, class_mode='binary')
    val_generator = val_datagen.flow_from_dataframe(val_df, directory=f'{local_processed_path}/val', x_col='filename', y_col='binary_label', target_size=(224, 224), batch_size=16, class_mode='binary')
    model = build_model()
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ModelCheckpoint('best_model_1.h5', monitor='val_loss', save_best_only=True)]
    model.fit(train_generator, validation_data=val_generator, epochs=100, callbacks=callbacks)
    model.save('best_model_1.keras')

    ##second model
    src.etl.python.subset_prepoc.main()
    second_model = src.models.model_2.second_model()
    with open("best_second_model.pkl", "wb") as file:
        pickle.dump(second_model, file)


