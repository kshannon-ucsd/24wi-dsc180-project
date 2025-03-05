# data_loader.py
import os
import boto3
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# S3 Configuration
BUCKET_NAME = "sagemaker-capstone-bucket"
s3 = boto3.client("s3")

def preprocess_img(img, img_size=256, crop_size=224, is_train=True):
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.image.central_crop(img, crop_size / img_size)
    if is_train:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.25)
        img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
    img = preprocess_input(img)
    return img

def download_and_preprocess_data(local_raw_path, local_processed_path):
    os.makedirs(local_raw_path, exist_ok=True)
    os.makedirs(local_processed_path, exist_ok=True)
    datasets = {"train": [], "val": [], "test": []}
    
    for folder in datasets.keys():
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{folder}/")
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                if key.endswith(".jpeg"):
                    filename = key.split('/')[-1]
                    label = "0" if "normal" in key else "1"
                    datasets[folder].append((filename, label))
                    raw_file_path = os.path.join(local_raw_path, folder, filename)
                    os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
                    s3.download_file(BUCKET_NAME, key, raw_file_path)
                    img = tf.io.read_file(raw_file_path)
                    img = tf.image.decode_jpeg(img, channels=3)
                    img = preprocess_img(img, is_train=(folder == "train"))
                    processed_file_path = os.path.join(local_processed_path, folder, filename)
                    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
                    cv2.imwrite(processed_file_path, img.numpy())
    return datasets