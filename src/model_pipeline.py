import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# 1. DANH SÁCH NHÃN CHUẨN (Khớp với Colab)
class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
translate = {
    "cane": "CHÓ", "cavallo": "NGỰA", "elefante": "VOI", "farfalla": "BƯỚM", 
    "gallina": "GÀ", "gatto": "MÈO", "mucca": "BÒ", "pecora": "CỪU", 
    "ragno": "NHỆN", "scoiattolo": "SÓC"
}

def build_extractor(weights_path="models/ultimate_cnn.weights.h5"):
    inputs = layers.Input(shape=(128, 128, 3))
    x = layers.RandomFlip()(inputs) 
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights=None)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    feature_layer = layers.Dense(256, activation='relu', name='feature_layer')(x)
    outputs = layers.Dense(10, activation='softmax')(feature_layer)
    
    full_model = models.Model(inputs, outputs)
    full_model.load_weights(weights_path) 
    return models.Model(inputs=full_model.input, outputs=full_model.get_layer('feature_layer').output)

class AnimalPredictor:
    def __init__(self, cnn_weights="models/ultimate_cnn.weights.h5", 
                 scaler_path="models/ultimate_scaler.pkl", 
                 xgb_path="models/ultimate_xgb.pkl"):
        print("⏳ Đang khởi động AI Pipeline...")
        self.extractor = build_extractor(cnn_weights)
        self.scaler = joblib.load(scaler_path)
        self.xgb_model = joblib.load(xgb_path)

    def predict(self, image_path):
        # Tiền xử lý ảnh
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_AREA)
        img_final = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)

        # Trích xuất đặc trưng và dự đoán qua XGBoost
        raw_feat = self.extractor.predict(img_final, verbose=0)
        scaled_feat = self.scaler.transform(raw_feat)
        pred_idx = self.xgb_model.predict(scaled_feat)[0]
        probs = self.xgb_model.predict_proba(scaled_feat)
        conf = np.max(probs)

        animal_en = class_names[int(pred_idx)]
        animal_vn = translate.get(animal_en, animal_en)
        return animal_vn, conf