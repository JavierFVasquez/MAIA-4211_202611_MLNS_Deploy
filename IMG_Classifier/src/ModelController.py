import Definitions

import numpy as np
import os.path as osp
import pandas as pd
from io import StringIO

# Importar la libreria joblib
import joblib
from src.DataPreprocessing import DataPreprocessing

class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")
        # Asegura en una variable la ruta de los modelos
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        # Almacena la ruta de cada modelo en una variable        
        self.pca_path = osp.join(self.model_path, "pca.joblib")
        self.scaler_path = osp.join(self.model_path, "scaler.joblib")
        self.model_path = osp.join(self.model_path, "model.joblib")

        # Cargar los modelos
        self.pca = joblib.load(self.pca_path)
        self.scaler = joblib.load(self.scaler_path)
        self.model = joblib.load(self.model_path)

        # Inicializar variables
        self.input_df = ""
        # Clase de preprocesamiento de la información
        self.d_processing = DataPreprocessing()

    def validate_data(self, df):
        #Compara los nombres de las columnas con el archivo
        return self.d_processing.get_columns().issubset(set(df.columns))
    
    def get_categories(self):
        print("ModelController.get_categories ->")
        return ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']    

    def load_input_data(self, input_data):
        print("ModelController.load_input_data ->")
        try:
            input_data_str = StringIO(input_data.getvalue().decode("utf-8"))
            self.input_df = pd.read_csv(input_data_str)
            is_valid = self.validate_data(self.input_df)
            return self.input_df, is_valid

        except:
            raise("Ocurrió un error al leer la información de entrada")

    def predict(self, data):
        print("ModelController.predict ->")
        X = data[1:].to_numpy()
        Y = data.iloc[0]
        # Escala los datos
        X_scaled = self.scaler.transform([X])
        # Reduce los datos
        X_reduced = self.pca.transform(X_scaled)
        # Genera la predicción
        y_pred = self.model.predict(X_reduced)[0]
        
        return X, Y, y_pred

