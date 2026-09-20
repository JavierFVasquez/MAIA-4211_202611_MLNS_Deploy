import numpy as np
import pandas as pd

class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.MAX_PIXEL_COUNT = 784

    def transform(self, df):
        print("DataPreprocessing.transform ->")
        return None

    def get_columns(self):
        print("DataPreprocessing.get_columns ->")
        # Genera los nombres de las columnas en una lista para la variable res
        res = ["label"] + [f"pixel{i}" for i in range(1, self.MAX_PIXEL_COUNT + 1)]

        return set(res)

    def get_cat_name(self, index):
        print("DataPreprocessing.get_cat_name ->")
        if index < 0 or index > 1:
            return ""
        return self.get_categories()[index]
    

