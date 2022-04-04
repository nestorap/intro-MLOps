''' Aquí creamos transform_to_dataframe para transformar los inputs en dataframe '''

from joblib import load
from scipy.sparse import data
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO

def get_model() -> Pipeline:
    model_path = os.environ.get("MODEL_PATH", "model/model.pkl")
    with open(model_path, "rb") as model_file:
        model = load(BytesIO(model_file.read()))
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transiction_dictionary = {key:[value] for key, value in class_model.dict().items()}
    data_frame = DataFrame(transiction_dictionary)
    return data_frame
