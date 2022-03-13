''' Características necesarias para hacer una predicción '''

from pydantic import BaseModel #serializamos los json que entran y salen de la aplicación

class PredictionRequest(BaseModel):
    # Variables que utiliza el modelo para predecir
    opening_gross: float
    screems: float
    production_budget: float
    title_year: int
    aspect_ratio: float
    duration: int
    cast_total_facebook_likes: float
    budget: float
    imb_score: float

class PredictionResponse(BaseModel):
    # Variable a predecir
    worldwide_gross: float
