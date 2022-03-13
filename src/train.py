''' En este notebook vamos a presentar los pasos para entrenar el modelo '''
from utils import update_model #importamos la funcion que nos guarda el modelo de utils
from utils import save_simple_metric_report, get_model_performance_test_set
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    format = "%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level = logging.INFO,
    datefmt = "%H:%M:%S",
    stream = sys.stderr
)

logger = logging.getLogger(__name__)

logger.info("Loading data...")
data = pd.read_csv("dataset/full_data.csv")

logger.info("Loading model...")
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)), # Incluimos la limpieza de los nan aqui
    ("core_model", GradientBoostingRegressor())
])

logger.info("separando dataset en train y test")
X = data.drop(["worldwide_gross"], axis=1)
y = data.loc[:,"worldwide_gross"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

logger.info("setting hiperparámetros to tune")

param_tuning = {"core_model__n_estimators":range(20,301,20)}

grid_search = GridSearchCV(model, param_grid=param_tuning, scoring="r2", cv=5)

logger.info("empezando el gridsearch...")
grid_search.fit(X_train, y_train)

logger.info("Hacemos cross validation del mejor modelo")
final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=5) # hacemos la validación con el mejor model_selection

train_score = np.mean(final_result["train_score"])
test_score = np.mean(final_result["test_score"])

assert train_score > 0.7
assert test_score > 0.65

logger.info(f"train score: {train_score}")
logger.info(f"test score: {test_score}")

logger.info("Actualizando el modelo") # Aquí tendremos una función que nos guarde el modelo. Lo añadiremos en el archivo utils
update_model(grid_search.best_estimator_) # best_estimator_ es el mejor modelo que se saca del GridSearch

logger.info("Generando reporte del modelo...")
validation_score = grid_search.best_estimator_.score(X_test, y_test)
save_simple_metric_report(train_score, test_score, validation_score, grid_search.best_estimator_) #función en utils

logger.info("Gráfica de evolución del modelo")
y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred) #función creada en utils

logger.info("training finish")
