'''En este script incluimos el preprocesado de los datos. Aquí vamos a tener todos los pasos
que seguimos en el notebook para obtener los datos completos, con su merge y todo eso, para finalmente
guardarlo en csv'''


from dvc import api
import pandas as pd
from io import StringIO # Para cuando descarguemos los archivos con la API de dvc
import sys
import logging

logging.basicConfig(
    format = '%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level = logging.INFO,
    datefmt = "%H:%M:%S", # formato de la fecha
    stream = sys.stderr
)

logger = logging.getLogger(__name__)

logging.info("Fetching data...")

# Descargamos los archivos. Tenemos que convertirlos después en datasets
movie_data_path = api.read("dataset/movies.csv", remote="dataset-track") # Para ver el nombre del parametro "remote" lo tenemos en la carpeta .dvc y config, donde esta toda la configuración
finantial_data_path = api.read("dataset/finantials.csv", remote="dataset-track")
opening_data_path = api.read("dataset/opening_gross.csv", remote="dataset-track")

movie_data = pd.read_csv(StringIO(movie_data_path))
fin_data = pd.read_csv(StringIO(finantial_data_path))
opening_data = pd.read_csv(StringIO(opening_data_path))

numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
movie_data = movie_data[numeric_columns + ["movie_title"]]

fin_data = fin_data.loc[:,["movie_title", "production_budget", "worldwide_gross"]]

fin_movie_data = pd.merge(
    fin_data,
    movie_data,
    on="movie_title",
    how="left"
)

full_movie_data = pd.merge(
    opening_data,
    fin_movie_data,
    on="movie_title",
    how="left"
)

full_movie_data = full_movie_data.drop(["gross", "movie_title"], axis=1)

full_movie_data.to_csv("dataset/full_data.csv", index=False)

logger.info("Data fetcherd and prepare...")
