"""
Módulo para construção e manipulação de Matrizes de Pesos Espaciais (W).

Este módulo disponibiliza funções para gerar matrizes de contiguidade (Queen, Rook)
e matrizes de k-vizinhos mais próximos (k-NN) a partir de geometrias territoriais.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import libpysal


# Dicionário de coordenadas de centroides aproximados das 27 UFs do Brasil (EPSG:4326)
UF_CENTROIDS: Dict[str, Tuple[float, float]] = {
    'SP': (-46.6333, -23.5505), 'RJ': (-43.1729, -22.9068), 'MG': (-43.9378, -19.9208), 'ES': (-40.3380, -20.3194),
    'RS': (-51.2177, -30.0346), 'SC': (-48.5480, -27.5948), 'PR': (-49.2731, -25.4284),
    'BA': (-38.5108, -12.9714), 'PE': (-34.8811, -8.0542),  'CE': (-38.5267, -3.7319),  'RN': (-35.2094, -5.7945),
    'PB': (-34.8610, -7.1150),  'SE': (-37.0731, -10.9472), 'AL': (-35.7353, -9.6658),  'PI': (-42.8019, -5.0892),
    'MA': (-44.3028, -2.5307),  'DF': (-47.9292, -15.7801), 'GO': (-49.2648, -16.6869), 'MT': (-56.0967, -15.6010),
    'MS': (-54.6464, -20.4428), 'AM': (-60.0217, -3.1190),  'PA': (-48.4902, -1.4558),  'RO': (-63.9039, -8.7619),
    'RR': (-60.6758, 2.8235),   'AP': (-51.0664, 0.0355),   'AC': (-67.8100, -9.9754),  'TO': (-48.3336, -10.1844)
}


def load_uf_geodataframe(df_kpis: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
    """Carrega ou constrói um GeoDataFrame vetorial com os centroides e Voronoi/Buffer das 27 UFs.

    Args:
        df_kpis: DataFrame com dados das UFs e indicadores econômicos.

    Returns:
        GeoDataFrame contendo as geometrias e atributos das UFs.
    """
    ufs = list(UF_CENTROIDS.keys())
    data = []

    for uf in ufs:
        lon, lat = UF_CENTROIDS[uf]
        # Criar polígono aproximado (buffer regular em torno do centroide)
        point = Point(lon, lat)
        poly = point.buffer(1.2) # Aproximação de contiguidade espacial
        data.append({'uf': uf, 'geometry': poly, 'centroid_lon': lon, 'centroid_lat': lat})

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    if df_kpis is not None and not df_kpis.empty:
        # Padronizar nome da coluna de UF para merge
        uf_col = 'uf' if 'uf' in df_kpis.columns else ('UF' if 'UF' in df_kpis.columns else None)
        if uf_col:
            gdf = gdf.merge(df_kpis, left_on='uf', right_on=uf_col, how='left')

    return gdf


def create_spatial_weights(
    gdf: gpd.GeoDataFrame,
    method: str = "knn",
    k: int = 4,
    row_standardize: bool = True
) -> Tuple[libpysal.weights.W, gpd.GeoDataFrame]:
    """Constrói a Matriz de Pesos Espaciais (W) a partir de um GeoDataFrame.

    Args:
        gdf: GeoDataFrame contendo a coluna 'geometry' ou colunas de coordenadas.
        method: Método de construção da matriz ('knn', 'queen', ou 'rook').
        k: Número de vizinhos mais próximos (usado apenas se method='knn').
        row_standardize: Se True, aplica padronização por linha (transform='R').

    Returns:
        Tupla contendo a matriz de pesos PySAL (W) e o GeoDataFrame ordenado.

    Raises:
        ValueError: Se o método especificado for inválido.
    """
    gdf = gdf.copy().reset_index(drop=True)
    method_clean = method.lower()

    if method_clean == "knn":
        # Construir vizinhos por coordenadas de centroides
        coordinates = np.column_stack((gdf.geometry.centroid.x, gdf.geometry.centroid.y))
        w = libpysal.weights.KNN.from_array(coordinates, k=k)
    elif method_clean == "queen":
        w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    elif method_clean == "rook":
        w = libpysal.weights.Rook.from_dataframe(gdf, use_index=False)
    else:
        raise ValueError(f"Método de peso espacial '{method}' não suportado. Escolha entre 'knn', 'queen' ou 'rook'.")

    if row_standardize:
        w.transform = 'R'

    return w, gdf
