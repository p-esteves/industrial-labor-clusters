"""
Módulo para teste estatístico de Autocorrelação Espacial Global (I de Moran).

Mede a dependência espacial global de variáveis econômicas (produtividade, emprego, inovação)
identificando se a distribuição no espaço é agrupada, dispersa ou aleatória.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import libpysal
from esda.moran import Moran


def calculate_global_moran(
    df: pd.DataFrame,
    variable_name: str,
    w: libpysal.weights.W,
    permutations: int = 999
) -> Dict[str, Any]:
    """Calcula o estatístico I de Moran Global para uma variável continuada.

    Args:
        df: DataFrame com as observações territoriais.
        variable_name: Nome da coluna da variável a ser testada.
        w: Matriz de pesos espaciais PySAL.
        permutations: Número de permutações de Monte Carlo para cálculo do p-valor.

    Returns:
        Dicionário com os resultados da estatística I de Moran:
        - moran_i: Valor do I de Moran observado [-1, 1]
        - p_value: P-valor simulado por Monte Carlo
        - z_score: Z-score padronizado
        - expected_i: Valor esperado sob hipótese nula de aleatoriedade
        - variable: Nome da variável analisada

    Raises:
        KeyError: Se variable_name não estiver presente em df.
    """
    if variable_name not in df.columns:
        raise KeyError(f"A variável '{variable_name}' não foi encontrada no DataFrame.")

    y = df[variable_name].values
    
    # Remover NaNs se houver
    if np.isnan(y).any():
        mean_val = np.nanmean(y)
        y = np.nan_to_num(y, nan=mean_val)

    mi = Moran(y, w, permutations=permutations)

    # Interpretação qualitativa
    if mi.p_sim < 0.05:
        pattern = "Agrupamento Espacial Significativo (Clustering)" if mi.I > mi.EI else "Dispersão Espacial Significativa"
    else:
        pattern = "Distribuição Espacial Aleatória (Sem Autocorrelação)"

    return {
        "variable": variable_name,
        "moran_i": float(mi.I),
        "expected_i": float(mi.EI),
        "p_value": float(mi.p_sim),
        "z_score": float(mi.z_sim),
        "p_norm": float(mi.p_norm),
        "pattern": pattern,
        "permutations": permutations
    }
