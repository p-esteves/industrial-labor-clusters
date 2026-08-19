"""
Módulo para Análise de Associação Espacial Local (LISA - Local Indicators of Spatial Association).

Identifica clusters locais de alto e baixo valor (Hotspots e Coldspots) e outliers espaciais
(High-Low e Low-High) com significância estatística.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import libpysal
from esda.moran import Moran_Local


LISA_LABELS: Dict[int, str] = {
    0: "Não Significativo",
    1: "High-High (Hotspot)",
    2: "Low-High (Outlier Espacial)",
    3: "Low-Low (Coldspot)",
    4: "High-Low (Outlier Espacial)"
}


def calculate_lisa_clusters(
    df: pd.DataFrame,
    variable_name: str,
    w: libpysal.weights.W,
    p_threshold: float = 0.05,
    permutations: int = 999
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Calcula estatísticas LISA locais e classifica as observações em quadrantes Moran.

    Quadrantes:
    1: High-High (Valor alto cercado por valores altos -> Polo Industrial Consolidador)
    2: Low-High (Valor baixo cercado por valores altos -> Periferia em desenvolvimento)
    3: Low-Low (Valor baixo cercado por valores baixos -> Zona de Estagnação)
    4: High-Low (Valor alto cercado por valores baixos -> Enclave Isolado)

    Args:
        df: DataFrame contendo as observações.
        variable_name: Nome da coluna da variável analisada.
        w: Matriz de pesos espaciais PySAL.
        p_threshold: Limiar de significância estatística (default: 0.05).
        permutations: Número de permutações de Monte Carlo.

    Returns:
        Tupla contendo:
        - DataFrame original enriquecido com colunas 'lisa_Is', 'lisa_p_value', 'lisa_quad' e 'lisa_label'.
        - Dicionário com a contagem de observações por quadrante LISA.
    """
    df_out = df.copy()
    y = df_out[variable_name].values
    
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=np.nanmean(y))

    lm = Moran_Local(y, w, transformation="r", permutations=permutations, seed=123)

    # Filtrar por significância estatística
    quads = lm.q.copy()
    sig = lm.p_sim < p_threshold
    
    # Zerar observações não significativas
    quads[~sig] = 0

    df_out["lisa_I"] = lm.Is
    df_out["lisa_p_value"] = lm.p_sim
    df_out["lisa_quad"] = quads
    df_out["lisa_label"] = [LISA_LABELS.get(q, "Não Significativo") for q in quads]

    # Resumo por quadrante
    counts = dict(pd.Series([LISA_LABELS[q] for q in quads]).value_counts())

    return df_out, counts
