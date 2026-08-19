"""
Módulo para renderização de gráficos e mapas cartográficos de Econometria Espacial.
"""

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import libpysal


def plot_moran_scatterplot(
    df: pd.DataFrame,
    variable_name: str,
    w: libpysal.weights.W,
    moran_i: float,
    p_value: float,
    output_path: Optional[str] = None
) -> plt.Figure:
    """Gera o Moran Scatterplot (Variável Padronizada vs Lag Espacial).

    Args:
        df: DataFrame com as observações.
        variable_name: Nome da variável analisada.
        w: Matriz de pesos espaciais PySAL.
        moran_i: Estatística I de Moran observada.
        p_value: P-valor simulado.
        output_path: Caminho opcional para salvar a imagem PNG.

    Returns:
        Objeto Figure do Matplotlib.
    """
    y = df[variable_name].values
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=np.nanmean(y))

    # Padronizar z = (y - mean) / std
    z = (y - np.mean(y)) / np.std(y)
    
    # Lag Espacial Wy
    w_lag = libpysal.weights.lag_spatial(w, z)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.set_style("whitegrid")

    # Dispersão
    ax.scatter(z, w_lag, color="#1f77b4", alpha=0.8, s=80, edgecolors="k", label="Observações (UFs)")

    # Linha de Regressão (inclinação = I de Moran)
    b, a = np.polyfit(z, w_lag, 1)
    x_range = np.linspace(z.min() - 0.5, z.max() + 0.5, 100)
    ax.plot(x_range, a + b * x_range, color="#d62728", linestyle="--", linewidth=2, label=f"Regressão (I = {moran_i:.3f})")

    # Linhas dos Quadrantes no ponto (0,0)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7)

    # Rótulos dos Quadrantes
    ax.text(z.max() * 0.6, w_lag.max() * 0.8, "High-High\n(Hotspots)", color="darkgreen", weight="bold", fontsize=11)
    ax.text(z.min() * 0.9, w_lag.min() * 0.8, "Low-Low\n(Coldspots)", color="darkred", weight="bold", fontsize=11)
    ax.text(z.min() * 0.9, w_lag.max() * 0.8, "Low-High\n(Outliers)", color="orange", weight="bold", fontsize=10)
    ax.text(z.max() * 0.6, w_lag.min() * 0.8, "High-Low\n(Outliers)", color="orange", weight="bold", fontsize=10)

    # Rótulos das UFs se a coluna existir
    if 'uf' in df.columns:
        for idx, row in df.iterrows():
            ax.annotate(row['uf'], (z[idx] + 0.03, w_lag[idx] + 0.03), fontsize=9, weight="semibold")

    ax.set_title(f"Moran Scatterplot — {variable_name}\nI de Moran: {moran_i:.4f} (p-value: {p_value:.4f})", fontsize=14, weight="bold")
    ax.set_xlabel(f"{variable_name} Padronizado (Z)", fontsize=12)
    ax.set_ylabel(f"Lag Espacial (W * Z)", fontsize=12)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return fig


def plot_lisa_map(
    gdf: gpd.GeoDataFrame,
    lisa_label_col: str = "lisa_label",
    title: str = "Mapa de Agrupamento Espacial (LISA)",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Gera o mapa de clusters territoriais LISA (Hotspots, Coldspots e Outliers).

    Args:
        gdf: GeoDataFrame contendo as geometrias e a coluna de rótulos LISA.
        lisa_label_col: Nome da coluna contendo os rótulos LISA.
        title: Título do mapa.
        output_path: Caminho opcional para salvar o arquivo de imagem PNG.

    Returns:
        Objeto Figure do Matplotlib.
    """
    color_map = {
        "High-High (Hotspot)": "#d73027",      # Vermelho (Polos Industriais)
        "Low-Low (Coldspot)": "#4575b4",       # Azul (Zonas de Estagnação)
        "High-Low (Outlier Espacial)": "#fdae61", # Laranja
        "Low-High (Outlier Espacial)": "#abd9e9", # Azul claro
        "Não Significativo": "#e0e0e0"         # Cinza claro
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(
        column=lisa_label_col,
        categorical=True,
        legend=True,
        cmap=None,
        color=[color_map.get(val, "#e0e0e0") for val in gdf[lisa_label_col]],
        edgecolor="white",
        linewidth=1.2,
        ax=ax
    )

    # Adicionar siglas das UFs nos centroides
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        uf_name = row.get('uf', '')
        ax.annotate(uf_name, (centroid.x, centroid.y), ha='center', va='center', fontsize=9, weight="bold", color="black")

    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return fig
