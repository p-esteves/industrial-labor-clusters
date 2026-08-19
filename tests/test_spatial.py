"""
Testes unitários automatizados para o módulo de Econometria Espacial (src/spatial).
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from src.spatial.spatial_weights import load_uf_geodataframe, create_spatial_weights
from src.spatial.moran_analysis import calculate_global_moran
from src.spatial.lisa_clustering import calculate_lisa_clusters


@pytest.fixture
def mock_uf_kpis() -> pd.DataFrame:
    """Fixture gerando DataFrame sintético de 27 UFs com indicador de produtividade."""
    ufs = [
        'SP', 'RJ', 'MG', 'ES', 'RS', 'SC', 'PR', 'BA', 'PE', 'CE',
        'RN', 'PB', 'SE', 'AL', 'PI', 'MA', 'DF', 'GO', 'MT', 'MS',
        'AM', 'PA', 'RO', 'RR', 'AP', 'AC', 'TO'
    ]
    np.random.seed(42)
    prod = np.random.normal(150000, 30000, size=len(ufs))
    # Efeito regional simulado
    for i, uf in enumerate(ufs):
        if uf in ['SP', 'RJ', 'SC', 'RS']:
            prod[i] += 80000

    return pd.DataFrame({'uf': ufs, 'produtividade_media_uf': prod})


def test_load_uf_geodataframe(mock_uf_kpis):
    """Testa o carregamento e enriquecimento do GeoDataFrame vetorial das UFs."""
    gdf = load_uf_geodataframe(mock_uf_kpis)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 27
    assert 'geometry' in gdf.columns
    assert 'produtividade_media_uf' in gdf.columns
    assert gdf.crs.to_string() == "EPSG:4326"


def test_create_spatial_weights(mock_uf_kpis):
    """Testa a geração e padronização por linha da Matriz de Pesos Espaciais W."""
    gdf = load_uf_geodataframe(mock_uf_kpis)
    w, gdf_out = create_spatial_weights(gdf, method="knn", k=4, row_standardize=True)
    
    assert w.n == 27
    assert w.transform == 'R'
    assert w.k == 4
    # Soma de cada linha padronizada deve ser 1.0 (ou 0 para isolados)
    for i in range(w.n):
        row_sum = sum(w.weights[i])
        assert pytest.approx(row_sum, 1e-5) == 1.0


def test_calculate_global_moran(mock_uf_kpis):
    """Testa o cálculo da autocorrelação espacial global I de Moran."""
    gdf = load_uf_geodataframe(mock_uf_kpis)
    w, _ = create_spatial_weights(gdf, method="knn", k=4)
    
    results = calculate_global_moran(mock_uf_kpis, "produtividade_media_uf", w, permutations=99)
    
    assert "moran_i" in results
    assert "p_value" in results
    assert "z_score" in results
    assert -1.0 <= results["moran_i"] <= 1.0
    assert 0.0 <= results["p_value"] <= 1.0


def test_calculate_lisa_clusters(mock_uf_kpis):
    """Testa o agrupamento LISA local e atribuição dos quadrantes Moran."""
    gdf = load_uf_geodataframe(mock_uf_kpis)
    w, _ = create_spatial_weights(gdf, method="knn", k=4)
    
    df_lisa, counts = calculate_lisa_clusters(mock_uf_kpis, "produtividade_media_uf", w, permutations=99)
    
    assert "lisa_I" in df_lisa.columns
    assert "lisa_quad" in df_lisa.columns
    assert "lisa_label" in df_lisa.columns
    assert len(df_lisa) == 27
    assert df_lisa["lisa_quad"].isin([0, 1, 2, 3, 4]).all()
