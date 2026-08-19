"""
Testes unitários automatizados para o módulo de Análise de Redes Industriais (src/network).
"""

import pytest
import pandas as pd
import networkx as nx
from src.network.graph_builder import build_industrial_network, SECTOR_CNAE_NODES
from src.network.network_metrics import calculate_network_centralities
from src.network.community_detection import detect_industrial_communities


def test_build_industrial_network():
    """Testa a construção do grafo dirigido de interdependência setorial."""
    G = build_industrial_network()
    
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == len(SECTOR_CNAE_NODES)
    assert G.number_of_edges() > 0
    
    # Verificar se todos os nós possuem metadados CNAE
    for node in G.nodes():
        assert "cnae" in G.nodes[node]
        assert "categoria" in G.nodes[node]


def test_calculate_network_centralities():
    """Testa o cálculo de métricas de centralidade topológica na rede."""
    G = build_industrial_network()
    df_metrics = calculate_network_centralities(G)
    
    assert isinstance(df_metrics, pd.DataFrame)
    assert len(df_metrics) == len(SECTOR_CNAE_NODES)
    
    expected_cols = [
        "setor", "betweenness_centrality", "pagerank",
        "closeness_centrality", "in_degree_weight", "out_degree_weight"
    ]
    for col in expected_cols:
        assert col in df_metrics.columns

    # Betweenness deve estar no intervalo [0, 1]
    assert (df_metrics["betweenness_centrality"] >= 0.0).all()
    assert (df_metrics["betweenness_centrality"] <= 1.0).all()


def test_detect_industrial_communities():
    """Testa o algoritmo de detecção de comunidades Louvain / Modularidade."""
    G = build_industrial_network()
    G_comm, community_map, modularity_q = detect_industrial_communities(G)
    
    assert isinstance(G_comm, nx.DiGraph)
    assert isinstance(community_map, dict)
    assert len(community_map) >= 1
    assert isinstance(modularity_q, float)
    
    # Todos os nós devem ter o atributo 'community'
    for node in G_comm.nodes():
        assert "community" in G_comm.nodes[node]
