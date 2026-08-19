"""
Módulo para cálculo de Métricas de Centralidade e Topologia de Redes Industriais.

Permite identificar setores estratégicos, motores de demanda intermediária e gargalos críticos
da cadeia de valor.
"""

from typing import Dict, Any
import pandas as pd
import networkx as nx


def calculate_network_centralities(G: nx.DiGraph) -> pd.DataFrame:
    """Calcula as principais métricas de centralidade para cada nó (setor) do grafo industrial.

    Métricas calculadas:
    - Betweenness Centrality: Identifica setores gargalos (pontes de intermediação).
    - In-Degree Centrality: Grau de entrada (setores dependentes de insumos).
    - Out-Degree Centrality: Grau de saída (setores fornecedores-chave).
    - PageRank: Relevância sistêmica acumulada na rede.
    - Closeness Centrality: Proximidade média aos demais setores da cadeia.

    Args:
        G: Grafo dirigido NetworkX da cadeia industrial.

    Returns:
        DataFrame ordenado descendentemente por Betweenness Centrality.
    """
    # 1. Centralidade de Intermediação (Betweenness)
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # 2. PageRank (Relevância Sistêmica)
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

    # 3. Centralidade de Proximidade (Closeness)
    closeness = nx.closeness_centrality(G)

    # 4. Graus de Entrada e Saída (In/Out Degree)
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    records = []
    for node in G.nodes():
        node_attr = G.nodes[node]
        records.append({
            "setor": node,
            "categoria": node_attr.get("categoria", "N/A"),
            "cnae": node_attr.get("cnae", "N/A"),
            "betweenness_centrality": float(betweenness.get(node, 0.0)),
            "pagerank": float(pagerank.get(node, 0.0)),
            "closeness_centrality": float(closeness.get(node, 0.0)),
            "in_degree_weight": float(in_degree.get(node, 0.0)),
            "out_degree_weight": float(out_degree.get(node, 0.0))
        })

    df_metrics = pd.DataFrame(records).sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)
    return df_metrics
