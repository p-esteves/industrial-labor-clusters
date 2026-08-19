"""
Módulo para construção do Grafo de Interdependência Setorial CNAE 2.0.

Modela as cadeias de suprimentos e encadeamentos produtivos industriais onde:
- Nós (Nodes) representam divisões industriais (CNAE).
- Arestas Dirigidas (Edges) representam os fluxos de insumos-produtos / demanda intermediária.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx

# Mapeamento dos 8 setores industriais de referência
SECTOR_CNAE_NODES: Dict[str, Dict[str, Any]] = {
    "Indústria Extrativa": {"cnae": "05-09", "categoria": "Primário", "base_weight": 1.5},
    "Alimentos e Bebidas": {"cnae": "10-11", "categoria": "Bens de Consumo", "base_weight": 1.2},
    "Têxtil e Confecção": {"cnae": "13-14", "categoria": "Bens de Consumo", "base_weight": 0.8},
    "Celulose e Papel": {"cnae": "17", "categoria": "Insumos Básicos", "base_weight": 1.0},
    "Química e Farmacêutica": {"cnae": "20-21", "categoria": "Alta Tecnologia / Insumos", "base_weight": 2.0},
    "Metalurgia": {"cnae": "24", "categoria": "Insumos Básicos", "base_weight": 1.8},
    "Automotiva e Equipamentos": {"cnae": "29-30", "categoria": "Bens de Capital", "base_weight": 2.2},
    "Máquinas e Aparelhos Elétricos": {"cnae": "27-28", "categoria": "Bens de Capital", "base_weight": 1.9}
}

# Matriz de Insumo-Produto Sintética (Coeficientes Técnicos Leontief de fornecimento)
DEFAULT_INPUT_OUTPUT_EDGES: List[Tuple[str, str, float]] = [
    ("Indústria Extrativa", "Metalurgia", 0.75),
    ("Indústria Extrativa", "Química e Farmacêutica", 0.60),
    ("Metalurgia", "Automotiva e Equipamentos", 0.85),
    ("Metalurgia", "Máquinas e Aparelhos Elétricos", 0.80),
    ("Química e Farmacêutica", "Alimentos e Bebidas", 0.45),
    ("Química e Farmacêutica", "Têxtil e Confecção", 0.50),
    ("Química e Farmacêutica", "Automotiva e Equipamentos", 0.55),
    ("Celulose e Papel", "Alimentos e Bebidas", 0.40),
    ("Celulose e Papel", "Química e Farmacêutica", 0.35),
    ("Máquinas e Aparelhos Elétricos", "Automotiva e Equipamentos", 0.70),
    ("Máquinas e Aparelhos Elétricos", "Metalurgia", 0.40),
    ("Alimentos e Bebidas", "Têxtil e Confecção", 0.20),
    ("Automotiva e Equipamentos", "Alimentos e Bebidas", 0.30),
    ("Química e Farmacêutica", "Celulose e Papel", 0.40),
]


def build_industrial_network(
    custom_edges: Optional[List[Tuple[str, str, float]]] = None,
    df_features: Optional[pd.DataFrame] = None
) -> nx.DiGraph:
    """Constrói o Grafo Dirigido e Ponderado de Encadeamento Produtivo CNAE.

    Args:
        custom_edges: Lista opcional de tuplas (Origem, Destino, Peso_Insumo).
        df_features: DataFrame opcional com métricas econômicas para enriquecer os nós.

    Returns:
        Instância de networkx.DiGraph contendo a topologia da rede industrial.
    """
    G = nx.DiGraph()

    # Adicionar Nós com atributos
    for sector, metadata in SECTOR_CNAE_NODES.items():
        G.add_node(
            sector,
            cnae=metadata["cnae"],
            categoria=metadata["categoria"],
            base_weight=metadata["base_weight"]
        )

    # Adicionar Arestas
    edges = custom_edges if custom_edges is not None else DEFAULT_INPUT_OUTPUT_EDGES
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))

    # Se houver atributos econômicos, enriquecer os nós
    if df_features is not None and not df_features.empty:
        sector_col = 'setor' if 'setor' in df_features.columns else None
        if sector_col:
            df_sec = df_features.groupby(sector_col).mean(numeric_only=True)
            for node in G.nodes():
                if node in df_sec.index:
                    for col in df_sec.columns:
                        G.nodes[node][col] = float(df_sec.loc[node, col])

    return G
