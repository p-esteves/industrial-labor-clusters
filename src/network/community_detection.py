"""
Módulo para Detecção de Comunidades Industriais e Arranjos Produtivos Locais (APLs).

Aplica algoritmos de otimização de modularidade (Louvain / Greedy Modularity) para identificar
sub-redes altamente integradas de setores produtivos.
"""

from typing import Dict, List, Tuple
import networkx as nx
import networkx.algorithms.community as nx_comm


def detect_industrial_communities(G: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[int, List[str]], float]:
    """Detecta comunidades/clusters topológicos de setores hiperconectados na rede industrial.

    Args:
        G: Grafo dirigido NetworkX da cadeia industrial.

    Returns:
        Tupla contendo:
        - Grafo G atualizado com o atributo 'community' em cada nó.
        - Dicionário mapeando ID da comunidade -> Lista de setores integrantes.
        - Pontuação de Modularidade Q.
    """
    G_copy = G.copy()
    
    # Converter para não-dirigido preservando os maiores pesos para o cálculo de modularidade
    G_undirected = G_copy.to_undirected(reciprocal=False)

    # Detecção de comunidades via otimização de modularidade
    communities_generator = nx_comm.greedy_modularity_communities(G_undirected, weight="weight")
    communities = list(communities_generator)

    # Calcular score de modularidade Q
    modularity_q = float(nx_comm.modularity(G_undirected, communities, weight="weight"))

    community_map: Dict[int, List[str]] = {}
    for comm_id, node_set in enumerate(communities):
        comm_nodes = list(node_set)
        community_map[comm_id] = comm_nodes
        for node in comm_nodes:
            G_copy.nodes[node]["community"] = comm_id
            G_copy.nodes[node]["community_label"] = f"Ecossistema {comm_id + 1}"

    return G_copy, community_map, modularity_q
