"""
Módulo para renderização de Grafos de Encadeamento Produtivo (Estáticos e Interativos em Pyvis HTML).
"""

from typing import Optional
import os
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network


def export_interactive_pyvis_network(
    G: nx.DiGraph,
    output_html_path: str = "results/figures/industrial_network_graph.html",
    title: str = "Rede de Interdependência e Encadeamento Produtivo Industrial"
) -> str:
    """Exporta o grafo de redes industriais como uma aplicação HTML 3D/2D interativa via Pyvis.

    Nós são dimensionados pela Centralidade de Intermediação (Betweenness) e coloridos
    por Comunidade Louvain. Arestas mostram os coeficientes de suprimento.

    Args:
        G: Grafo dirigido NetworkX.
        output_html_path: Caminho do arquivo HTML de saída.
        title: Título do cabeçalho da visualização.

    Returns:
        Caminho absoluto do arquivo HTML gerado.
    """
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    net = Network(height="750px", width="100%", notebook=False, directed=True)
    net.toggle_physics(True)

    # Calcular entrelaçamento para dimensionamento de nós
    betweenness = nx.betweenness_centrality(G, weight="weight")

    # Cores por comunidade Louvain
    community_colors = {
        0: "#e41a1c", # Vermelho
        1: "#377eb8", # Azul
        2: "#4daf4a", # Verde
        3: "#984ea3"  # Roxo
    }

    for node in G.nodes():
        node_attr = G.nodes[node]
        bw_score = betweenness.get(node, 0.1)
        node_size = 25 + (bw_score * 70)
        
        comm_id = node_attr.get("community", 0)
        color = community_colors.get(comm_id, "#ff7f00")

        hover_info = (
            f"<b>Setor:</b> {node}<br>"
            f"<b>CNAE:</b> {node_attr.get('cnae', 'N/A')}<br>"
            f"<b>Categoria:</b> {node_attr.get('categoria', 'N/A')}<br>"
            f"<b>Betweenness Centrality:</b> {bw_score:.4f}<br>"
            f"<b>Comunidade:</b> Ecossistema {comm_id + 1}"
        )

        net.add_node(
            node,
            label=node,
            title=hover_info,
            size=node_size,
            color=color,
            font={"size": 14, "face": "arial", "color": "#000000"}
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 0.5)
        net.add_edge(
            u, v,
            value=weight * 3,
            title=f"Fornecimento {u} → {v}: Coef. {weight:.2f}",
            arrowStrikethrough=False
        )

    # Definir opções visuais e salvar
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 120
        }
      }
    }
    """)

    net.write_html(output_html_path)
    return os.path.abspath(output_html_path)


def plot_static_network_graph(
    G: nx.DiGraph,
    output_png_path: Optional[str] = None
) -> plt.Figure:
    """Gera um gráfico estático do grafo de redes industriais usando Matplotlib.

    Args:
        G: Grafo dirigido NetworkX.
        output_png_path: Caminho opcional para salvar a imagem PNG.

    Returns:
        Objeto Figure do Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = nx.spring_layout(G, k=1.2, seed=42)

    betweenness = nx.betweenness_centrality(G, weight="weight")
    node_sizes = [3000 * (betweenness.get(n, 0.1) + 0.1) for n in G.nodes()]

    # Colorir por comunidade se disponível
    node_colors = [G.nodes[n].get("community", 0) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        alpha=0.9,
        ax=ax
    )

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=18,
        edge_color="gray",
        width=1.8,
        alpha=0.7,
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight="bold",
        font_color="black",
        ax=ax
    )

    ax.set_title("Grafo de Interdependência Setorial e Encadeamento Produtivo", fontsize=15, weight="bold")
    ax.set_axis_off()
    plt.tight_layout()

    if output_png_path:
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        fig.savefig(output_png_path, dpi=300)
        plt.close(fig)

    return fig
