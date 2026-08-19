"""
Pipeline Master de Orquestração — Industrial Labor Clusters (v2.0)

Executa sequencialmente as 3 Camadas Analíticas do Framework:
1. Camada Tabular (PySpark MLlib K-Means)
2. Camada de Econometria Espacial (PySAL & GeoPandas - Moran's I & LISA)
3. Camada de Análise de Redes Industriais (NetworkX & Pyvis - Louvain & Centralidade)
"""

import os
import sys
import pandas as pd

# Adicionar raiz do repositório ao sys.path
sys.path.insert(0, os.path.abspath("."))

from src.spatial.spatial_weights import load_uf_geodataframe, create_spatial_weights
from src.spatial.moran_analysis import calculate_global_moran
from src.spatial.lisa_clustering import calculate_lisa_clusters
from src.spatial.__init__ import __all__

from src.network.graph_builder import build_industrial_network
from src.network.network_metrics import calculate_network_centralities
from src.network.community_detection import detect_industrial_communities

from src.viz.spatial_plots import plot_moran_scatterplot, plot_lisa_map
from src.viz.network_plots import export_interactive_pyvis_network, plot_static_network_graph


def run_full_pipeline():
    print("=" * 80)
    print("      FRAMEWORK DE INTELIGÊNCIA TERRITORIAL & ESTRUTURA INDUSTRIAL v2.0")
    print("=" * 80)

    # Criar diretórios de saída
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # -------------------------------------------------------------------------
    # CAMADA 1: ECONOMETRIA ESPACIAL & GEOGRAFIA ECONÔMICA (PySAL / GeoPandas)
    # -------------------------------------------------------------------------
    print("\n[CAMADA 1] Executando Econometria Espacial & Geografia Econômica...")
    
    # Carregar dados processados de KPIs por UF
    kpi_file = "data/outputs/kpis_summary.csv"
    if os.path.exists(kpi_file):
        df_kpis = pd.read_csv(kpi_file, sep=";")
        df_kpis.columns = [c.lower() for c in df_kpis.columns]
    else:
        # Fallback sintético se o CSV Spark ainda não tiver sido gerado
        print("  Aviso: File 'data/outputs/kpis_summary.csv' não encontrado. Gerando dados de demonstração.")
        ufs = ['SP', 'RJ', 'MG', 'ES', 'RS', 'SC', 'PR', 'BA', 'PE', 'CE', 'RN', 'PB', 'SE', 'AL', 'PI', 'MA', 'DF', 'GO', 'MT', 'MS', 'AM', 'PA', 'RO', 'RR', 'AP', 'AC', 'TO']
        df_kpis = pd.DataFrame({
            'uf': ufs,
            'produtividade_r_por_trabalhador': [220000 if u in ['SP', 'RJ', 'SC', 'RS'] else 130000 for u in ufs]
        })

    target_var = 'produtividade_r_por_trabalhador' if 'produtividade_r_por_trabalhador' in df_kpis.columns else df_kpis.columns[1]

    # 1.1 Construir GeoDataFrame vetorial e Matriz W
    gdf = load_uf_geodataframe(df_kpis)
    w, gdf_ordered = create_spatial_weights(gdf, method="knn", k=4, row_standardize=True)
    print(f"  -> Matriz de Pesos Espaciais W construída (27 UFs, {w.k}-NN padronizada por linha).")

    # 1.2 Autocorrelação Espacial Global (I de Moran)
    moran_res = calculate_global_moran(gdf_ordered, target_var, w, permutations=999)
    print(f"  -> I de Moran Global ({target_var}): {moran_res['moran_i']:.4f} (p-value: {moran_res['p_value']:.4f})")
    print(f"     Padrão Detectado: {moran_res['pattern']}")

    # 1.3 Local Indicators of Spatial Association (LISA)
    df_lisa, lisa_counts = calculate_lisa_clusters(gdf_ordered, target_var, w, p_threshold=0.10)
    print("  -> Distribuição dos Clusters Espaciais LISA:")
    for quad_name, count in lisa_counts.items():
        print(f"     * {quad_name}: {count} UFs")

    # 1.4 Renderizar e salvar mapas/gráficos espaciais
    fig_scatter = plot_moran_scatterplot(
        df_lisa, target_var, w, moran_res['moran_i'], moran_res['p_value'],
        output_path="results/figures/moran_scatterplot.png"
    )
    fig_lisa = plot_lisa_map(
        df_lisa, lisa_label_col="lisa_label",
        title=f"Polos Industriais e Clusters Espaciais (LISA — {target_var})",
        output_path="results/figures/lisa_hotspot_map.png"
    )
    df_lisa[['uf', target_var, 'lisa_I', 'lisa_p_value', 'lisa_label']].to_csv("results/tables/lisa_clusters.csv", index=False)
    print("  -> Artefatos salvos em results/figures/ e results/tables/lisa_clusters.csv")

    # -------------------------------------------------------------------------
    # CAMADA 2: ANÁLISE DE REDES E ENCADEAMENTO PRODUTIVO (NetworkX / Pyvis)
    # -------------------------------------------------------------------------
    print("\n[CAMADA 2] Executando Análise de Redes Industriais & Encadeamento Produtivo...")
    
    # 2.1 Construir Grafo CNAE de Insumo-Produto
    G = build_industrial_network()
    print(f"  -> Grafo CNAE construído com {G.number_of_nodes()} setores e {G.number_of_edges()} conexões de fornecimento.")

    # 2.2 Algoritmo de Louvain para Detecção de Comunidades
    G_comm, communities, modularity_q = detect_industrial_communities(G)
    print(f"  -> Detecção de Comunidades (Louvain/Modularidade Q = {modularity_q:.4f}):")
    for comm_id, members in communities.items():
        print(f"     * Ecossistema {comm_id + 1}: {', '.join(members)}")

    # 2.3 Cálculo de Métricas de Centralidade
    df_metrics = calculate_network_centralities(G_comm)
    print("\n  -> Ranking de Setores Industriais por Centralidade de Intermediação (Betweenness):")
    print(df_metrics[['setor', 'cnae', 'betweenness_centrality', 'pagerank']].to_string(index=False))

    # Save ranking to CSV
    df_metrics.to_csv("results/tables/network_centrality_ranking.csv", index=False)

    # 2.4 Renderizar Grafos Estáticos e Interativos em Pyvis HTML
    plot_static_network_graph(G_comm, output_png_path="results/figures/industrial_network_graph.png")
    html_path = export_interactive_pyvis_network(
        G_comm,
        output_html_path="results/figures/industrial_network_graph.html",
        title="Grafo Interativo de Encadeamento Produtivo Industrial"
    )
    print(f"\n  -> Grafo interativo HTML gerado com sucesso: {html_path}")

    print("\n=" * 80)
    print("      EXECUÇÃO DO PIPELINE CONCLUÍDA COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
