from pyvis.network import Network
import pandas as pd
from graph_utils import build_graph

def visualize_sloka_graph(sloka_id="Sloka-0"):
    import pandas as pd
    from pyvis.network import Network
    from graph_utils import build_graph

    df = pd.read_csv("dataset/dataset1.csv")
    G = build_graph(df)

    net = Network(height="600px", width="100%", notebook=False)
    sub_nodes = [sloka_id] + list(G[sloka_id])
    subgraph = G.subgraph(sub_nodes)

    net.from_nx(subgraph)

    output_file = "sloka_knowledge_graph.html"
    net.write_html(output_file)
    return output_file  # ✅ return path