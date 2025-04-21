import networkx as nx

def build_graph(df):
    G = nx.DiGraph()

    for idx, row in df.iterrows():
        sloka_id = row.get("Sloka ID", f"Sloka-{idx}")
        sloka = row["Sloka (English Translation)"]
        chapter_verse = row.get("Chapter-Verse", "0:0")
        theme = row.get("Comparison/Relations", "General")

        # Parse chapter and verse
        chapter_num = chapter_verse.split(":")[0]
        chapter = f"Chapter {chapter_num}"

        # Add node with metadata
        G.add_node(sloka_id, text=sloka, chapter=chapter, theme=theme)

        # Add semantic relations
        G.add_edge(sloka_id, chapter, relation="BelongsTo")
        G.add_edge(sloka_id, theme, relation="HasTheme")

    return G


def get_knowledge_context(G, sloka_id):
    context = {
        "chapter": None,
        "theme": None,
        "related_slokas": []
    }

    if sloka_id not in G:
        return context

    # Find direct connections
    for neighbor in G[sloka_id]:
        relation = G[sloka_id][neighbor].get("relation")
        if relation == "BelongsTo":
            context["chapter"] = neighbor
        elif relation == "HasTheme":
            context["theme"] = neighbor

    # Find other slokas with same theme
    theme = context["theme"]
    if theme:
        for node in G.nodes:
            if node != sloka_id and G.nodes[node].get("theme") == theme:
                context["related_slokas"].append(node)

    return context