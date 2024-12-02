# tsne.py

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import openai
from constants import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
import logging

openai.api_key = OPENAI_API_KEY

def get_embeddings(texts, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS):
    embeddings = []
    for text in texts:
        response = openai.embeddings.create(
            input=[text],
            model=model,
            dimensions=dimensions  # Optional parameter
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

def plot_tsne(embeddings, highlighted_indices, num_chunks, query_index, response_index, pre_reduce_dim=False):
    n_samples, n_features = embeddings.shape

    # Set perplexity to a value less than n_samples
    perplexity = min(30, n_samples - 1)
    if perplexity < 1:
        perplexity = 1  # Minimum perplexity is 1

    # reduce dimensions before t-SNE to speed up computation
    if pre_reduce_dim and n_features > 50:
        # Adjust n_components based on n_samples and n_features
        n_components = min(50, n_samples - 1, n_features)
        pca = PCA(n_components=n_components, random_state=42)
        embeddings = pca.fit_transform(embeddings)
        logging.info(f"Reduced embeddings to {embeddings.shape[1]} dimensions using PCA.")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Prepare data for Plotly
    labels = []
    colors = []
    for idx in range(n_samples):
        if idx == response_index:
            label = 'Response'
            color = 'green'
        elif idx == query_index:
            label = 'Query'
            color = 'red'
        elif idx in highlighted_indices[:-2]:
            label = f'Retrieved Chunk {idx}'
            color = 'blue'
        else:
            label = f'Chunk {idx}'
            color = 'yellow'
        labels.append(label)
        colors.append(color)

    # Create a DataFrame for plotting
    import pandas as pd
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels,
        'color': colors,
        'index': range(n_samples),
    })

    # Plot using Plotly
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='color',
        hover_data=['label', 'index'],
        color_discrete_map={
            'yellow': 'yellow',
            'blue': 'blue',
            'red': 'red',
            'green': 'green',
        },
        labels={'color': 'Point Type'},
    )

    # Use the same shape for all points
    fig.update_traces(marker=dict(symbol='circle', size=8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend=dict(itemsizing='constant'))
    return fig
