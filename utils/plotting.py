"""
Utility functions for plotting.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import wandb


def plot_heatmap(
    matrix, flipped_matrix, labels_names, labels_names_gt, log_name="Correlation Matrix"
):
    # Create the heatmap using Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=flipped_matrix,
            x=labels_names,  # Column labels
            y=labels_names_gt[::-1],  # Row labels
            colorscale="RdBu_r",  # Color scale similar to Seaborn's 'coolwarm'
            zmin=-1,
            zmax=1,  # Assuming correlation values range from -1 to 1
        )
    )

    # Update layout for better readability at large scales
    fig.update_layout(
        autosize=False,
        width=800,  # Maintain square aspect
        height=800,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            constrain="domain",
        ),
        yaxis=dict(showgrid=False, zeroline=False, constrain="domain"),
        margin=dict(l=10, r=10, t=90, b=10),  # Reduced margins
        plot_bgcolor="rgba(0,0,0,0)",  # Set background color to transparent
    )
    fig.update_xaxes(fixedrange=False)  # Allow x-axis to be zoomed independently
    fig.update_yaxes(fixedrange=False)  # Allow y-axis to be zoomed independently

    wandb.log({f"Interactive {log_name}": wandb.Plotly(fig)})
    # Assuming c_corr is your correlation matrix
    sns.set_theme(style="white")

    # Create a heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        matrix,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )  # , yticklabels=concept_names_graph, xticklabels=concept_names_graph
    plt.title("Correlation Matrix Heatmap")

    # Save the heatmap to a file
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    wandb.log({log_name: wandb.Image(Image.open(buf))})


def compute_and_plot_heatmap(
    matrix, concepts_true, concept_names_graph, config, log_name=None
):
    # Reorder CUB concepts to group colors&shapes instead of concept groups
    if config.data.dataset == "CUB":
        new_group = np.array(
            [
                (name.split(": ")[1] if not name.isdigit() else name)
                for name in concept_names_graph
            ]
        )
        unique_groups, index, counts = np.unique(
            new_group, return_counts=True, return_index=True
        )
        # reorder groups and counts to preserve order of unique groups
        unique_groups = unique_groups[np.argsort(index)]
        counts = counts[np.argsort(index)]

        # Get the indices that sort by descending frequency
        sorted_indices = np.argsort(-counts, kind="stable")

        # Use these indices to reorder the unique_groups and counts arrays
        unique_groups = unique_groups[sorted_indices]

        # Get the indices that sort new_group to fit the new order
        new_rowcol = np.argwhere((unique_groups == new_group[:, None]))
        assert (
            new_rowcol[:, 0] == np.arange(len(new_rowcol))
        ).all(), "Error in reordering"
        permutation = np.argsort(new_rowcol[:, 1], kind="stable")
        perm_matrix = matrix[permutation][:, permutation]
        perm_flipped_matrix = np.flipud(perm_matrix)
        perm_flipped_matrix = np.vstack(
            [
                np.append(
                    concepts_true[0].cpu().numpy(),
                )[permutation],
                perm_flipped_matrix,
            ]
        )
        labels_names = [concept_names_graph[i] for i in permutation]
        labels_names_new = np.append(labels_names, "Ground Truth")
        plot_heatmap(
            matrix=perm_matrix,
            flipped_matrix=perm_flipped_matrix,
            labels_names=labels_names,
            labels_names_gt=labels_names_new,
            log_name=(
                "Averaged CUB concept-ordered Correlation Matrix"
                if log_name is not None
                else "CUB concept-ordered Correlation Matrix"
            ),
        )

    # Perform clustering for permutation into cliques (except synthetic with predefined order)
    if config.data.get("sim_type") == "correlated_c":
        permutation = [i for i in range(len(concept_names_graph))]

    else:
        dist_matrix = 1 - ((matrix + matrix.T) / 2)
        linkage_matrix = linkage(dist_matrix, method="complete", optimal_ordering=True)
        dendro = dendrogram(linkage_matrix, no_plot=True)
        permutation = dendro["leaves"]

    # Apply permutation
    perm_matrix = matrix[permutation][:, permutation]
    perm_flipped_matrix = np.flipud(perm_matrix)
    perm_flipped_matrix = np.vstack(
        [
            np.append(
                concepts_true[0].cpu().numpy(),
            )[permutation],
            perm_flipped_matrix,
        ]
    )
    labels_names = [concept_names_graph[i] for i in permutation]
    labels_names_new = np.append(labels_names, "Ground Truth")

    args = {
        "matrix": perm_matrix,
        "flipped_matrix": perm_flipped_matrix,
        "labels_names": labels_names,
        "labels_names_gt": labels_names_new,
    }
    if log_name is not None:
        args["log_name"] = log_name

    plot_heatmap(**args)
