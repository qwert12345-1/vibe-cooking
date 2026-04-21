"""Plotly 2-D K-Means scatter with click-to-explore interaction.

The interaction model: after a query, we identify the *single* dominant cluster
among the top-K recommendations, highlight it, and draw a circle around it to
signal to the user "recipes in this region of the map are likely also a match
for what you have on hand." Non-dominant clusters are drawn at low opacity so
they recede into the background but remain visible for exploration.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.graph_objects as go

from src.recommender import RecipeRecommender


def _dominant_cluster(engine: RecipeRecommender, highlight_indices: list[int]) -> int | None:
    """Return the cluster id that appears most often among highlighted recipes."""
    if not highlight_indices:
        return None
    labels = [int(engine.cluster_labels[i]) for i in highlight_indices
              if 0 <= i < len(engine.recipes)]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _matching_circle_around(
    center: tuple[float, float],
    other_points: np.ndarray,
    *,
    percentile: float = 80.0,
) -> tuple[float, float, float] | None:
    """Return (cx, cy, r) for a circle centered at `center` that encloses
    `percentile`% of `other_points`.

    Semantically: "draw a neighborhood around the user's position big enough
    to cover most of the recommended recipes." Using the 80th percentile (not
    the maximum) keeps the circle tight when one or two recommendations land
    in a distant part of the t-SNE map due to the projection's distance
    distortion.
    """
    if other_points.size == 0:
        return None
    cx, cy = float(center[0]), float(center[1])
    dists = np.linalg.norm(other_points - np.array([cx, cy]), axis=1)
    r = float(np.percentile(dists, percentile))
    if r <= 0:
        return None
    return cx, cy, r


def _circle_polyline(cx: float, cy: float, r: float, n: int = 120) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)


def build_cluster_figure(
    engine: RecipeRecommender,
    *,
    highlight_indices: list[int] | None = None,
    sample_per_cluster: int = 400,
    user_position: tuple[float, float] | None = None,
) -> go.Figure:
    """Scatter plot of the 2-D recipe projection.

    - When no highlights are provided: all clusters shown at normal opacity.
    - When highlights are provided: the cluster containing the most top-K
      recommendations is drawn in full color with a dashed circle around it;
      all other clusters fade to low opacity, and the individual recommended
      recipes are marked with red stars.
    """
    coords = engine.projection_2d
    labels = engine.cluster_labels
    rng = np.random.default_rng(0)

    focus_cid = _dominant_cluster(engine, highlight_indices or [])
    has_focus = focus_cid is not None

    fig = go.Figure()
    unique_clusters = sorted(set(int(c) for c in labels.tolist()))
    for cid in unique_clusters:
        idx = np.where(labels == cid)[0]
        if idx.size > sample_per_cluster:
            idx = rng.choice(idx, size=sample_per_cluster, replace=False)
        cluster_desc = engine.cluster_label_guess(cid)
        hover = [
            f"<b>{engine.recipes[int(i)].name[:60]}</b><br>"
            f"<i>cluster {cid}: {cluster_desc}</i><br>"
            f"cuisine: {', '.join(engine.recipes[int(i)].cuisine) or '—'}<br>"
            f"{int(engine.recipes[int(i)].calories)} kcal"
            for i in idx
        ]

        is_focus = has_focus and cid == focus_cid
        marker_size = 7 if is_focus else 5
        marker_opacity = 0.85 if is_focus else 0.55
        trace_name = (
            f"★ matching region — cluster {cid}: {engine.cluster_label_guess(cid)}"
            if is_focus
            else f"cluster {cid}: {engine.cluster_label_guess(cid)}"
        )
        # When a query is active, hide non-focus clusters entirely (they stay in
        # the legend as 'legendonly' — user can click to re-show them). Keeping
        # them on at low opacity was visually noisy.
        visibility: bool | str = True if (not has_focus or is_focus) else "legendonly"
        fig.add_trace(
            go.Scatter(
                x=coords[idx, 0],
                y=coords[idx, 1],
                mode="markers",
                name=trace_name,
                marker=dict(size=marker_size, opacity=marker_opacity),
                hovertext=hover,
                hoverinfo="text",
                customdata=idx,
                visible=visibility,
            )
        )

    # Matching region: a circle centered on the user's query position, sized
    # to enclose ~80% of the top-K recommendations. This reads naturally as
    # "recipes near you on the map are matches." Only drawn when we have both
    # a user position and recommendations.
    if user_position is not None and highlight_indices:
        hi = np.array(
            [i for i in highlight_indices if 0 <= i < len(engine.recipes)],
            dtype=int,
        )
        if hi.size:
            circle = _matching_circle_around(
                user_position, coords[hi], percentile=80.0
            )
            if circle is not None:
                cx, cy, r = circle
                poly = _circle_polyline(cx, cy, r)
                fig.add_trace(
                    go.Scatter(
                        x=poly[:, 0],
                        y=poly[:, 1],
                        mode="lines",
                        line=dict(color="#d62728", width=2, dash="dash"),
                        fill="toself",
                        fillcolor="rgba(214,39,40,0.06)",
                        name="matching region",
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )

    # User's input projected onto the same 2-D space — a single bold marker
    # that says "this is where your fridge sits in the recipe universe."
    if user_position is not None:
        ux, uy = user_position
        fig.add_trace(
            go.Scatter(
                x=[ux], y=[uy],
                mode="markers+text",
                marker=dict(size=22, color="#1d3557", symbol="diamond",
                            line=dict(width=2, color="white")),
                text=["you"],
                textposition="top center",
                textfont=dict(size=13, color="#1d3557"),
                name="your input position",
                hovertext=["your input projected into the recipe map"],
                hoverinfo="text",
            )
        )

    # Individual top-K recommendations as red stars on top.
    if highlight_indices:
        hi = np.array(
            [i for i in highlight_indices if 0 <= i < len(engine.recipes)],
            dtype=int,
        )
        if hi.size:
            hover = [f"<b>★ {engine.recipes[int(i)].name[:60]}</b>" for i in hi]
            fig.add_trace(
                go.Scatter(
                    x=coords[hi, 0],
                    y=coords[hi, 1],
                    mode="markers",
                    name="your recommendations",
                    marker=dict(size=13, color="#d62728", symbol="star",
                                line=dict(width=1, color="white")),
                    hovertext=hover,
                    hoverinfo="text",
                    customdata=hi,
                )
            )

    fig.update_layout(
        title=dict(
            text="Recipe Universe",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            pad=dict(t=20),
        ),
        xaxis_title="latent-1",
        yaxis_title="latent-2",
        height=720,
        legend=dict(
            font=dict(size=10),
            x=0.58,
            xanchor="left",
            y=0.82,
            yanchor="top",
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(148,163,184,0.30)",
            borderwidth=1,
        ),
        margin=dict(l=32, r=32, t=56, b=24),
        # Lock axis aspect ratio to 1:1 so the "matching region" circle
        # actually looks like a circle (not a horizontally squashed ellipse
        # just because the plot container is wider than it is tall).
        xaxis=dict(domain=[0.04, 0.54], constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
    )
    return fig


def build_elbow_figure(inertias: dict[int, float], chosen_k: int) -> go.Figure:
    """Simple line plot of elbow analysis."""
    ks = sorted(inertias.keys())
    ys = [inertias[k] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=ys, mode="lines+markers", name="inertia"))
    fig.add_trace(
        go.Scatter(
            x=[chosen_k],
            y=[inertias[chosen_k]],
            mode="markers",
            marker=dict(size=14, color="#d62728"),
            name=f"chosen k={chosen_k}",
        )
    )
    fig.update_layout(
        title="Elbow analysis",
        xaxis_title="k",
        yaxis_title="inertia (SSE)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
