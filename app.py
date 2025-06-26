import pandas as pd
import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA
FILE_PATH = "Dataset_VisContest_Rapid_Alloy_development_v3.txt"

COLUMNS_TO_PLOT = [
    "Al", "Si", "Cu", "Mg", "Fe", "delta_T", "eut. frac.[%]",
    "YS(MPa)", "hardness(Vickers)",
    "Therm.conductivity(W/(mK))", "Therm. diffusivity(m2/s)"
]

df = pd.read_csv(FILE_PATH, sep="\t")
df_filtered = df[COLUMNS_TO_PLOT + ["CSC"]].dropna().reset_index(drop=True)
df_filtered["CSC_clipped"] = df_filtered["CSC"].clip(0, 1)

# PCA -------------------------------------------------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_filtered[COLUMNS_TO_PLOT])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

pc1_var, pc2_var = pca.explained_variance_ratio_[:2] * 100
components  = pca.components_.T
ARROW_SCALE = 3

MIN_VALS = df_filtered[COLUMNS_TO_PLOT].min()
MAX_VALS = df_filtered[COLUMNS_TO_PLOT].max()

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DASH LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H3("Interactive Alloy Explorer: PCA • Radar • Parallel Coords"),
        html.Div(
            [
                html.Label("Select Features to Show Arrows:"),
                dcc.Checklist(
                    id="feature-checklist",
                    options=[{"label": c, "value": c} for c in COLUMNS_TO_PLOT],
                    value=COLUMNS_TO_PLOT,
                    inline=True,
                ),
            ]
        ),
        html.Div(
            [
                html.Label("Select Data Display Mode:"),
                dcc.RadioItems(
                    id="data-mode",
                    options=[
                        {"label": "All Points", "value": "all"},
                        {"label": "Vectors Only", "value": "vectors"},
                    ],
                    value="all",
                    inline=True,
                ),
            ]
        ),
        dcc.Graph(id="pca-graph",      style={"height": "650px"}),
        dcc.Graph(id="radar-chart",    style={"height": "420px"}),
        dcc.Graph(id="parallel-graph", style={"height": "600px"}),
    ],
    style={"maxWidth": "1600px", "margin": "0 auto"},
)

# ──────────────────────────────────────────────────────────────────────────────
# 3-A. PCA & RADAR
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("pca-graph", "figure"),
    Output("radar-chart", "figure"),
    Input("feature-checklist", "value"),
    Input("data-mode", "value"),
    Input("pca-graph", "hoverData"),
    prevent_initial_call=False,
)
def update_pca_and_radar(selected_features, mode, hoverData):

    # ╭─ PCA ───────────────────────────────────────────────────────────────╮
    pca_fig = go.Figure()

    if mode != "vectors":
        pca_fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=df_filtered["CSC_clipped"],
                    colorscale="Tealrose",
                    cmin=0, cmax=1,
                    colorbar=dict(title="CSC", x=1.02),
                ),
                text=df_filtered.index.astype(str),
                hoverinfo="text",
                showlegend=False,
            )
        )

    for i, feat in enumerate(COLUMNS_TO_PLOT):
        if feat in selected_features:
            pca_fig.add_trace(
                go.Scatter(
                    x=[0, components[i, 0] * ARROW_SCALE],
                    y=[0, components[i, 1] * ARROW_SCALE],
                    mode="lines+text",
                    text=[None, feat],
                    textposition="top center",
                    line=dict(color="black", width=2),
                    showlegend=False,
                )
            )

    pca_fig.update_layout(
        xaxis_title=f"PC1 ({pc1_var:.1f}% var)",
        yaxis_title=f"PC2 ({pc2_var:.1f}% var)",
        title="PCA Biplot of Alloy Compositions and CSC",
        hovermode="closest",
        dragmode="lasso",
        plot_bgcolor="white",
        margin=dict(l=60, r=60, t=50, b=60),
    )
    # ╰──────────────────────────────────────────────────────────────────────╯

    # ╭─ Determine hovered index ───────────────────────────────────────────╮
    idx = 0
    if hoverData and hoverData.get("points"):
        try:
            idx = int(hoverData["points"][0]["text"])
        except Exception:
            idx = 0
    # ╰──────────────────────────────────────────────────────────────────────╯

    # ╭─ RADAR ──────────────────────────────────────────────────────────────╮
    real_vals   = df_filtered.loc[idx, COLUMNS_TO_PLOT]
    scaled_vals = ((real_vals - MIN_VALS) / (MAX_VALS - MIN_VALS)).fillna(0)

    theta    = COLUMNS_TO_PLOT + [COLUMNS_TO_PLOT[0]]
    r_scaled = scaled_vals.tolist() + [scaled_vals.iloc[0]]
    r_real   = real_vals.tolist()   + [real_vals.iloc[0]]

    radar_fig = go.Figure()

    radar_fig.add_trace(
        go.Scatterpolar(
            r=r_scaled,
            theta=theta,
            fill="toself",
            name=f"Alloy {idx}",
            hovertemplate="<b>%{theta}</b><br>Value: %{customdata:.4g}<extra></extra>",
            customdata=np.array(r_real),
        )
    )

    # ——— stats cards ———
    for k, feat in enumerate(COLUMNS_TO_PLOT):
        angle_deg = k * 360 / len(COLUMNS_TO_PLOT)
        angle_rad = math.radians(angle_deg)

        # alternate two rings to avoid crowding
        r_card   = 1.25 if k % 2 == 0 else 1.40
        x_paper  = 0.5 + 0.5 * r_card * math.cos(angle_rad)
        y_paper  = 0.5 + 0.5 * r_card * math.sin(angle_rad)

        # orient text block outward
        xanchor = "left"  if -90 < angle_deg < 90  else "right"
        yanchor = "bottom" if   0 < angle_deg < 180 else "top"

        mid_val = 0.5 * (MIN_VALS[feat] + MAX_VALS[feat])

        txt = (
            f"<b>{feat}</b><br>"
            f"min&nbsp;{MIN_VALS[feat]:.3g}<br>"
            f"mid&nbsp;{mid_val:.3g}<br>"
            f"max&nbsp;{MAX_VALS[feat]:.3g}"
        )

        radar_fig.add_annotation(
            x=x_paper, y=y_paper,
            xref="paper", yref="paper",
            text=txt,
            showarrow=False,
            font=dict(size=9, color="black"),
            align="left",
            textangle=0,
            xanchor=xanchor,
            yanchor=yanchor,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=0.5,
        )

    radar_fig.update_layout(
        title=f"Hovered Alloy Feature Breakdown (Index {idx})",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(rotation=90, showticklabels=False),
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=20, b=20),
    )
    # ╰──────────────────────────────────────────────────────────────────────╯

    return pca_fig, radar_fig


# ──────────────────────────────────────────────────────────────────────────────
# 3-B. PARALLEL-COORDS (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("parallel-graph", "figure"),
    Input("pca-graph", "selectedData"),
    prevent_initial_call=False,
)
def update_parallel_coordinates(selection):

    if selection and selection.get("points"):
        sel_idx = [int(p["text"]) for p in selection["points"] if "text" in p]
        df_sel, df_rest = df_filtered.iloc[sel_idx], df_filtered.drop(sel_idx)

        fig = px.parallel_coordinates(
            df_rest,
            dimensions=COLUMNS_TO_PLOT,
            color="CSC_clipped",
            color_continuous_scale=["rgba(200,200,200,0.2)",
                                    "rgba(200,200,200,0.2)"],
            range_color=[0, 1],
        )
        fig_sel = px.parallel_coordinates(
            df_sel,
            dimensions=COLUMNS_TO_PLOT,
            color="CSC_clipped",
            color_continuous_scale=px.colors.diverging.Tealrose,
            range_color=[0, 1],
        )
        for tr in fig_sel.data:
            fig.add_trace(tr)
        title = f"Parallel-Coordinates — {len(sel_idx)} Alloy(s) Selected"
    else:
        fig = px.parallel_coordinates(
            df_filtered,
            dimensions=COLUMNS_TO_PLOT,
            color="CSC_clipped",
            color_continuous_scale=px.colors.diverging.Tealrose,
            range_color=[0, 1],
        )
        title = "Parallel-Coordinates — All Alloys"

    fig.update_layout(title=title, margin=dict(l=60, r=60, t=50, b=60))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(debug=True)
