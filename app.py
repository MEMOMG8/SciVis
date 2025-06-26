import pandas as pd
import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output

# ───────────────────────────────────────────────────────────────
# 1  DATA
# ───────────────────────────────────────────────────────────────
FILE_PATH = "Dataset_VisContest_Rapid_Alloy_development_v3.txt"

COLUMNS_TO_PLOT = [
    "Al", "Si", "Cu", "Mg", "Fe", "delta_T", "eut. frac.[%]",
    "YS(MPa)", "hardness(Vickers)",
    "Therm.conductivity(W/(mK))", "Therm. diffusivity(m2/s)",
]

USE_COLS = COLUMNS_TO_PLOT + ["CSC"]          # <— now it’s defined

#   read only those 12 + 1 columns, cast to float32  → ~9 MB RAM
df = (
    pd.read_csv(FILE_PATH, sep="\t", usecols=USE_COLS)
      .astype("float32")
)

df_filtered = df.dropna().reset_index(drop=True)
df_filtered["CSC_clipped"] = df_filtered["CSC"].clip(0, 1)

# PCA -----------------------------------------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_filtered[COLUMNS_TO_PLOT])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

pc1_var, pc2_var = pca.explained_variance_ratio_[:2] * 100
components  = pca.components_.T
ARROW_SCALE = 3

MIN_VALS = df_filtered[COLUMNS_TO_PLOT].min()
MAX_VALS = df_filtered[COLUMNS_TO_PLOT].max()

# ───────────────────────────────────────────────────────────────
# 2  DASH LAYOUT
# ───────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
server = app.server      # for Render / gunicorn

app.layout = html.Div(
    [
        html.H3("Interactive Alloy Explorer • PCA | Radar | Parallel"),
        html.Div(
            [
                html.Label("Feature arrows:"),
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
                html.Label("Display mode:"),
                dcc.RadioItems(
                    id="data-mode",
                    options=[
                        {"label": "All Points",    "value": "all"},
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

# ───────────────────────────────────────────────────────────────
# 3-A  PCA & RADAR CALLBACK
# ───────────────────────────────────────────────────────────────
@app.callback(
    Output("pca-graph",   "figure"),
    Output("radar-chart", "figure"),
    Input("feature-checklist", "value"),
    Input("data-mode",          "value"),
    Input("pca-graph",          "hoverData"),
    prevent_initial_call=False,
)
def update_pca_and_radar(selected_features, mode, hoverData):

    # ——— PCA scatter/biplot ———
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
                text=df_filtered.index.astype(str),  # index → hover/selection id
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
        xaxis_title=f"PC1 ({pc1_var:.1f} % var.)",
        yaxis_title=f"PC2 ({pc2_var:.1f} % var.)",
        title="PCA Biplot",
        hovermode="closest",
        dragmode="lasso",
        plot_bgcolor="white",
        margin=dict(l=60, r=60, t=50, b=60),
    )

    # which point is hovered? default 0
    idx = 0
    if hoverData and hoverData.get("points"):
        try:
            idx = int(hoverData["points"][0]["text"])
        except Exception:
            idx = 0

    # ——— radar for that alloy ———
    real_vals   = df_filtered.loc[idx, COLUMNS_TO_PLOT]
    scaled_vals = ((real_vals - MIN_VALS) / (MAX_VALS - MIN_VALS)).fillna(0)

    r_scaled = (scaled_vals.tolist() + [scaled_vals.iloc[0]])
    r_real   = real_vals.tolist() + [real_vals.iloc[0]]
    theta    = COLUMNS_TO_PLOT + [COLUMNS_TO_PLOT[0]]

    radar_fig = go.Figure()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=r_scaled,
            theta=theta,
            fill="toself",
            name=f"Alloy {idx}",
            customdata=np.array(r_real),
            hovertemplate="<b>%{theta}</b><br>Value: %{customdata}<extra></extra>",
            line=dict(width=2),
        )
    )
    radar_fig.update_layout(
        title=f"Feature profile — alloy {idx}",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(rotation=90),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=50),
    )

    return pca_fig, radar_fig


# ───────────────────────────────────────────────────────────────
# 3-B  PARALLEL COORDS CALLBACK (unchanged)
# ───────────────────────────────────────────────────────────────
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
            color_continuous_scale=["rgba(200,200,200,0.2)"]*2,
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
        title = f"Parallel Coordinates — {len(sel_idx)} alloy(s) selected"
    else:
        fig = px.parallel_coordinates(
            df_filtered,
            dimensions=COLUMNS_TO_PLOT,
            color="CSC_clipped",
            color_continuous_scale=px.colors.diverging.Tealrose,
            range_color=[0, 1],
        )
        title = "Parallel Coordinates — all alloys"

    fig.update_layout(title=title, margin=dict(l=50, r=50, t=70, b=70))
    return fig


# ───────────────────────────────────────────────────────────────
# 4  MAIN
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(debug=True)
