import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from lorenz.lorenz_systems import LorenzSystems
import traceback
import sys

# Initialize Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    external_scripts=[
        'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'
    ]
)

# Equations Mapping
EQUATIONS = {
    'L63': r"""
$$\frac{dx}{dt} = \sigma(y - x)$$

$$\frac{dy}{dt} = x(\rho - z) - y$$

$$\frac{dz}{dt} = xy - \beta z$$
""",
    'L96': r"""
For $i = 1, \dots, N$:
$$\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F$$
"""
}

# Sidebar Layout
sidebar = html.Div(
    [
        html.H2("Lorenz Simulator", className="display-6"),
        html.Hr(),
        html.P("Interactive Lorenz Dynamics", className="lead"),
        
        dbc.Label("Select Lorenz Model"),
        dcc.Dropdown(
            id="model-select",
            options=[
                {"label": "Lorenz 63", "value": "L63"},
                {"label": "Lorenz 96", "value": "L96"},
            ],
            value="L63",
            clearable=False,
        ),
        
        # Equation Area
        html.Div(
            id="equation-display", 
            className="mt-3 mb-3 p-3 bg-light border rounded",
            children=dcc.Markdown(EQUATIONS['L63'], mathjax=True)
        ),

        # Parameters
        html.Div([
            # L63 Params
            html.Div([
                dbc.Label("Sigma (σ)"),
                dbc.Input(id="sigma-input", type="number", value=10.0, step=0.1),
                dbc.Label("Beta (β)"),
                dbc.Input(id="beta-input", type="number", value=2.6, step=0.1),
                dbc.Label("Rho (ρ)"),
                dbc.Input(id="rho-input", type="number", value=28.0, step=0.1),
                dbc.Label("Initial Y0"),
                dbc.Input(id="y0-input", type="number", value=10.0, step=0.1),
                dbc.Label("Initial Z0"),
                dbc.Input(id="z0-input", type="number", value=1.0, step=0.1),
            ], id="l63-params-div"),
            
            # L96 Params
            html.Div([
                dbc.Label("Forcing (F)"),
                dbc.Input(id="f-input", type="number", value=8.0, step=0.1),
                dbc.Label("Dimension (N)"),
                dbc.Input(id="n-input", type="number", value=40, step=1),
                html.Hr(),
                dbc.Label("Display Variables (indices)"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="v1-idx", type="number", value=0, min=0, step=1)),
                    dbc.Col(dbc.Input(id="v2-idx", type="number", value=1, min=0, step=1)),
                    dbc.Col(dbc.Input(id="v3-idx", type="number", value=2, min=0, step=1)),
                ]),
            ], id="l96-params-div", style={"display": "none"}),
        ], className="mt-3"),
        
        # Simulation Settings
        dbc.Card([
            dbc.CardHeader("Global Settings"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("dt"),
                        dbc.Input(id="dt-input", type="number", value=0.01, step=0.01),
                    ]),
                    dbc.Col([
                        dbc.Label("Steps"),
                        dbc.Input(id="steps-input", type="number", value=500, step=100),
                    ]),
                ]),
                dbc.Label("Initial X0 / State[0]"),
                dbc.Input(id="x0-input", type="number", value=4.0, step=0.1),
                html.Hr(),
                dbc.Label("Ensemble Size"),
                dcc.Slider(id="ens-size-slider", min=1, max=20, step=1, value=5, marks={1: '1', 5: '5', 10: '10', 20: '20'}),
                dbc.Label("Perturbation Std"),
                dbc.Input(id="pert-input", type="number", value=0.05, step=0.01),
            ])
        ], className="mt-3"),

        dbc.Button("🚀 Run Simulation", id="run-btn", color="primary", className="mt-4 w-100", size="lg"),
    ],
    className="sidebar shadow-sm",
    style={
        "position": "fixed", "top": 0, "left": 0, "bottom": 0, 
        "width": "26rem", "padding": "2rem 1rem", "overflowY": "auto",
        "backgroundColor": "#ffffff", "borderRight": "1px solid #dee2e6"
    },
)

content = html.Div(
    [
        dcc.Loading(
            id="loading-sim",
            type="circle",
            children=[
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("3D Dynamic Visualization"),
                        dbc.CardBody(dcc.Graph(id="3d-plot", style={"height": "60vh"}))
                    ], className="shadow-sm border-0"), width=12),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("X (t)"), dbc.CardBody(dcc.Graph(id="x-plot", style={"height": "22vh"}))], className="shadow-sm border-0"), width=4),
                    dbc.Col(dbc.Card([dbc.CardHeader("Y (t)"), dbc.CardBody(dcc.Graph(id="y-plot", style={"height": "22vh"}))], className="shadow-sm border-0"), width=4),
                    dbc.Col(dbc.Card([dbc.CardHeader("Z (t)"), dbc.CardBody(dcc.Graph(id="z-plot", style={"height": "22vh"}))], className="shadow-sm border-0"), width=4),
                ], className="mt-4"),
            ]
        ),
        dbc.Alert(id="stats-display", color="primary", className="mt-4 shadow-sm")
    ],
    style={"marginLeft": "28rem", "marginRight": "2rem", "paddingTop": "2rem"},
)

app.layout = html.Div([sidebar, content])

# --- Callbacks ---

# Clientside callback to trigger MathJax typesetting after equation update
app.clientside_callback(
    """
    function(children) {
        function typeset() {
            if (window.MathJax && window.MathJax.Hub) {
                window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, "equation-display"]);
            } else {
                // Retry in 500ms if MathJax not ready
                setTimeout(typeset, 500);
            }
        }
        typeset();
        return window.dash_clientside.no_update;
    }
    """,
    Output("equation-display", "id"),
    Input("equation-display", "children"),
)

@app.callback(
    [Output("l63-params-div", "style"),
     Output("l96-params-div", "style"),
     Output("equation-display", "children")],
    [Input("model-select", "value")]
)
def toggle_ui(model):
    if model == 'L63':
        return {"display": "block"}, {"display": "none"}, dcc.Markdown(EQUATIONS['L63'], mathjax=True)
    else:
        return {"display": "none"}, {"display": "block"}, dcc.Markdown(EQUATIONS['L96'], mathjax=True)

@app.callback(
    [Output("3d-plot", "figure"),
     Output("x-plot", "figure"),
     Output("y-plot", "figure"),
     Output("z-plot", "figure"),
     Output("stats-display", "children")],
    [Input("run-btn", "n_clicks")],
    [State("model-select", "value"),
     State("dt-input", "value"),
     State("steps-input", "value"),
     State("x0-input", "value"),
     State("y0-input", "value"),
     State("z0-input", "value"),
     State("ens-size-slider", "value"),
     State("pert-input", "value"),
     State("sigma-input", "value"),
     State("beta-input", "value"),
     State("rho-input", "value"),
     State("f-input", "value"),
     State("n-input", "value"),
     State("v1-idx", "value"),
     State("v2-idx", "value"),
     State("v3-idx", "value")]
)
def execute_sim(n_clicks, model, dt, steps, x0, y0, z0, ens_size, pert, sigma, beta, rho, f_param, n_vars, v1, v2, v3):
    if n_clicks is None:
        return [go.Figure()]*4 + ["Ready to simulate..."]
    
    try:
        steps = int(steps or 500)
        dt = float(dt or 0.01)
        time = np.linspace(0, steps*dt, steps)
        
        if model == 'L63':
            x_init = [float(x0 or 4.0), float(y0 or 10.0), float(z0 or 1.0)]
            params = {'sigma': float(sigma or 10.0), 'beta': float(beta or 2.6), 'rho': float(rho or 28.0)}
            labels = ['X', 'Y', 'Z']
            indices = [0, 1, 2]
        else:
            N = int(n_vars if n_vars is not None else 40)
            x_init = np.ones(N) * float(x0 if x0 is not None else 4.0)
            # Add some slight variation to the truth state in L96 to see patterns faster
            x_init[0] += 0.01
            params = {'F': float(f_param if f_param is not None else 8.0)}
            
            # Robust parsing of display indices
            v_idxs = [v1, v2, v3]
            defaults = [0, 1, 2]
            indices = []
            for i, v in enumerate(v_idxs):
                try:
                    val = int(v) if v is not None else defaults[i]
                except (ValueError, TypeError):
                    val = defaults[i]
                indices.append(min(max(0, val), N-1))
            
            labels = [f'x<sub>{i+1}</sub>' for i in indices]

        # Run Truth
        true_traj = LorenzSystems.generate_trajectory('63' if model == 'L63' else '96', x_init, dt, steps, **params)
        
        # Run Ensemble
        ens_trajs = []
        for _ in range(int(ens_size or 5)):
            x0_pert = np.array(x_init) + np.random.normal(0, float(pert or 0.05), len(x_init))
            ens_trajs.append(LorenzSystems.generate_trajectory('63' if model == 'L63' else '96', x0_pert, dt, steps, **params))

        # Build Plots
        fig_3d = go.Figure()
        for e in ens_trajs:
            fig_3d.add_trace(go.Scatter3d(
                x=e[:,indices[0]], y=e[:,indices[1]], z=e[:,indices[2]], 
                mode='lines', line=dict(color='red', width=1), opacity=0.3, showlegend=False
            ))
        fig_3d.add_trace(go.Scatter3d(
            x=true_traj[:,indices[0]], y=true_traj[:,indices[1]], z=true_traj[:,indices[2]], 
            mode='lines', line=dict(color='darkblue', width=4), name='Truth'
        ))
        # Add start point marker
        fig_3d.add_trace(go.Scatter3d(
            x=[true_traj[0, indices[0]]], 
            y=[true_traj[0, indices[1]]], 
            z=[true_traj[0, indices[2]]],
            mode='markers',
            marker=dict(size=8, color='limegreen', symbol='circle'),
            name='Start Location'
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title=labels[0], 
                yaxis_title=labels[1], 
                zaxis_title=labels[2]
            ), 
            margin=dict(l=0, r=0, b=0, t=0)
        )

        def mk1d(idx_in_data, lbl):
            f = go.Figure()
            for e in ens_trajs: 
                f.add_trace(go.Scatter(x=time, y=e[:,idx_in_data], mode='lines', line=dict(color='red', width=1), opacity=0.3, showlegend=False))
            f.add_trace(go.Scatter(x=time, y=true_traj[:,idx_in_data], mode='lines', line=dict(color='darkblue', width=2), name='Truth'))
            f.update_layout(xaxis_title="Time", yaxis_title=lbl, margin=dict(l=20, r=20, b=20, t=20), showlegend=False)
            return f

        stats = f"Simulation Complete. Final State ({labels[0]}, {labels[1]}, {labels[2]}): {true_traj[-1, indices]}"
        return fig_3d, mk1d(indices[0], labels[0]), mk1d(indices[1], labels[1]), mk1d(indices[2], labels[2]), stats
    except Exception as e:
        print(f"Error in Callback: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        return [go.Figure()]*4 + [f"Error: {str(e)}"]

if __name__ == '__main__':
    app.run(debug=True, port=5006, host='0.0.0.0')
