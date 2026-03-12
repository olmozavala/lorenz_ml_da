import dash
from dash import dcc, html, Input, Output, State, ALL, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import threading
import time
import os
import sys
import traceback
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from MachineLearning import DenseNN, ResDenseNN, LSTMNN, save_model, load_model
from datasets.LorenzDataset import LorenzDataset, _SYSTEM_DIMS, _SYSTEM_IC_CENTER
from Training import train_model, EarlyStopping, recursive_rollout
from lorenz.lorenz_systems import LorenzSystems

# --- App Settings ---
os.makedirs("models", exist_ok=True)

# Global state to track training progress
training_state = {
    'is_training': False,
    'stop_requested': False,
    'history': {'epochs': [], 'train_loss': [], 'val_loss': []},
    'current_epoch': 0,
    'rollout_steps': 1,
    'model_name': 'dash_model',
    'sample_trajectory': None  # Store one raw trajectory for preview
}

# Global state to track evaluation progress (mirrors training_state pattern)
eval_state = {
    'is_running': False,
    'progress': 0,
    'total': 0,
    'result': None,
    'error': None,
}

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LUX], 
                suppress_callback_exceptions=True,
                title="Lorenz ML Studio")

# --- Layout Components ---

sidebar = html.Div([
    html.H4("ML Training Studio", className="mt-2"),
    html.Hr(),
    
    dbc.Accordion([
        dbc.AccordionItem([
            dbc.Label("Model Type"),
            dcc.Dropdown(
                id="model-type",
                options=[
                    {"label": "Dense NN", "value": "Dense"},
                    {"label": "Res-Dense NN", "value": "ResDense"},
                    {"label": "LSTM", "value": "LSTM"},
                ],
                value="ResDense"
            ),
            dbc.Label("Lorenz System"),
            dcc.Dropdown(
                id="system-type",
                options=[
                    {"label": "Lorenz 63", "value": "63"},
                    {"label": "Lorenz 96 (N=40)", "value": "96"},
                    {"label": "Lorenz 2005 Model III (N=480)", "value": "05"},
                ],
                value="63"
            ),
            dbc.Label("History Steps (Input size)"),
            dbc.Input(id="prev-steps", type="number", value=3),
        ], title="Architecture"),

        dbc.AccordionItem([
            html.Small(
                "L63 uses σ=10 / ρ=28 / β=8/3 (fixed). "
                "L96 uses F=8.0 (fixed). "
                "The fields below apply only to Lorenz 2005 Model III.",
                className="text-muted d-block mb-2"
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Label("L05 Forcing (F)"),
                    dbc.Input(id="l05-F", type="number", value=15.0, step=0.5),
                ]),
                dbc.Col([
                    dbc.Label("L05 Spatial Scale (K)"),
                    dbc.Input(id="l05-K", type="number", value=32, step=1),
                ]),
            ]),
        ], title="System Parameters"),

        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([dbc.Label("dt"), dbc.Input(id="dt", type="number", value=0.001)]),
                dbc.Col([dbc.Label("Save_Dt (Skip)"), dbc.Input(id="save-dt", type="number", value=10)]),
            ]),
            dbc.Row([
                dbc.Col([dbc.Label("Random Locations"), dbc.Input(id="num-locs", type="number", value=10)]),
                dbc.Col([dbc.Label("Samples/Loc"), dbc.Input(id="ns-per-loc", type="number", value=10000)]),
            ]),
            dbc.Label("Split Train/Val/Test (%)"),
            dbc.Row([
                dbc.Col(dbc.Input(id="split-train", type="number", value=70)),
                dbc.Col(dbc.Input(id="split-val", type="number", value=20)),
                dbc.Col(dbc.Input(id="split-test", type="number", value=10)),
            ]),
        ], title="Data Generation"),
        
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([dbc.Label("Batch Size"), dbc.Input(id="batch-size", type="number", value=2048)]),
                dbc.Col([dbc.Label("Patience"), dbc.Input(id="patience", type="number", value=20)]),
            ]),
            dbc.Label("Hidden Layers (e.g., 64,64,32)"),
            dbc.Input(id="hidden-layers", type="text", value="64,64,32"),
            dbc.Label("Loss Function"),
            dcc.Dropdown(
                id="loss-func",
                options=[{"label": "MSE", "value": "MSE"}, {"label": "Huber", "value": "Huber"}],
                value="MSE"
            ),
        ], title="Training Params"),
    ], always_open=True),
    
    dbc.Button("🔥 Start Training", id="start-btn", color="danger", className="mt-4 w-100", size="lg"),
    dbc.Button("🛑 Stop Training", id="stop-btn", color="secondary", className="mt-2 w-100", disabled=True),
    dbc.Card([
        dbc.CardHeader("Model Configuration"),
        dbc.CardBody(html.Pre(id="model-config-summary", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "fontSize": "0.8rem", "margin": 0},
                             children="Select a model in the Evaluation tab."))
    ], className="mt-3"),
], className="sidebar p-4 bg-light shadow", style={"width": "26rem", "position": "fixed", "top": 0, "left": 0, "bottom": 0, "overflowY": "auto"})

content = html.Div([
    dcc.Tabs(id="tabs", value='tab-train', children=[
        dcc.Tab(label='📈 Training', value='tab-train', className="custom-tab", selected_className="custom-tab--selected", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H5("Real-time Learning Curves", className="mt-4"),
                        dcc.Graph(id="loss-graph", style={"height": "550px"}),
                        dcc.Interval(id="interval-update", interval=1000, n_intervals=0, disabled=True),
                    ], lg=6, xs=12),
                    dbc.Col([
                        html.H5("Sample Training Trajectory", className="mt-4"),
                        dcc.Graph(id="sample-traj-plot", style={"height": "550px"}),
                    ], lg=6, xs=12),
                ]),
                dbc.Card([
                    dbc.CardHeader("Training Console"),
                    dbc.CardBody([
                        html.Pre(id="training-log", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "height": "100px", "overflowY": "auto"})
                    ])
                ], className="mt-3")
            ], fluid=True)
        ]),
        dcc.Tab(label='🧪 Evaluation', value='tab-eval', className="custom-tab", selected_className="custom-tab--selected", children=[
            dbc.Container([
                html.H3("Model vs Truth Evaluation", className="mt-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Trained Model"),
                        dcc.Dropdown(id="eval-model-select", options=[]),
                        dbc.Button("🔄 Refresh List", id="refresh-models-btn", color="info", className="mt-2 btn-sm"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Forecast Steps"),
                        dbc.Input(id="eval-steps", type="number", value=200),
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Ensemble Size"),
                        dbc.Input(id="eval-ens-size", type="number", value=20),
                    ], width=2),
                    dbc.Col([
                        dbc.Label("IC Noise (σ)"),
                        dbc.Input(id="eval-noise", type="number", value=0.1, step=0.01),
                    ], width=2),
                    dbc.Col([
                        dbc.Button("⚡ Run Eval", id="eval-run-btn", color="success", className="mt-4 w-100"),
                        html.Div(id="eval-progress", className="mt-1",
                                 style={"fontSize": "0.8rem", "color": "#666"}),
                    ], width=3),
                ]),
                dcc.Interval(id="eval-interval", interval=500, n_intervals=0, disabled=True),
                html.Hr(),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="eval-3d-plot", style={"height": "55vh"}))], className="shadow-sm border-0"), width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Plot 1 Variable"),
                        dcc.Dropdown(id="eval-var-select-0", options=[], value=None,
                                     placeholder="Select variable...", searchable=True),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Plot 2 Variable"),
                        dcc.Dropdown(id="eval-var-select-1", options=[], value=None,
                                     placeholder="Select variable...", searchable=True),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Plot 3 Variable"),
                        dcc.Dropdown(id="eval-var-select-2", options=[], value=None,
                                     placeholder="Select variable...", searchable=True),
                    ], width=4),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Plot Style", className="me-2"),
                        dbc.RadioItems(
                            id="eval-plot-type",
                            options=[{"label": "Line", "value": "line"},
                                     {"label": "Polar", "value": "polar"}],
                            value="line", inline=True,
                        ),
                    ], width="auto"),
                ], className="mt-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="eval-x-plot", style={"height": "35vh"}), width=4),
                    dbc.Col(dcc.Graph(id="eval-y-plot", style={"height": "35vh"}), width=4),
                    dbc.Col(dcc.Graph(id="eval-z-plot", style={"height": "35vh"}), width=4),
                ], className="mt-2")
            ], fluid=True)
        ]),
    ])
], style={"marginLeft": "28rem", "paddingRight": "2rem"})

app.layout = html.Div([
    sidebar, content,
    dcc.Store(id="eval-fine-dt"),
    dcc.Store(id="eval-save-dt"),
    dcc.Store(id="eval-traj-store"),
])

# --- Callbacks ---

def training_thread_func(config):
    global training_state
    try:
        # 1. Dataset Generation
        # Build physics parameters passed to LorenzSystems / LorenzDataset.
        # N is no longer included — _SYSTEM_DIMS in LorenzDataset enforces fixed sizes.
        sys_params = {}
        if config['system_type'] == '96':
            sys_params['F'] = 8.0           # fixed forcing for L96
        elif config['system_type'] == '05':
            sys_params['F'] = float(config.get('l05_F', 15.0))
            sys_params['K'] = int(config.get('l05_K', 32))
            
        dataset = LorenzDataset(
            system_type=config['system_type'],
            dt=config['dt'],
            Ns=config['samples_per_loc'],
            save_Dt=config['save_dt'],
            num_start_locations=config['num_locs'],
            prev_time_steps=config['prev_steps'],
            **sys_params
        )
        
        # Store first raw trajectory for visualization (data_list is pre-normalization)
        training_state['sample_trajectory'] = dataset.data_list[0] if dataset.data_list else None
        
        # Splits
        total = len(dataset)
        train_sz = int(total * config['split_train'] / 100)
        val_sz = int(total * config['split_val'] / 100)
        test_sz = total - train_sz - val_sz
        train_set, val_set, test_set = random_split(dataset, [train_sz, val_sz, test_sz])
        
        # Optimization: Use pin_memory and more workers
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=(device.type=='cuda'))
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], pin_memory=(device.type=='cuda'))
        
        # Model
        input_size = _SYSTEM_DIMS[config['system_type']]  # 3 / 40 / 480 fixed by DAPyr
        hidden_list = [int(x.strip()) for x in config['hidden_layers'].split(',')]

        arch_meta = {
            'model_type':      config['model_type'],
            'input_size':      input_size,
            'prev_time_steps': config['prev_steps'],
            'hidden_layers':   hidden_list,
            'system':          config['system_type'],
            'N':               input_size,         # legacy field; same as input_size
            'system_params':   sys_params,         # stored for eval reconstruction
        }
        
        if config['model_type'] == 'Dense':
            model = DenseNN(input_size, config['prev_steps'], input_size, hidden_list, nn.ReLU, None)
        elif config['model_type'] == 'ResDense':
            model = ResDenseNN(input_size, config['prev_steps'], input_size, hidden_list, nn.ReLU, None)
        elif config['model_type'] == 'LSTM':
            model = LSTMNN(input_size, config['prev_steps'], input_size, hidden_list[0])
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss() if config['loss_func'] == 'MSE' else nn.HuberLoss()
        early_stopping = EarlyStopping(patience=config['patience'])
        
        def progress(epoch, t_loss, v_loss, rollout):
            training_state['history']['epochs'].append(epoch)
            training_state['history']['train_loss'].append(t_loss)
            training_state['history']['val_loss'].append(v_loss)
            training_state['current_epoch'] = epoch
            training_state['rollout_steps'] = rollout
            if training_state['stop_requested']:
                return True # Should implement stop check in Training.py loop
            
        model_name = f"{config['model_type']}_L{config['system_type']}_{int(time.time())}"
        training_state['model_name'] = model_name
        
        # Save config + architecture + scaler to YAML right away so it's available for eval even if training is stopped
        config['architecture'] = arch_meta
        config['train_mean'] = dataset.scaler.mean_.tolist()
        config['train_std'] = dataset.scaler.scale_.tolist()
        config_path = f"models/{model_name}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        train_model(model, model_name, train_loader, val_loader, criterion, optimizer, 500, early_stopping, progress_callback=progress, device=device)
        
        save_model(model, f"models/{model_name}", dataset.scaler.mean_, dataset.scaler.scale_, arch_meta)
        
    except Exception as e:
        print(f"Error in training thread: {e}")
        traceback.print_exc()
    finally:
        training_state['is_training'] = False

@app.callback(
    [Output("interval-update", "disabled"),
     Output("start-btn", "disabled"),
     Output("stop-btn", "disabled"),
     Output("training-log", "children")],
    [Input("start-btn", "n_clicks")],
    [State("model-type", "value"), State("system-type", "value"),
     State("dt", "value"), State("save-dt", "value"), State("prev-steps", "value"),
     State("num-locs", "value"), State("ns-per-loc", "value"), State("batch-size", "value"),
     State("patience", "value"), State("hidden-layers", "value"), State("loss-func", "value"),
     State("split-train", "value"), State("split-val", "value"), State("split-test", "value"),
     State("l05-F", "value"), State("l05-K", "value")],
     prevent_initial_call=True
)
def start_tra(n, m, s, dt, s_dt, prev, locs, ns, bs, pat, hidden, loss, st, sv, ste, l05_f, l05_k):
    global training_state
    training_state = {
        'is_training': True, 'stop_requested': False,
        'history': {'epochs': [], 'train_loss': [], 'val_loss': []},
        'current_epoch': 0, 'rollout_steps': 1, 'model_name': '',
        'sample_trajectory': None, '_traj_drawn': False
    }

    config = {
        'model_type': m, 'system_type': s, 'dt': dt, 'save_dt': s_dt,
        'prev_steps': prev, 'num_locs': locs, 'samples_per_loc': ns,
        'batch_size': bs, 'patience': pat, 'hidden_layers': hidden, 'loss_func': loss,
        'split_train': st, 'split_val': sv, 'split_test': ste,
        'l05_F': l05_f if l05_f is not None else 15.0,
        'l05_K': l05_k if l05_k is not None else 32,
    }
    
    threading.Thread(target=training_thread_func, args=(config,), daemon=True).start()
    return False, True, False, "Initializating dataset and model...\n"

@app.callback(
    [Output("stop-btn", "disabled", allow_duplicate=True),
     Output("training-log", "children", allow_duplicate=True)],
    [Input("stop-btn", "n_clicks")],
    prevent_initial_call=True
)
def stop_training(n):
    global training_state
    training_state['stop_requested'] = True
    return True, "Stop requested. Training will halt after current epoch..."

@app.callback(
    [Output("loss-graph", "figure"),
     Output("training-log", "children", allow_duplicate=True),
     Output("start-btn", "disabled", allow_duplicate=True),
     Output("stop-btn", "disabled", allow_duplicate=True),
     Output("interval-update", "disabled", allow_duplicate=True),
     Output("sample-traj-plot", "figure")],
    [Input("interval-update", "n_intervals")],
    prevent_initial_call=True
)
def update_metrics(n):
    history = training_state['history']
    fig = go.Figure()
    if history['epochs']:
        fig.add_trace(go.Scatter(x=history['epochs'], y=history['train_loss'], name='Train', line=dict(color='#e74c3c')))
        fig.add_trace(go.Scatter(x=history['epochs'], y=history['val_loss'], name='Val', line=dict(color='#3498db')))
    fig.update_layout(template="simple_white", yaxis_type="log", margin=dict(l=40, r=40, t=40, b=40),
                      yaxis=dict(exponentformat='power'),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    # Draw the trajectory plot once data is available
    traj = training_state.get('sample_trajectory')
    if traj is not None and not training_state.get('_traj_drawn'):
        traj_fig = go.Figure()
        nx = traj.shape[1]
        if nx >= 3:
            traj_fig.add_trace(go.Scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode='lines', line=dict(color='royalblue', width=3), name='Training Sample'))
            traj_fig.add_trace(go.Scatter3d(x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]], mode='markers', marker=dict(size=6, color='limegreen'), name='Start'))
            traj_fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), margin=dict(l=0, r=0, b=0, t=0))
            training_state['_traj_drawn'] = True
    else:
        traj_fig = dash.no_update
    
    done = not training_state['is_training'] and training_state['current_epoch'] > 0
    
    if done:
        status = f"DONE. Saved: {training_state['model_name']}.pth"
        return fig, status, False, True, True, traj_fig
    
    status = f"Running Epoch: {training_state['current_epoch']} | Rollout Stage: {training_state['rollout_steps']}"
    return fig, status, True, False, False, traj_fig



@app.callback(
    Output("eval-model-select", "options"),
    [Input("refresh-models-btn", "n_clicks"), Input("tabs", "value")]
)
def refresh_models_list(n, tab):
    files = [f for f in os.listdir("models") if f.endswith(".pth")]
    return [{"label": f, "value": f} for f in sorted(files, reverse=True)]

@app.callback(
    [Output("model-config-summary", "children"),
     Output("eval-fine-dt", "data"),
     Output("eval-save-dt", "data"),
     Output("eval-var-select-0", "options"),
     Output("eval-var-select-1", "options"),
     Output("eval-var-select-2", "options"),
     Output("eval-var-select-0", "value"),
     Output("eval-var-select-1", "value"),
     Output("eval-var-select-2", "value")],
    Input("eval-model-select", "value"),
    prevent_initial_call=True
)
def show_model_config(model_file):
    n_extra = 8  # number of outputs beyond the summary text
    no_updates = tuple(dash.no_update for _ in range(n_extra))
    if not model_file:
        return "Select a model to see its configuration.", *no_updates
    try:
        path = os.path.join("models", model_file)

        # Find matching YAML (strip _best_model suffix if present)
        base = model_file.replace('_best_model.pth', '').replace('.pth', '')
        yml_path = os.path.join("models", f"{base}.yml")

        # Try architecture from checkpoint first, fall back to YAML
        meta = {}
        if os.path.exists(path):
            cp = torch.load(path, map_location='cpu', weights_only=False)
            meta = cp.get('architecture', {})

        cfg = {}
        if os.path.exists(yml_path):
            with open(yml_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            # YAML may contain architecture if checkpoint doesn't
            if not meta and 'architecture' in cfg:
                meta = cfg['architecture']

        lines = ["═══ Architecture ═══"]
        lines.append(f"  Model Type:      {meta.get('model_type', '?')}")
        lines.append(f"  Lorenz System:   L{meta.get('system', '?')}")
        lines.append(f"  Input Size (nx): {meta.get('input_size', '?')}")
        lines.append(f"  History Steps:   {meta.get('prev_time_steps', '?')}")
        lines.append(f"  Hidden Layers:   {meta.get('hidden_layers', '?')}")

        fine_dt_out = dash.no_update
        save_dt_out = dash.no_update
        if cfg:
            lines.append(f"\n═══ Training Parameters ═══")
            dt_val = cfg.get('dt', 0)
            sdt_val = cfg.get('save_dt', 1)
            lines.append(f"  Integration dt:  {dt_val}")
            lines.append(f"  Save Dt (skip):  {sdt_val}")
            if isinstance(dt_val, (int, float)) and isinstance(sdt_val, (int, float)):
                effective_dt = dt_val * sdt_val
                fine_dt_out = dt_val
                save_dt_out = sdt_val
                lines.append(f"  Effective Δt:    {effective_dt:.4f}")
            lines.append(f"  Random Locs:     {cfg.get('num_locs', '?')}")
            lines.append(f"  Samples/Loc:     {cfg.get('samples_per_loc', '?')}")
            lines.append(f"  Batch Size:      {cfg.get('batch_size', '?')}")
            lines.append(f"  Patience:        {cfg.get('patience', '?')}")
            lines.append(f"  Loss Function:   {cfg.get('loss_func', '?')}")
            lines.append(f"  Split (T/V/Te):  {cfg.get('split_train','?')}% / {cfg.get('split_val','?')}% / {cfg.get('split_test','?')}%")
        else:
            lines.append("\n⚠ No YAML config found for this model.")

        # Build variable selector options based on system type
        nx = meta.get('input_size', 3)
        sys_type = meta.get('system', '63')
        if sys_type == '63':
            var_options = [
                {"label": "X (dim 0)", "value": 0},
                {"label": "Y (dim 1)", "value": 1},
                {"label": "Z (dim 2)", "value": 2},
            ]
            default_vars = [0, 1, 2]
        else:
            var_options = [{"label": f"Z{i}", "value": i} for i in range(nx)]
            default_vars = [0, nx // 4, nx // 2]

        return ("\n".join(lines), fine_dt_out, save_dt_out,
                var_options, var_options, var_options,
                default_vars[0], default_vars[1], default_vars[2])
    except Exception as e:
        return f"Error reading config: {str(e)}", *no_updates


def _generate_one_truth(sys_type, xi, fine_dt, save_dt_int, steps, eval_dt,
                        use_subsampling, eval_params):
    """Generate a single truth trajectory (called from thread pool)."""
    if use_subsampling:
        n_fine = steps * save_dt_int
        traj_fine = LorenzSystems.generate_trajectory(
            sys_type, xi, fine_dt, n_fine, **eval_params)
        return traj_fine[::save_dt_int]
    else:
        return LorenzSystems.generate_trajectory(
            sys_type, xi, eval_dt, steps, **eval_params)


def _eval_thread_func(config):
    """Background thread: generate truth + ML ensembles, update eval_state."""
    global eval_state
    try:
        model_file = config['model_file']
        steps = config['steps']
        ens_size = config['ens_size']
        noise_std = config['noise_std']
        fine_dt_val = config['fine_dt_val']
        save_dt_val = config['save_dt_val']

        path = os.path.join("models", model_file)
        cp = torch.load(path, map_location='cpu', weights_only=False)

        base = model_file.replace('_best_model.pth', '').replace('.pth', '')
        yml_path = os.path.join("models", f"{base}.yml")

        meta = cp.get('architecture', None)
        cfg = {}
        if os.path.exists(yml_path):
            with open(yml_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}

        if meta is None:
            meta = cfg.get('architecture', {})
        if not meta:
            raise ValueError("No architecture metadata found in checkpoint or YAML!")

        nx = meta['input_size']
        prev_steps = meta['prev_time_steps']

        if fine_dt_val is not None and save_dt_val is not None:
            eval_dt = float(fine_dt_val) * int(save_dt_val)
        else:
            eval_dt = 0.01

        mt = meta['model_type']
        if mt in ('Dense', 'DenseNN'):
            m_class = DenseNN
        elif mt in ('ResDense', 'ResDenseNN'):
            m_class = ResDenseNN
        else:
            m_class = LSTMNN
        if mt in ('LSTM', 'LSTMNN'):
            model = m_class(nx, prev_steps, nx, meta['hidden_layers'][0])
        else:
            model = m_class(nx, prev_steps, nx, meta['hidden_layers'], nn.ReLU, None)

        if 'model_state_dict' in cp:
            model.load_state_dict(cp['model_state_dict'])
            mean, std = cp['train_mean'], cp['train_std']
        else:
            model.load_state_dict(cp)
            if 'train_mean' in cfg and 'train_std' in cfg:
                mean = np.array(cfg['train_mean'])
                std = np.array(cfg['train_std'])
            else:
                full_pth = os.path.join("models", f"{base}.pth")
                if os.path.exists(full_pth):
                    full_cp = torch.load(full_pth, map_location='cpu', weights_only=False)
                    mean, std = full_cp['train_mean'], full_cp['train_std']
                else:
                    raise ValueError("Cannot find train_mean/train_std in YAML or checkpoint.")
        model.eval()

        sys_type = meta['system']
        if noise_std is None:
            noise_std = 0.1

        if sys_type == '63':
            x0 = np.array([1.0, 1.0, 1.0])
        else:
            center = _SYSTEM_IC_CENTER.get(sys_type, 8.0)
            x0 = np.ones(nx) * center + np.random.normal(0, 0.1, nx)

        eval_params = meta.get('system_params', {})
        if not eval_params:
            if sys_type == '96':
                eval_params = {'F': 8.0}
            elif sys_type == '05':
                eval_params = {'F': 15.0, 'K': 32}

        use_subsampling = fine_dt_val is not None and save_dt_val is not None
        fine_dt = float(fine_dt_val) if use_subsampling else 0.0
        save_dt_int = int(save_dt_val) if use_subsampling else 1

        # --- Parallel truth ensemble generation ---
        eval_state['total'] = ens_size
        eval_state['progress'] = 0

        # Pre-compute ICs (must be done in main thread for reproducibility)
        ics = []
        for i in range(ens_size):
            xi = x0 + np.random.normal(0, noise_std, nx) if i > 0 else x0.copy()
            ics.append(xi)

        n_workers = min(8, max(1, os.cpu_count() or 4))
        truth_trajs = [None] * ens_size
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for i, xi in enumerate(ics):
                fut = pool.submit(_generate_one_truth, sys_type, xi,
                                  fine_dt, save_dt_int, steps, eval_dt,
                                  use_subsampling, eval_params)
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                truth_trajs[idx] = fut.result()
                eval_state['progress'] += 1

        # --- ML ensemble ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        all_hists = np.stack([t[:prev_steps] for t in truth_trajs])
        hists_norm = (all_hists - mean) / std
        input_seqs = torch.tensor(hists_norm.reshape(ens_size, -1),
                                  dtype=torch.float32).to(device)

        with torch.no_grad():
            preds_norm = recursive_rollout(model, input_seqs,
                                           steps - prev_steps, prev_steps, device)
            preds = preds_norm.cpu().numpy() * std + mean

        ml_trajs = [np.vstack([all_hists[i], preds[i]]) for i in range(ens_size)]

        eval_state['result'] = {
            "truth_trajs": truth_trajs,
            "ml_trajs": ml_trajs,
            "eval_dt": eval_dt,
            "sys_type": sys_type,
            "nx": nx,
            "x0": x0,
        }

    except Exception as e:
        traceback.print_exc()
        eval_state['error'] = str(e)
    finally:
        eval_state['is_running'] = False


@app.callback(
    [Output("eval-interval", "disabled"),
     Output("eval-run-btn", "disabled"),
     Output("eval-progress", "children")],
    Input("eval-run-btn", "n_clicks"),
    [State("eval-model-select", "value"),
     State("eval-steps", "value"),
     State("eval-ens-size", "value"), State("eval-noise", "value"),
     State("eval-fine-dt", "data"), State("eval-save-dt", "data")],
    prevent_initial_call=True
)
def eval_run_start(n, model_file, steps, ens_size, noise_std, fine_dt_val, save_dt_val):
    """Launch evaluation in a background thread, return immediately."""
    global eval_state
    if not model_file:
        return True, False, "No model selected."

    eval_state = {
        'is_running': True,
        'progress': 0,
        'total': ens_size or 20,
        'result': None,
        'error': None,
    }

    config = {
        'model_file': model_file,
        'steps': steps,
        'ens_size': ens_size,
        'noise_std': noise_std,
        'fine_dt_val': fine_dt_val,
        'save_dt_val': save_dt_val,
    }
    threading.Thread(target=_eval_thread_func, args=(config,), daemon=True).start()
    return False, True, "Starting evaluation..."


@app.callback(
    [Output("eval-traj-store", "data"),
     Output("eval-interval", "disabled", allow_duplicate=True),
     Output("eval-run-btn", "disabled", allow_duplicate=True),
     Output("eval-progress", "children", allow_duplicate=True)],
    Input("eval-interval", "n_intervals"),
    prevent_initial_call=True
)
def eval_poll(n):
    """Poll the background evaluation thread and update UI."""
    global eval_state
    prog = eval_state['progress']
    total = eval_state['total']

    if eval_state['is_running']:
        return (dash.no_update, dash.no_update, dash.no_update,
                f"Integrating ensemble member {prog}/{total}...")

    # Done — check for errors
    if eval_state['error']:
        msg = f"Error: {eval_state['error']}"
        return None, True, False, msg

    # Success — send lightweight trigger (data stays server-side in eval_state)
    import time as _time
    return {"ts": _time.time()}, True, False, f"Done ({total} members)."


# ---------------------------------------------------------------------------
# Plot callbacks — each reads from the trajectory store independently
# ---------------------------------------------------------------------------

@app.callback(
    Output("eval-3d-plot", "figure"),
    Input("eval-traj-store", "data"),
    prevent_initial_call=True
)
def update_3d_plot(store_trigger):
    """Render the main 3D/Hovmoller plot from server-side trajectory data."""
    if store_trigger is None or eval_state.get('result') is None:
        return go.Figure()
    try:
        data = eval_state['result']
        truth_trajs = data["truth_trajs"]
        ml_trajs = data["ml_trajs"]
        eval_dt = data["eval_dt"]
        sys_type = data["sys_type"]
        nx = data["nx"]
        x0 = data["x0"]
        steps = len(truth_trajs[0])
        time_axis = np.arange(steps) * eval_dt

        if sys_type == '63':
            zi = 2
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=[x0[0]], y=[x0[1]], z=[x0[zi]],
                mode='markers', marker=dict(size=8, color='limegreen', symbol='diamond'),
                name='Initial Condition', showlegend=True
            ))
            for i, t in enumerate(truth_trajs):
                fig.add_trace(go.Scatter3d(
                    x=t[:, 0], y=t[:, 1], z=t[:, zi],
                    mode='lines', line=dict(color='rgba(0, 0, 180, 0.4)', width=2),
                    showlegend=(i == 0), name='Truth Ensemble'
                ))
            for i, m in enumerate(ml_trajs):
                fig.add_trace(go.Scatter3d(
                    x=m[:, 0], y=m[:, 1], z=m[:, zi],
                    mode='lines', line=dict(color='rgba(255, 50, 50, 0.35)', width=2),
                    showlegend=(i == 0), name='ML Ensemble'
                ))
            fig.update_layout(
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                margin=dict(l=0, r=0, b=0, t=0),
            )
        else:
            mean_truth = np.mean(np.stack(truth_trajs), axis=0)
            mean_ml = np.mean(np.stack(ml_trajs), axis=0)
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=mean_truth.T, x=time_axis, colorscale='RdBu_r',
                colorbar=dict(title='State', x=0.45, len=0.9),
                name='Truth', showscale=True, xaxis='x', yaxis='y',
            ))
            fig.add_trace(go.Heatmap(
                z=mean_ml.T, x=time_axis, colorscale='RdBu_r',
                colorbar=dict(title='ML', x=1.0, len=0.9),
                name='ML', showscale=True, xaxis='x2', yaxis='y',
            ))
            fig.update_layout(
                title=f"L{sys_type} Hovmoller — Truth (left) vs ML (right) ensemble mean",
                xaxis=dict(title='Time', domain=[0, 0.47]),
                xaxis2=dict(title='Time', domain=[0.53, 1.0]),
                yaxis=dict(title='State Index'),
                margin=dict(l=0, r=0, b=0, t=30),
            )
        return fig
    except Exception as e:
        traceback.print_exc()
        err = go.Figure()
        err.update_layout(title=f"Error: {str(e)}", template="simple_white")
        return err


def _mk_aux_plot(var_idx, plot_type="line"):
    """Build a single auxiliary figure (line or polar) for the given variable index."""
    data = eval_state['result']
    truth_trajs = data["truth_trajs"]
    ml_trajs = data["ml_trajs"]
    eval_dt = data["eval_dt"]
    nx = data["nx"]
    sys_type = data["sys_type"]
    idx = int(var_idx) if var_idx is not None else 0
    idx = max(0, min(idx, nx - 1))

    # Build title label
    if sys_type == '63':
        dim_name = {0: 'X', 1: 'Y', 2: 'Z'}.get(idx, f'Dim {idx}')
    else:
        dim_name = f'Z{idx}'

    fig = go.Figure()

    if plot_type == "polar":
        # --- Polar plot: theta wraps every 1 MTU ---
        T_MTU = 1.0
        all_vals = [t[:, idx] for t in truth_trajs] + [m[:, idx] for m in ml_trajs]
        r_min = min(v.min() for v in all_vals)
        r_offset = r_min - 0.1  # shift so minimum radius is 0.1

        for i, t in enumerate(truth_trajs):
            time_arr = np.arange(len(t)) * eval_dt
            theta = (time_arr % T_MTU) / T_MTU * 360.0
            fig.add_trace(go.Scatterpolar(
                r=t[:, idx] - r_offset, theta=theta, mode='lines',
                line=dict(color='rgba(0, 0, 180, 0.3)', width=1),
                showlegend=(i == 0), name='Truth', thetaunit='degrees',
            ))
        for i, m in enumerate(ml_trajs):
            time_arr = np.arange(len(m)) * eval_dt
            theta = (time_arr % T_MTU) / T_MTU * 360.0
            fig.add_trace(go.Scatterpolar(
                r=m[:, idx] - r_offset, theta=theta, mode='lines',
                line=dict(color='rgba(255, 50, 50, 0.25)', width=1),
                showlegend=(i == 0), name='ML', thetaunit='degrees',
            ))
        fig.update_layout(
            title=f"{dim_name} — Polar (1 rev = 1 MTU)",
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(
                    tickvals=[0, 90, 180, 270],
                    ticktext=['0', '0.25 MTU', '0.5 MTU', '0.75 MTU'],
                    direction='clockwise',
                ),
            ),
            margin=dict(l=20, r=20, t=40, b=20), height=300, showlegend=True,
        )
    else:
        # --- Standard line plot ---
        steps = len(truth_trajs[0])
        time_axis = np.arange(steps) * eval_dt
        for i, t in enumerate(truth_trajs):
            fig.add_trace(go.Scattergl(
                x=time_axis, y=t[:, idx], mode='lines',
                line=dict(color='rgba(0, 0, 180, 0.4)', width=1),
                showlegend=(i == 0), name='Truth',
            ))
        for i, m in enumerate(ml_trajs):
            t_ml = np.arange(len(m)) * eval_dt
            fig.add_trace(go.Scattergl(
                x=t_ml, y=m[:, idx], mode='lines',
                line=dict(color='rgba(255, 50, 50, 0.35)', width=1),
                showlegend=(i == 0), name='ML',
            ))
        fig.update_layout(
            title=f"{dim_name} (t)",
            xaxis_title='Time', margin=dict(l=20, r=20, t=30, b=20), height=300,
        )

    return fig


@app.callback(
    Output("eval-x-plot", "figure"),
    [Input("eval-traj-store", "data"),
     Input("eval-var-select-0", "value"),
     Input("eval-plot-type", "value")],
    prevent_initial_call=True
)
def update_aux_0(store_trigger, var_idx, plot_type):
    if store_trigger is None or eval_state.get('result') is None:
        return go.Figure()
    try:
        return _mk_aux_plot(var_idx, plot_type)
    except Exception as e:
        traceback.print_exc()
        err = go.Figure()
        err.update_layout(title=f"Error: {str(e)}", template="simple_white")
        return err


@app.callback(
    Output("eval-y-plot", "figure"),
    [Input("eval-traj-store", "data"),
     Input("eval-var-select-1", "value"),
     Input("eval-plot-type", "value")],
    prevent_initial_call=True
)
def update_aux_1(store_trigger, var_idx, plot_type):
    if store_trigger is None or eval_state.get('result') is None:
        return go.Figure()
    try:
        return _mk_aux_plot(var_idx, plot_type)
    except Exception as e:
        traceback.print_exc()
        err = go.Figure()
        err.update_layout(title=f"Error: {str(e)}", template="simple_white")
        return err


@app.callback(
    Output("eval-z-plot", "figure"),
    [Input("eval-traj-store", "data"),
     Input("eval-var-select-2", "value"),
     Input("eval-plot-type", "value")],
    prevent_initial_call=True
)
def update_aux_2(store_trigger, var_idx, plot_type):
    if store_trigger is None or eval_state.get('result') is None:
        return go.Figure()
    try:
        return _mk_aux_plot(var_idx, plot_type)
    except Exception as e:
        traceback.print_exc()
        err = go.Figure()
        err.update_layout(title=f"Error: {str(e)}", template="simple_white")
        return err


if __name__ == '__main__':
    LorenzSystems.warmup()  # pre-compile DAPyr JIT kernels in background
    app.run(debug=True, port=8050, host='0.0.0.0')
