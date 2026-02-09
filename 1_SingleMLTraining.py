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

from MachineLearning import DenseNN, ResDenseNN, LSTMNN, save_model, load_model
from datasets.LorenzDataset import LorenzDataset
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
    'model_name': 'dash_model'
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
                options=[{"label": "Lorenz 63", "value": "63"}, {"label": "Lorenz 96", "value": "96"}],
                value="63"
            ),
            html.Div([
                dbc.Label("L96 Dimension (N)"),
                dbc.Input(id="l96-n", type="number", value=40),
            ], id="l96-n-div", style={"display": "none"}),
            dbc.Label("History Steps (Input size)"),
            dbc.Input(id="prev-steps", type="number", value=3),
        ], title="Architecture"),
        
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([dbc.Label("dt"), dbc.Input(id="dt", type="number", value=0.01)]),
                dbc.Col([dbc.Label("Save_Dt (Skip)"), dbc.Input(id="save-dt", type="number", value=10)]),
            ]),
            dbc.Row([
                dbc.Col([dbc.Label("Random Locations"), dbc.Input(id="num-locs", type="number", value=10)]),
                dbc.Col([dbc.Label("Samples/Loc"), dbc.Input(id="ns-per-loc", type="number", value=5000)]),
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
                dbc.Col([dbc.Label("Batch Size"), dbc.Input(id="batch-size", type="number", value=1024)]),
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
], className="sidebar p-4 bg-light shadow", style={"width": "26rem", "position": "fixed", "top": 0, "left": 0, "bottom": 0, "overflowY": "auto"})

content = html.Div([
    dcc.Tabs(id="tabs", value='tab-train', children=[
        dcc.Tab(label='📈 Training', value='tab-train', className="custom-tab", selected_className="custom-tab--selected", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3("Real-time Learning Curves", className="mt-4"),
                        dcc.Graph(id="loss-graph", style={"height": "500px"}),
                        dcc.Interval(id="interval-update", interval=1000, n_intervals=0, disabled=True),
                    ], width=12),
                ]),
                dbc.Card([
                    dbc.CardHeader("Training Console"),
                    dbc.CardBody([
                        html.Pre(id="training-log", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "height": "100px", "overflowY": "auto"})
                    ])
                ], className="mt-4")
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
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Forecast Steps"),
                        dbc.Input(id="eval-steps", type="number", value=100),
                    ], width=2),
                    dbc.Col([
                        dbc.Label("Ensemble Size"),
                        dbc.Input(id="eval-ens-size", type="number", value=5),
                    ], width=2),
                    dbc.Col([
                        dbc.Button("⚡ Run Eval", id="eval-run-btn", color="success", className="mt-4 w-100"),
                    ], width=4),
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id="eval-3d-plot", style={"height": "55vh"}))], className="shadow-sm border-0"), width=12),
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="eval-x-plot", style={"height": "25vh"}), width=4),
                    dbc.Col(dcc.Graph(id="eval-y-plot", style={"height": "25vh"}), width=4),
                    dbc.Col(dcc.Graph(id="eval-z-plot", style={"height": "25vh"}), width=4),
                ], className="mt-3")
            ], fluid=True)
        ]),
    ])
], style={"marginLeft": "28rem", "paddingRight": "2rem"})

app.layout = html.Div([sidebar, content])

# --- Callbacks ---

@app.callback(
    Output("l96-n-div", "style"),
    Input("system-type", "value")
)
def toggle_n_l96(system):
    return {"display": "block"} if system == "96" else {"display": "none"}

def training_thread_func(config):
    global training_state
    try:
        # 1. Dataset Generation
        sys_params = {}
        if config['system_type'] == '96':
            sys_params['N'] = config['n_l96']
            sys_params['F'] = 8.0
            
        dataset = LorenzDataset(
            system_type=config['system_type'],
            dt=config['dt'],
            Ns=config['samples_per_loc'],
            save_Dt=config['save_dt'],
            num_start_locations=config['num_locs'],
            prev_time_steps=config['prev_steps'],
            **sys_params
        )
        
        # Splits
        total = len(dataset)
        train_sz = int(total * config['split_train'] / 100)
        val_sz = int(total * config['split_val'] / 100)
        test_sz = total - train_sz - val_sz
        train_set, val_set, test_set = random_split(dataset, [train_sz, val_sz, test_sz])
        
        # Optimization: Use pin_memory and more workers
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=(device.type=='cuda'), num_workers=2)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], pin_memory=(device.type=='cuda'), num_workers=2)
        
        # Model
        input_size = 3 if config['system_type'] == '63' else config['n_l96']
        hidden_list = [int(x.strip()) for x in config['hidden_layers'].split(',')]
        
        arch_meta = {
            'model_type': config['model_type'],
            'input_size': input_size,
            'prev_time_steps': config['prev_steps'],
            'hidden_layers': hidden_list,
            'system': config['system_type'],
            'N': sys_params.get('N', 40)
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
        
        train_model(model, model_name, train_loader, val_loader, criterion, optimizer, 500, early_stopping, progress_callback=progress, device=device)
        
        save_model(model, f"models/{model_name}", dataset.scaler.mean_, dataset.scaler.scale_, arch_meta)
        
        # Save config to YAML
        config_path = f"models/{model_name}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
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
    [State("model-type", "value"), State("system-type", "value"), State("l96-n", "value"),
     State("dt", "value"), State("save-dt", "value"), State("prev-steps", "value"),
     State("num-locs", "value"), State("ns-per-loc", "value"), State("batch-size", "value"),
     State("patience", "value"), State("hidden-layers", "value"), State("loss-func", "value"),
     State("split-train", "value"), State("split-val", "value"), State("split-test", "value")],
     prevent_initial_call=True
)
def start_tra(n, m, s, l_n, dt, s_dt, prev, locs, ns, bs, pat, hidden, loss, st, sv, ste):
    global training_state
    training_state = {
        'is_training': True, 'stop_requested': False,
        'history': {'epochs': [], 'train_loss': [], 'val_loss': []},
        'current_epoch': 0, 'rollout_steps': 1, 'model_name': ''
    }
    
    config = {
        'model_type': m, 'system_type': s, 'n_l96': l_n, 'dt': dt, 'save_dt': s_dt,
        'prev_steps': prev, 'num_locs': locs, 'samples_per_loc': ns,
        'batch_size': bs, 'patience': pat, 'hidden_layers': hidden, 'loss_func': loss,
        'split_train': st, 'split_val': sv, 'split_test': ste
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
     Output("interval-update", "disabled", allow_duplicate=True)],
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
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    done = not training_state['is_training'] and training_state['current_epoch'] > 0
    
    if done:
        status = f"DONE. Saved: {training_state['model_name']}.pth"
        return fig, status, False, True, True  # re-enable start, disable stop, disable interval
    
    status = f"Running Epoch: {training_state['current_epoch']} | Rollout Stage: {training_state['rollout_steps']}"
    return fig, status, True, False, False  # keep start disabled, stop enabled, interval active


@app.callback(
    Output("eval-model-select", "options"),
    [Input("refresh-models-btn", "n_clicks"), Input("tabs", "value")]
)
def refresh_models_list(n, tab):
    files = [f for f in os.listdir("models") if f.endswith(".pth")]
    return [{"label": f, "value": f} for f in sorted(files, reverse=True)]

@app.callback(
    [Output("eval-3d-plot", "figure"),
     Output("eval-x-plot", "figure"),
     Output("eval-y-plot", "figure"),
     Output("eval-z-plot", "figure")],
    [Input("eval-run-btn", "n_clicks")],
    [State("eval-model-select", "value"), State("eval-steps", "value"), State("eval-ens-size", "value")],
    prevent_initial_call=True
)
def run_eval(n, model_file, steps, ens_size):
    if not model_file: return [go.Figure()]*4
    
    path = os.path.join("models", model_file)
    cp = torch.load(path, map_location='cpu', weights_only=False)
    meta = cp['architecture']
    
    nx = meta['input_size']
    prev_steps = meta['prev_time_steps']
    
    # Init Model
    m_class = DenseNN if meta['model_type'] == 'Dense' else (ResDenseNN if meta['model_type'] == 'ResDense' else LSTMNN)
    if meta['model_type'] == 'LSTM':
        model = m_class(nx, prev_steps, nx, meta['hidden_layers'][0])
    else:
        model = m_class(nx, prev_steps, nx, meta['hidden_layers'], nn.ReLU, None)
    
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    mean, std = cp['train_mean'], cp['train_std']
    
    # Build Evaluation
    # 1. True Trajectory (Lorenz)
    dt = 0.01 # Assumption for eval
    x0 = np.random.normal(0, 1, nx)
    sys = meta['system']
    true_traj = LorenzSystems.generate_trajectory(sys, x0, dt, steps, N=meta.get('N', 40))
    
    # ML Forecast (Batchized on GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate initial history for all ensemble members
    hist_list = []
    for i in range(ens_size):
        curr_x0 = x0 + np.random.normal(0, 0.01, nx) if i > 0 else x0
        hist = LorenzSystems.generate_trajectory(sys, curr_x0, dt, prev_steps, N=meta.get('N', 40))
        hist_list.append(hist)
    
    all_hists = np.stack(hist_list) # (ens, prev_steps, nx)
    hists_norm = (all_hists - mean) / std
    input_seqs = torch.tensor(hists_norm.reshape(ens_size, -1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds_norm = recursive_rollout(model, input_seqs, steps - prev_steps, prev_steps, device)
        preds = preds_norm.cpu().numpy() * std + mean # (ens, steps-prev, nx)
    
    ml_trajs = [np.vstack([all_hists[i], preds[i]]) for i in range(ens_size)]

    # Plots
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=true_traj[:,0], y=true_traj[:,1], z=true_traj[:,2 if nx>=3 else 0], 
                                 mode='lines', line=dict(color='blue', width=4), name='Truth'))
    for i, m in enumerate(ml_trajs):
        fig3d.add_trace(go.Scatter3d(x=m[:,0], y=m[:,1], z=m[:,2 if nx>=3 else 0], 
                                     mode='lines', line=dict(color='red', width=1), opacity=0.4, 
                                     showlegend=(i==0), name='ML Prediction'))
    fig3d.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), margin=dict(l=0,r=0,b=0,t=0))
    
    def mk1d(idx, title):
        f = go.Figure()
        f.add_trace(go.Scatter(y=true_traj[:, idx], name='Truth', line=dict(color='blue')))
        for m in ml_trajs:
            f.add_trace(go.Scatter(y=m[:, idx], line=dict(color='red', width=1), opacity=0.3, showlegend=False))
        f.update_layout(title=title, margin=dict(l=20,r=20,t=30,b=20), height=200)
        return f

    return fig3d, mk1d(0, "X (t)"), mk1d(1, "Y (t)"), mk1d(2, "Z (t)")

if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')
