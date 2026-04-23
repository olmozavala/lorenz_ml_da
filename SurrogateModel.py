import os
import yaml
import numpy as np
import torch
import torch.nn as nn

from MachineLearning import DenseNN, ResDenseNN, LSTMNN, RNN
torch.set_default_dtype(torch.float64)

_ACTIVATION_MAP = {
    'ReLU':    nn.ReLU,
    'Tanh':    nn.Tanh,
    'Sigmoid': nn.Sigmoid,
}


class SurrogateModel:
    """
    Generic loader and predictor for any pretrained Lorenz surrogate model.

    Supports all four model types (DenseNN, ResDenseNN, LSTMNN, RNN)
    saved by Main_ML.py.  Handles normalisation internally so the public API
    operates entirely in physical space.

    Parameters
    ----------
    pth_path : str
        Path to a .pth checkpoint (full checkpoint from save_model(), or a bare
        state_dict from a _best_model.pth file).
    device : str or torch.device, optional
        Target device.  Defaults to CUDA if available, else CPU.
    yml_path : str, optional
        Explicit path to the .yml sidecar.  If None the class tries to
        auto-discover it next to the .pth file.

    Examples
    --------
    >>> surrogate = SurrogateModel('models/RNN_L63_trial1_1774989331.pth')
    >>> history = np.random.randn(4, 3)          # (prev_time_steps, N)
    >>> next_state = surrogate.predict(history)   # (N,)
    >>> traj = surrogate.rollout(history, 100)    # (100, N)
    """

    def __init__(self, pth_path, device=None, yml_path=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        checkpoint, yml_meta = self._load_files(pth_path, yml_path)

        is_full_checkpoint = 'model_state_dict' in checkpoint
        state_dict = checkpoint['model_state_dict'] if is_full_checkpoint else checkpoint

        # Merge metadata: checkpoint architecture + YAML sidecar
        arch = checkpoint.get('architecture', {}) if is_full_checkpoint else {}
        meta = {**yml_meta, **arch}  # checkpoint architecture takes priority

        if 'model_type' not in meta:
            raise ValueError(
                f"Cannot determine model_type from checkpoint or YAML sidecar. "
                f"Provide a yml_path or use a full checkpoint."
            )

        self.model_type = meta['model_type']
        self.input_size = meta['input_size']
        self.prev_time_steps = meta['prev_time_steps']
        self.metadata = meta

        # Resolve normalisation stats: checkpoint > YAML
        self.train_mean = self._resolve_array(
            checkpoint.get('train_mean') if is_full_checkpoint else None,
            yml_meta.get('train_mean'),
        )
        self.train_std = self._resolve_array(
            checkpoint.get('train_std') if is_full_checkpoint else None,
            yml_meta.get('train_std'),
        )

        # Build and load model
        self.model = self._build_model(meta, yml_meta)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._hidden = None  # recurrent hidden state (LSTMNN/RNN only)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, state_history):
        """
        Single-step prediction in physical space.

        Parameters
        ----------
        state_history : np.ndarray
            Shape ``(prev_time_steps, N)`` or ``(prev_time_steps * N,)``,
            ordered oldest-first: ``[s_{t-k+1}, ..., s_t]``.

        Returns
        -------
        np.ndarray of shape ``(N,)`` — the predicted next state.
        """
        state_history = np.asarray(state_history, dtype=np.float64)
        if state_history.ndim == 1:
            state_history = state_history.reshape(self.prev_time_steps, self.input_size)

        normalised = (state_history - self.train_mean) / self.train_std
        flat = normalised.reshape(1, -1)
        dtype = next(self.model.parameters()).dtype
        x = torch.as_tensor(flat, dtype=dtype, device=self.device)

        with torch.no_grad():
            if isinstance(self.model, (LSTMNN, RNN)):
                pred, self._hidden = self.model(x, self._hidden)
            else:
                pred = self.model(x)
            pred = pred.cpu().numpy().astype(np.float64).squeeze(0)

        return pred * self.train_std + self.train_mean

    def rollout(self, state_history, num_steps):
        """
        Multi-step autoregressive prediction in physical space.

        Each step's output is denormalised, appended to the window, and
        renormalised before the next forward pass.

        Parameters
        ----------
        state_history : np.ndarray
            Shape ``(prev_time_steps, N)``, ordered oldest-first.
        num_steps : int
            Number of future steps to predict.

        Returns
        -------
        np.ndarray of shape ``(num_steps, N)`` — predicted trajectory.
        """
        state_history = np.asarray(state_history, dtype=np.float64)
        if state_history.ndim == 1:
            state_history = state_history.reshape(self.prev_time_steps, self.input_size)

        self.reset_hidden()  # clear hidden state for fresh rollout
        window = state_history.copy()  # (prev_time_steps, N) in physical space
        predictions = []

        for _ in range(num_steps):
            pred = self.predict(window)       # (N,) physical space
            predictions.append(pred)
            window = np.concatenate([window[1:], pred[np.newaxis, :]], axis=0)

        return np.stack(predictions, axis=0)

    def __call__(self, state_history):
        """Alias for :meth:`predict`."""
        return self.predict(state_history)

    def reset_hidden(self, batch_size=None):
        """Reset hidden state for recurrent models. No-op for feedforward."""
        self._hidden = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_files(pth_path, yml_path):
        """Load checkpoint and YAML sidecar, returning (checkpoint_dict, yml_dict)."""
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

        # Auto-discover YAML sidecar
        if yml_path is None:
            base = pth_path.replace('.pth', '')
            # Strip _best_model suffix if present
            if base.endswith('_best_model'):
                base = base[:-len('_best_model')]
            candidate = base + '.yml'
            if os.path.isfile(candidate):
                yml_path = candidate

        yml_meta = {}
        if yml_path is not None:
            with open(yml_path, 'r') as f:
                raw = yaml.safe_load(f) or {}
            # Flatten: pull architecture sub-dict fields into top level
            arch_from_yml = raw.get('architecture', {})
            yml_meta = {**raw, **arch_from_yml}

        return checkpoint, yml_meta

    def _build_model(self, meta, yml_meta):
        """Instantiate the correct nn.Module from merged metadata."""
        model_type = meta['model_type']
        input_size = meta['input_size']
        prev_time_steps = meta['prev_time_steps']
        output_size = input_size
        hidden_layers = meta['hidden_layers']

        if model_type in ('DenseNN', 'ResDenseNN'):
            act_name = meta.get('hidden_activation',
                                yml_meta.get('hidden_activation', 'ReLU'))
            hidden_act = _ACTIVATION_MAP.get(act_name, nn.ReLU)
            cls = DenseNN if model_type == 'DenseNN' else ResDenseNN
            return cls(input_size, prev_time_steps, output_size,
                       hidden_layers, hidden_act, None)

        elif model_type == 'LSTMNN':
            hidden_size = hidden_layers[0]
            num_layers = int(meta.get('num_layers', yml_meta.get('num_layers', 1)))
            return LSTMNN(input_size, prev_time_steps, output_size,
                          hidden_size, num_layers)

        elif model_type == 'RNN':
            hidden_size = hidden_layers[0]
            num_layers = int(meta.get('num_layers', yml_meta.get('num_layers', 1)))
            nonlinearity = meta.get('rnn_nonlinearity',
                                    yml_meta.get('rnn_nonlinearity', 'tanh'))
            return RNN(input_size, prev_time_steps, output_size,
                       hidden_size, num_layers, nonlinearity)

        else:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                "Expected 'DenseNN', 'ResDenseNN', 'LSTMNN', or 'RNN'."
            )

    @staticmethod
    def _resolve_array(primary, fallback):
        """Return whichever source is available as a numpy array."""
        src = primary if primary is not None else fallback
        if src is None:
            raise ValueError(
                "Normalisation statistics (train_mean / train_std) not found in "
                "checkpoint or YAML sidecar."
            )
        return np.asarray(src, dtype=np.float64)
