import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models.base.PipelineWrapper import RTRegressor


def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class _DnnModel(nn.Module):
    def __init__(self, n_features, number_of_neurons_per_layer=512, number_of_hidden_layers=2, dropout_between_layers=0, activation='gelu'):
        super().__init__()
        # TODO: NO LLEGAN LOS PARÁMETROS!!!

        self.hidden_layers = []
        # First hidden layer
        self.hidden_layers.append(nn.Linear(n_features, number_of_neurons_per_layer))
        nn.init.zeros_(self.hidden_layers[0].bias)

        # Intermediate hidden layers
        for hidden_layer in range(1, number_of_hidden_layers):
            print(hidden_layer)
            self.hidden_layers.append(nn.Linear(number_of_neurons_per_layer, number_of_neurons_per_layer))
            nn.init.zeros_(self.hidden_layers[hidden_layer].bias)

        # Output layer
        self.l_out = nn.Linear(number_of_neurons_per_layer, 1)
        nn.init.zeros_(self.l_out.bias)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = F.silu
        self.dropout = nn.Dropout(dropout_between_layers)

    # TODO: NO SE MUY BIEN QUE HACE O HACÍA ESTO, PERO DE TODOS MODOS YA NO LO HACE
    def forward(self, x):
        for hidden_layer in range(0, len(self.hidden_layers)):
            x = self.dropout(self.activation(self.hidden_layers[hidden_layer](x)))
        return self.l_out(x)


class _SkDnn(BaseEstimator, RegressorMixin):
    def __init__(self, number_of_hidden_layers=2, dropout_between_layers=0, activation='gelu', number_of_neurons_per_layer=512,
                 lr=3e-4, max_number_of_epochs=30, annealing_rounds=2, swa_epochs=20, batch_size=64, device=get_default_device()):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        self.n_epochs = self.annealing_rounds * self.max_number_of_epochs

    def _init_hidden_model(self, n_features):
        self._model = _DnnModel(n_features).to(self.device)
        min_lr = 0.1 * self.lr
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer, T_0=self.max_number_of_epochs, T_mult=1, eta_min=min_lr
        )
        self._swa_model = AveragedModel(self._model)
        self._swa_scheduler = SWALR(self._optimizer, swa_lr=min_lr)

    def fit(self, X, y):
        self._init_hidden_model(X.shape[1])
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).view(-1, 1))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        train_iters = len(data_loader)
        for epoch in range(self.n_epochs):
            for i, (xb, yb) in enumerate(data_loader):
                self._batch_step(xb, yb)
                self._scheduler.step(epoch + i / train_iters)

        self._swa_model.train()
        for epoch in range(self.swa_epochs):
            for xb, yb in data_loader:
                self._batch_step(xb, yb)
            self._swa_model.update_parameters(self._model)
            self._swa_scheduler.step()

        return self

    def _batch_step(self, xb, yb):
        self._optimizer.zero_grad()
        pred = self._model(xb.to(self.device))
        loss = F.l1_loss(pred, target=yb.to(self.device))
        loss.backward()
        self._optimizer.step()

    def predict(self, X):
        self._model.eval()
        self._swa_model.eval()
        with torch.no_grad():
            return self._swa_model(torch.from_numpy(X).to(self.device)).cpu().numpy().flatten()

    def __getstate__(self):
        state = super().__getstate__().copy()
        if '_model' in state.keys():
            for key in ['_model', '_optimizer', '_scheduler', '_swa_model', '_swa_scheduler']:
                state[key] = state[key].state_dict()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        if '_model' in state.keys():
            self._init_hidden_model(state['_model']['l1.weight'].shape[1])
            for key in ['_model', '_optimizer', '_scheduler', '_swa_model', '_swa_scheduler']:
                torch_model = getattr(self, key)
                torch_model.load_state_dict(state.pop(key))
                setattr(self, key, torch_model)


class SkDnn(RTRegressor):
    def __init__(self, number_of_hidden_layers=2, dropout_between_layers=0, activation='gelu', number_of_neurons_per_layer=512,
                 lr=3e-4, max_number_of_epochs=30, annealing_rounds=2, swa_epochs=20, batch_size=32,
                 device=get_default_device(),
                 use_col_indices='all',
                 binary_col_indices=None, var_p=0, transform_output=True):
        super().__init__(use_col_indices, binary_col_indices, var_p, transform_output)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return _SkDnn(**self._rt_regressor_params())
