"""
Modul zur Modelldefinition: einfache LSTM- und CNN-LSTM-Modelle
Geeignet für den Einsatz in der Zeitreihenvorhersage
"""

import torch
import torch.nn as nn
import numpy as np
from config import config
from utils import Utils


class SimpleLSTM(nn.Module):
    """Einfaches LSTM-Modell für Zeitreihenvorhersagen"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Vorwärtsdurchlauf: (batch, seq, features) -> (batch, 1)"""
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)


class SimpleCNNLSTM(nn.Module):
    """CNN-LSTM: zuerst Merkmalsextraktion per CNN, danach zeitliche Modellierung per LSTM"""

    def __init__(self, input_size, cnn_filters=32, kernel_size=3,
                 hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleCNNLSTM, self).__init__()

        self.input_size = input_size
        self.cnn_filters = cnn_filters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(cnn_filters)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Vorwärtsdurchlauf: (batch, seq, features) -> (batch, 1)"""
        # CNN: (batch, seq, features) -> (batch, features, seq)
        x_cnn = x.transpose(1, 2)
        conv_out = self.relu(self.batch_norm(self.conv1d(x_cnn)))

        # LSTM: (batch, features, seq) -> (batch, seq, features)
        lstm_input = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_input)

        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output)


class ModelFactory:
    """Modellfabrik zur vereinfachten Erstellung"""

    @staticmethod
    def create_lstm(input_size, **kwargs):
        Utils.set_seed()
        return SimpleLSTM(input_size, **kwargs)

    @staticmethod
    def create_cnn_lstm(input_size, **kwargs):
        Utils.set_seed()
        return SimpleCNNLSTM(input_size, **kwargs)

    @staticmethod
    def get_model_info(model):
        """Modellinformationen ermitteln"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'model_type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


def test_models():
    """Funktionstest der Modelle"""
    print("Modelle werden getestet...")

    batch_size, seq_len, input_size = 10, 5, 7
    test_input = torch.randn(batch_size, seq_len, input_size)
    print(f"Form der Testdaten: {test_input.shape}")

    print("\nTest des LSTM-Modells:")
    lstm_model = ModelFactory.create_lstm(input_size)
    lstm_output = lstm_model(test_input)
    print(f"Ausgabeform LSTM: {lstm_output.shape}")
    print(f"LSTM-Modellinfo: {ModelFactory.get_model_info(lstm_model)}")

    print("\nTest des CNN-LSTM-Modells:")
    cnn_lstm_model = ModelFactory.create_cnn_lstm(input_size)
    cnn_lstm_output = cnn_lstm_model(test_input)
    print(f"Ausgabeform CNN-LSTM: {cnn_lstm_output.shape}")
    print(f"CNN-LSTM-Modellinfo: {ModelFactory.get_model_info(cnn_lstm_model)}")

    assert lstm_output.shape == (batch_size, 1), "Form der LSTM-Ausgabe ist falsch"
    assert cnn_lstm_output.shape == (batch_size, 1), "Form der CNN-LSTM-Ausgabe ist falsch"

    print("\nAlle Modultests erfolgreich.")
    return lstm_model, cnn_lstm_model


def create_models_for_energy_prediction():
    """Modelle für die Energieprognose erzeugen"""
    print("Modelle für die Energieprognose werden erstellt...")

    input_size = len(config.FEATURE_COLS)

    lstm_model = ModelFactory.create_lstm(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT_RATE
    )

    cnn_lstm_model = ModelFactory.create_cnn_lstm(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT_RATE
    )

    print(f"Anzahl Eingangsmerkmale: {input_size}")
    print(f"LSTM-Modellparameter: {ModelFactory.get_model_info(lstm_model)}")
    print(f"CNN-LSTM-Modellparameter: {ModelFactory.get_model_info(cnn_lstm_model)}")

    return lstm_model, cnn_lstm_model


if __name__ == "__main__":
    test_models()
    print("\n" + "=" * 50)

    try:
        energy_lstm, energy_cnn_lstm = create_models_for_energy_prediction()
        print("Modelle für die Energieprognose wurden erfolgreich erstellt.")
    except Exception as e:
        print(f"Fehler beim Erstellen der Energieprognosemodelle: {e}")
