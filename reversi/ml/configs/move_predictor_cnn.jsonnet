{
    "reader": {"type": "reversi_move_prediction",
        "feature_extractor": {
            "type": "cnn",
            "flatten": false,
            "add_legal_positions": false
        }
    },
    "train_data_path": "/data/local/li0123/othello_data/ffo_wthor/WTH_2020.wtb",
    "validation_data_path": "/data/local/li0123/othello_data/ffo_wthor/WTH_2020.wtb",
    "model": {
        "type": "move_predictor",
        "board_encoder": {"type": "reversi_conv", "num_channels": [2, 64, 64, 128, 128]},
    },
    "trainer": {"type": "default", "num_max_epochs": 20, "patience": 1},
    "validation_metric": "+accuracy",
    "batch_size": 32,
    "optimizer": {"type": "adam", "lr": 0.001},
}
