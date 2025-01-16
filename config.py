from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    model_path: str
    model_dims: list[int, int, int]
    model_act: str
    train_epochs: int
    batch_size: int
    learning_rate: float


base_config = Config(
    model_path="trained_models/base_mlp.t",
    model_dims=[784, 128, 10],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001
)

hyper_config_scaled = Config(
    model_path="trained_models/hyper_mlp_scaled.t",
    model_dims=[3, 1024, 1],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.01
)

hyper_config_binary = Config(
    model_path="trained_models/hyper_mlp_binary.t",
    model_dims=[3, 64, 512, 64, 1],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.0001
)

low_rank_config = Config(
    model_path="trained_models/low_rank_mlp.t",
    model_dims=[784, 128, 10],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001
)
