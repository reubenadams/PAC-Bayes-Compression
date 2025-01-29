from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model_path: str
    model_dims: list[int, int, int]
    model_act: str
    train_epochs: int
    batch_size: int
    learning_rate: float
    dataset: str
    new_size: tuple[int, int] = None


base_mnist_config = Config(
    model_path="trained_models/mnist_base_mlp.t",
    model_dims=[784, 128, 10],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001,
    dataset="MNIST",
)

hyper_mnist_config_scaled = Config(
    model_path="trained_models/mnist_hyper_mlp_scaled.t",
    model_dims=[3, 1024, 1],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.01,
    dataset="MNIST",
)

hyper_mnist_config_binary = Config(
    model_path="trained_models/mnist_hyper_mlp_binary.t",
    model_dims=[3, 64, 512, 64, 1],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.0001,
    dataset="MNIST",
)

low_rank_mnist_config = Config(
    model_path="trained_models/mnist_low_rank_mlp.t",
    model_dims=[784, 128, 10],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001,
    dataset="MNIST",
)

low_rank_CIFAR100_config = Config(
    model_path="trained_models/cifar100_low_rank_mlp.t",
    model_dims=[100, 100, 100],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001,
    dataset="CIFAR100",
)

full_mnist_config = Config(
    model_path="trained_models/mnist_2x2_full_mlp.t",
    model_dims=[4, 32, 10],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001,
    dataset="MNIST",
    new_size=(2, 2),
)

dist_data_mnist_config = Config(
    model_path="trained_models/mnist_2x2_dist_data_mlp.t",
    model_dims=[4, 32, 10],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001,
    dataset="MNIST",
    new_size=(2, 2),
)
