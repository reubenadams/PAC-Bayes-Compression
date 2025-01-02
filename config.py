from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    model_path: str
    model_dims: tuple[int, int, int]
    model_act: str
    train_epochs: int
    batch_size: int
    learning_rate: float
