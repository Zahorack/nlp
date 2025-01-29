from typing import Callable, Iterator


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        delta: float,
        epochs: int,
        callbacks: list[Callable[[bool, float, int], None]] | None = None,
    ):
        """
        :param patience: number of epochs without improvement
        :param delta: epoch improvement tolerance
        :param epochs: number of training epochs
        :param callbacks: callback functions
        """
        self.patience = patience
        self.delta = delta
        self.epochs = epochs
        self.current_epoch = 0
        self.epochs_without_change = 0
        self.last_loss: float | None = None
        self.callbacks = callbacks or []

    def _callback(self, improvement: bool, loss: float):
        for cb in self.callbacks:
            cb(improvement, loss, self.epochs_without_change)

    @property
    def needs_training(self) -> bool:
        """
        :return: True if training should continue, False if should stop early
        """
        return self.epochs_without_change <= self.patience and self.current_epoch < self.epochs

    def update(self, loss: float) -> bool:
        """
        :param loss: validation loss
        :return: whether it needs training
        """
        improvement = False
        if self.last_loss is None or (1 - (loss / self.last_loss) > self.delta):
            self.last_loss = loss
            self.epochs_without_change = 0
            improvement = True
        else:
            self.epochs_without_change += 1

        self._callback(improvement, loss)

        return self.needs_training

    def __iter__(self) -> Iterator[int]:
        while self.needs_training:
            yield self.current_epoch
            self.current_epoch += 1
