import logging
import random
from typing import Iterable

import torch
from torch.utils.data import IterableDataset

from nlp.enums import SpecialTokens


class NextTokenPredictionIterableDataset(IterableDataset):
    def __init__(
        self,
        X: list[list[int]],
        window_size: int,
        vocabulary_size: int,
        buffer_size: int = 10000,
        shuffle: bool = False,
        masking_rate: float = 0.0,
    ):
        self.data = X
        self.window_size = window_size
        self.vocabulary_size = vocabulary_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.masking_rate = masking_rate

    def _random_masking(self, token_sequence: list[int]) -> list[int]:
        if not 0 <= self.masking_rate <= 1:
            raise ValueError("Masking rate must be between 0 and 1")

        if self.masking_rate == 0.0:
            return token_sequence

        total_items = len(token_sequence)
        num_items_to_mask = round(total_items * self.masking_rate)

        # Select random indices to mask
        indices_to_mask = random.sample(range(total_items), num_items_to_mask)

        # Replace the items at the selected indices with MASK token
        for index in indices_to_mask:
            token_sequence[index] = SpecialTokens.MASK.value

        return token_sequence

    def _to_input_tensor(self, inputs: list[int]) -> torch.Tensor:
        """
        Prepend START token and pad to constant window size
        """
        inputs = self._random_masking(inputs)
        inputs = torch.Tensor([SpecialTokens.START.value] + inputs)

        if len(inputs) < self.window_size + 1:
            pad_size = self.window_size + 1 - len(inputs)
            inputs = torch.nn.functional.pad(
                inputs, (0, pad_size), mode="constant", value=SpecialTokens.PAD.value
            )

        return inputs.to(torch.int64)

    def _to_label_tensor(self, labels: list[int]) -> torch.Tensor:
        """
        Encode to one-hot tensor
        """
        return (
            torch.nn.functional.one_hot(
                torch.Tensor(labels).to(torch.int64), num_classes=self.vocabulary_size
            )
            .squeeze()
            .to(torch.float)
        )

    def _shuffle_and_yield(
        self, buffer: list[tuple[list[int], list[int], int]]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Shuffle buffer randomly and yield tensors
        """
        if self.shuffle:
            random.shuffle(buffer)

        while buffer:
            inputs, labels, length = buffer.pop(0)
            yield self._to_input_tensor(inputs), self._to_label_tensor(labels), length

    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Generate sliding windows for next token prediction task
        """
        buffer: list[tuple[list[int], list[int], int]] = []

        for sequence in self.data:
            if not sequence:
                logging.warning("Empty sequence can't be used for next token prediction!")

            # First context window iterative expanding
            for end in range(0, min(len(sequence), self.window_size)):
                inputs = sequence[:end]
                labels = sequence[end : end + 1]  # +1 prediction length
                buffer.append((inputs, labels, len(inputs) + 1))  # +1 START token

            # Sliding window phase
            for start in range(len(sequence) - self.window_size):
                inputs = sequence[start + 1 : start + self.window_size]
                labels = sequence[
                    start + self.window_size : start + self.window_size + 1
                ]  # +1 prediction length
                buffer.append((inputs, labels, len(inputs) + 1))  # +1 START token

            if len(buffer) >= self.buffer_size:
                yield from self._shuffle_and_yield(buffer)

        yield from self._shuffle_and_yield(buffer)
