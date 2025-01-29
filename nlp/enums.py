from enum import IntEnum, unique


@unique
class SpecialTokens(IntEnum):
    PAD = 0
    START = 1
    STOP = 2
    DELIMITER = 3
    MASK = 4
    OTHER = 5
