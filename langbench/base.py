from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, data: dict) -> float:
        """Calculate the metric score from provided data."""
        pass

    @abstractmethod
    def details(self) -> str:
        """Return a detailed explanation of what this metric measures."""
        pass
