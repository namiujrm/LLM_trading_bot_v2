"""Domain Value Objects for financial pricing and sizing."""

from __future__ import annotations

from pydantic import BaseModel


class Price(BaseModel):
    value: float

    def validate_positive(self) -> None:
        if self.value <= 0.0:
            raise ValueError(f"Price must be strictly positive, got {self.value}")


class Quantity(BaseModel):
    value: float

    def validate_positive(self) -> None:
        if self.value <= 0.0:
            raise ValueError(f"Quantity must be strictly positive, got {self.value}")


class Percentage(BaseModel):
    value: float

    def validate_ratio(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"Percentage ratio must be in [0.0, 1.0], got {self.value}")
