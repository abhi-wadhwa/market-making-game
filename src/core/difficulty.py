"""Difficulty levels for the market-making game.

Each difficulty level controls:
- Adverse selection intensity (alpha)
- Asset volatility (sigma)
- Jump frequency and size
- Order flow toxicity
- Number of rounds
- Position limits
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class DifficultyLevel(Enum):
    """Available difficulty levels."""

    TUTORIAL = "tutorial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class DifficultyConfig:
    """Configuration parameters for a difficulty level.

    Attributes
    ----------
    level : DifficultyLevel
        The difficulty level.
    name : str
        Human-readable name.
    description : str
        Description of what makes this level challenging.
    n_rounds : int
        Number of trading rounds.
    initial_value : float
        Starting true value of the asset.
    sigma : float
        Per-round volatility.
    jump_intensity : float
        Poisson intensity for jumps.
    jump_size : float
        Average jump magnitude.
    alpha : float
        Probability of informed trader (adverse selection).
    arrival_rate : float
        Average orders per round.
    max_position : int
        Maximum allowed absolute position.
    inventory_penalty : float
        Quadratic penalty coefficient for inventory.
    initial_cash : float
        Starting cash.
    """

    level: DifficultyLevel
    name: str
    description: str
    n_rounds: int
    initial_value: float
    sigma: float
    jump_intensity: float
    jump_size: float
    alpha: float
    arrival_rate: float
    max_position: int
    inventory_penalty: float
    initial_cash: float


# Pre-defined difficulty presets
DIFFICULTY_PRESETS: Dict[DifficultyLevel, DifficultyConfig] = {
    DifficultyLevel.TUTORIAL: DifficultyConfig(
        level=DifficultyLevel.TUTORIAL,
        name="Tutorial",
        description=(
            "Learn the basics. Low volatility, very few informed traders, "
            "and generous position limits. The true value barely moves."
        ),
        n_rounds=20,
        initial_value=100.0,
        sigma=0.1,
        jump_intensity=0.0,
        jump_size=0.0,
        alpha=0.1,
        arrival_rate=1.0,
        max_position=30,
        inventory_penalty=0.005,
        initial_cash=10000.0,
    ),
    DifficultyLevel.EASY: DifficultyConfig(
        level=DifficultyLevel.EASY,
        name="Easy",
        description=(
            "Mild adverse selection and moderate volatility. "
            "Good for practicing spread management."
        ),
        n_rounds=30,
        initial_value=100.0,
        sigma=0.3,
        jump_intensity=0.02,
        jump_size=2.0,
        alpha=0.2,
        arrival_rate=1.0,
        max_position=25,
        inventory_penalty=0.008,
        initial_cash=10000.0,
    ),
    DifficultyLevel.MEDIUM: DifficultyConfig(
        level=DifficultyLevel.MEDIUM,
        name="Medium",
        description=(
            "Significant informed trading and meaningful jumps. "
            "You need to manage inventory actively."
        ),
        n_rounds=40,
        initial_value=100.0,
        sigma=0.5,
        jump_intensity=0.05,
        jump_size=3.0,
        alpha=0.3,
        arrival_rate=1.0,
        max_position=20,
        inventory_penalty=0.01,
        initial_cash=10000.0,
    ),
    DifficultyLevel.HARD: DifficultyConfig(
        level=DifficultyLevel.HARD,
        name="Hard",
        description=(
            "High adverse selection, volatile price with frequent jumps. "
            "Tight position limits demand precise quoting."
        ),
        n_rounds=50,
        initial_value=100.0,
        sigma=0.8,
        jump_intensity=0.08,
        jump_size=4.0,
        alpha=0.4,
        arrival_rate=1.2,
        max_position=15,
        inventory_penalty=0.015,
        initial_cash=10000.0,
    ),
    DifficultyLevel.EXPERT: DifficultyConfig(
        level=DifficultyLevel.EXPERT,
        name="Expert",
        description=(
            "Extremely toxic order flow, high volatility, large jumps, "
            "and strict position limits. Only the best survive."
        ),
        n_rounds=60,
        initial_value=100.0,
        sigma=1.2,
        jump_intensity=0.12,
        jump_size=5.0,
        alpha=0.5,
        arrival_rate=1.5,
        max_position=10,
        inventory_penalty=0.02,
        initial_cash=10000.0,
    ),
}


def get_difficulty_config(level: DifficultyLevel) -> DifficultyConfig:
    """Get the configuration for a difficulty level.

    Parameters
    ----------
    level : DifficultyLevel
        The desired difficulty.

    Returns
    -------
    DifficultyConfig
        Configuration for that level.
    """
    return DIFFICULTY_PRESETS[level]


def get_difficulty_by_name(name: str) -> DifficultyConfig:
    """Get difficulty config by name string.

    Parameters
    ----------
    name : str
        Case-insensitive difficulty name.

    Returns
    -------
    DifficultyConfig
        Configuration for that level.

    Raises
    ------
    ValueError
        If name does not match any level.
    """
    name_lower = name.lower().strip()
    for level in DifficultyLevel:
        if level.value == name_lower:
            return DIFFICULTY_PRESETS[level]
    valid = [l.value for l in DifficultyLevel]
    raise ValueError(f"Unknown difficulty '{name}'. Valid: {valid}")
