"""
Cascadia Scorer - Core Data Models

This module defines the data classes used throughout the pipeline.
Uses dataclasses for clean, type-hinted data structures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import json


class AnimalType(Enum):
    """Wildlife token types in Cascadia"""
    BEAR = "bear"
    ELK = "elk"
    SALMON = "salmon"
    HAWK = "hawk"
    FOX = "fox"
    NONE = "none"  # Empty tile (no token placed)


class HabitatType(Enum):
    """Habitat types in Cascadia"""
    FOREST = "forest"
    RIVER = "river"
    MOUNTAIN = "mountain"
    PRAIRIE = "prairie"
    WETLAND = "wetland"
    UNKNOWN = "unknown"


@dataclass
class AxialCoord:
    """
    Axial coordinate system for hexagonal grids.
    Uses (q, r) coordinates which simplify neighbor calculations.
    
    In axial coordinates, the 6 neighbors of a hex are:
    - (q+1, r), (q+1, r-1), (q, r-1)
    - (q-1, r), (q-1, r+1), (q, r+1)
    """
    q: int
    r: int
    
    def __hash__(self):
        return hash((self.q, self.r))
    
    def __eq__(self, other):
        if isinstance(other, AxialCoord):
            return self.q == other.q and self.r == other.r
        return False
    
    def get_neighbors(self) -> List['AxialCoord']:
        """Returns the 6 neighboring hex coordinates"""
        directions = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        return [AxialCoord(self.q + dq, self.r + dr) for dq, dr in directions]
    
    def distance_to(self, other: 'AxialCoord') -> int:
        """Calculate hex distance to another coordinate"""
        return (abs(self.q - other.q) 
                + abs(self.q + self.r - other.q - other.r) 
                + abs(self.r - other.r)) // 2
    
    def to_dict(self) -> Dict:
        return {"q": self.q, "r": self.r}


@dataclass
class PixelCoord:
    """Pixel coordinates in the image"""
    x: float
    y: float
    
    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y}


@dataclass
class Tile:
    """
    Represents a single hex tile on a player's board.
    """
    id: str
    animal: AnimalType
    habitat: HabitatType
    coord: AxialCoord
    pixel_center: Optional[PixelCoord] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        habitat_val = self.habitat.value if isinstance(self.habitat, HabitatType) else str(self.habitat)
        return {
            "id": self.id,
            "animal": self.animal.value,
            "habitat": habitat_val,
            "coord": self.coord.to_dict(),
            "pixel_center": self.pixel_center.to_dict() if self.pixel_center else None,
            "confidence": self.confidence
        }


@dataclass
class PlayerBoard:
    """
    Represents a player's complete board state.
    """
    player_id: int
    tiles: List[Tile] = field(default_factory=list)
    player_name: Optional[str] = None
    nature_tokens: int = 0
    source_image_path: Optional[str] = None
    
    def get_tiles_by_animal(self, animal: AnimalType) -> List[Tile]:
        """Get all tiles with a specific animal type"""
        return [t for t in self.tiles if t.animal == animal]
    
    def get_tile_at(self, coord: AxialCoord) -> Optional[Tile]:
        """Get tile at specific coordinate"""
        for tile in self.tiles:
            if tile.coord == coord:
                return tile
        return None
    
    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "tiles": [t.to_dict() for t in self.tiles],
            "nature_tokens": self.nature_tokens,
            "source_image_path": self.source_image_path
        }


@dataclass
class ScoreBreakdown:
    """
    Detailed score breakdown for a player.
    """
    bear_score: int = 0
    elk_score: int = 0
    salmon_score: int = 0
    hawk_score: int = 0
    fox_score: int = 0
    habitat_bonus: int = 0
    nature_token_score: int = 0
    
    # Detailed breakdown for verification
    bear_details: Dict = field(default_factory=dict)
    elk_details: Dict = field(default_factory=dict)
    salmon_details: Dict = field(default_factory=dict)
    hawk_details: Dict = field(default_factory=dict)
    fox_details: Dict = field(default_factory=dict)
    
    @property
    def total_score(self) -> int:
        return (self.bear_score + self.elk_score + self.salmon_score +
                self.hawk_score + self.fox_score + self.habitat_bonus +
                self.nature_token_score)
    
    def to_dict(self) -> Dict:
        return {
            "bear_score": self.bear_score,
            "elk_score": self.elk_score,
            "salmon_score": self.salmon_score,
            "hawk_score": self.hawk_score,
            "fox_score": self.fox_score,
            "habitat_bonus": self.habitat_bonus,
            "nature_token_score": self.nature_token_score,
            "total_score": self.total_score,
            "details": {
                "bear": self.bear_details,
                "elk": self.elk_details,
                "salmon": self.salmon_details,
                "hawk": self.hawk_details,
                "fox": self.fox_details
            }
        }


@dataclass
class PlayerResult:
    """
    Complete result for a single player.
    """
    player_id: int
    player_name: Optional[str]
    board: PlayerBoard
    scores: ScoreBreakdown
    
    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "board": self.board.to_dict(),
            "scores": self.scores.to_dict()
        }


@dataclass
class GameResult:
    """
    Complete game result with all players.
    """
    game_id: str
    players: List[PlayerResult]
    winner_id: int
    winner_name: Optional[str]
    winning_score: int
    margin: int
    
    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "players": [p.to_dict() for p in self.players],
            "winner": {
                "player_id": self.winner_id,
                "player_name": self.winner_name,
                "winning_score": self.winning_score,
                "margin": self.margin
            },
            "summary": {
                "scores": {f"Player {p.player_id}": p.scores.total_score 
                          for p in self.players}
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
