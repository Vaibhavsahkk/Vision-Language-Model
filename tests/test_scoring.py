"""
Phase 7: Testing & Reproducibility

This module provides:
- Unit tests for each scoring rule
- Edge case tests
- End-to-end pipeline tests
- Reproducibility verification

Run with: pytest tests/test_scoring.py -v
"""

import pytest
from typing import List

from src.models import (
    AnimalType, HabitatType, AxialCoord, Tile, PlayerBoard, ScoreBreakdown
)
from src.phase3_graph import BoardGraph, HexGrid
from src.phase4_scoring import (
    BearScorer, ElkScorer, SalmonScorer, HawkScorer, FoxScorer,
    ScoringEngine
)


# ============================================================================
# Test Fixtures - Helper Functions
# ============================================================================

def create_tile(q: int, r: int, animal: AnimalType, 
                habitat: HabitatType = HabitatType.UNKNOWN) -> Tile:
    """Helper to create a tile"""
    return Tile(
        id=f"tile_{q}_{r}",
        animal=animal,
        habitat=habitat,
        coord=AxialCoord(q, r)
    )


def create_board_with_tiles(player_id: int, tiles: List[Tile]) -> BoardGraph:
    """Helper to create a board graph from tiles"""
    player_board = PlayerBoard(player_id=player_id, tiles=tiles)
    board_graph = BoardGraph(player_id)
    board_graph.build_from_player_board(player_board)
    return board_graph


# ============================================================================
# Hex Grid Tests
# ============================================================================

class TestHexGrid:
    """Tests for hexagonal grid utilities"""
    
    def test_get_neighbors(self):
        """Test that we get exactly 6 neighbors"""
        coord = AxialCoord(0, 0)
        neighbors = HexGrid.get_neighbors(coord)
        assert len(neighbors) == 6
        
        expected = [
            AxialCoord(1, 0), AxialCoord(1, -1), AxialCoord(0, -1),
            AxialCoord(-1, 0), AxialCoord(-1, 1), AxialCoord(0, 1)
        ]
        for exp in expected:
            assert exp in neighbors
    
    def test_distance(self):
        """Test hex distance calculation"""
        assert HexGrid.distance(AxialCoord(0, 0), AxialCoord(0, 0)) == 0
        assert HexGrid.distance(AxialCoord(0, 0), AxialCoord(1, 0)) == 1
        assert HexGrid.distance(AxialCoord(0, 0), AxialCoord(2, 0)) == 2
        assert HexGrid.distance(AxialCoord(0, 0), AxialCoord(1, 1)) == 2
    
    def test_are_adjacent(self):
        """Test adjacency check"""
        assert HexGrid.are_adjacent(AxialCoord(0, 0), AxialCoord(1, 0))
        assert HexGrid.are_adjacent(AxialCoord(0, 0), AxialCoord(0, 1))
        assert not HexGrid.are_adjacent(AxialCoord(0, 0), AxialCoord(2, 0))
        assert not HexGrid.are_adjacent(AxialCoord(0, 0), AxialCoord(0, 0))


# ============================================================================
# Bear Scoring Tests
# ============================================================================

class TestBearScoring:
    """Tests for bear scoring rules - Card B (Mother and Cubs)"""
    
    def test_no_bears(self):
        """Test scoring with no bears"""
        tiles = [create_tile(0, 0, AnimalType.FOX)]
        board = create_board_with_tiles(1, tiles)
        
        scorer = BearScorer()
        score, details = scorer.score(board)
        
        assert score == 0
        assert details['num_groups_of_3'] == 0
    
    def test_single_bear(self):
        """Single bears don't score"""
        tiles = [create_tile(0, 0, AnimalType.BEAR)]
        board = create_board_with_tiles(1, tiles)
        
        scorer = BearScorer()
        score, details = scorer.score(board)
        
        assert score == 0
        assert details['num_groups_of_3'] == 0
        assert len(details['singles']) == 1
    
    def test_pair_no_score(self):
        """Pairs don't score in Card B"""
        tiles = [
            create_tile(0, 0, AnimalType.BEAR),
            create_tile(1, 0, AnimalType.BEAR),  # Adjacent to (0,0)
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = BearScorer()
        score, details = scorer.score(board)
        
        assert score == 0  # Pairs don't score in Card B
        assert details['num_groups_of_3'] == 0
    
    def test_group_of_three(self):
        """Group of exactly 3 = 10 points"""
        tiles = [
            create_tile(0, 0, AnimalType.BEAR),
            create_tile(1, 0, AnimalType.BEAR),
            create_tile(0, 1, AnimalType.BEAR),  # All three connected
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = BearScorer()
        score, details = scorer.score(board)
        
        assert score == 10
        assert details['num_groups_of_3'] == 1
    
    def test_group_of_four_no_score(self):
        """Groups of 4+ don't score"""
        tiles = [
            create_tile(0, 0, AnimalType.BEAR),
            create_tile(1, 0, AnimalType.BEAR),
            create_tile(0, 1, AnimalType.BEAR),
            create_tile(1, -1, AnimalType.BEAR),  # 4 connected
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = BearScorer()
        score, details = scorer.score(board)
        
        assert score == 0
        assert details['num_groups_of_3'] == 0


# ============================================================================
# Elk Scoring Tests  
# ============================================================================

class TestElkScoring:
    """Tests for elk scoring rules"""
    
    def test_no_elk(self):
        """No elk = 0 points"""
        tiles = [create_tile(0, 0, AnimalType.BEAR)]
        board = create_board_with_tiles(1, tiles)
        
        scorer = ElkScorer()
        score, details = scorer.score(board)
        
        assert score == 0
    
    def test_single_elk(self):
        """Single elk = 2 points"""
        tiles = [create_tile(0, 0, AnimalType.ELK)]
        board = create_board_with_tiles(1, tiles)
        
        scorer = ElkScorer()
        score, details = scorer.score(board)
        
        assert score == 2
    
    def test_elk_line_of_three(self):
        """Line of 3 elk = 9 points"""
        tiles = [
            create_tile(0, 0, AnimalType.ELK),
            create_tile(1, 0, AnimalType.ELK),
            create_tile(2, 0, AnimalType.ELK),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = ElkScorer()
        score, details = scorer.score(board)
        
        assert score == 9
    
    def test_multiple_elk_groups(self):
        """Multiple groups sum their scores"""
        tiles = [
            # Group 1: 2 elk = 5 points
            create_tile(0, 0, AnimalType.ELK),
            create_tile(1, 0, AnimalType.ELK),
            # Group 2: 1 elk = 2 points
            create_tile(10, 10, AnimalType.ELK),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = ElkScorer()
        score, details = scorer.score(board)
        
        assert score == 7  # 5 + 2


# ============================================================================
# Salmon Scoring Tests
# ============================================================================

class TestSalmonScoring:
    """Tests for salmon scoring rules - Card C"""
    
    def test_salmon_run_of_two_no_score(self):
        """Run of 2 salmon = 0 points (Card C needs 3+)"""
        tiles = [
            create_tile(0, 0, AnimalType.SALMON),
            create_tile(1, 0, AnimalType.SALMON),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = SalmonScorer()
        score, details = scorer.score(board)
        
        assert score == 0
    
    def test_salmon_run_of_three(self):
        """Run of 3 salmon = 10 points"""
        tiles = [
            create_tile(0, 0, AnimalType.SALMON),
            create_tile(1, 0, AnimalType.SALMON),
            create_tile(2, 0, AnimalType.SALMON),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = SalmonScorer()
        score, details = scorer.score(board)
        
        assert score == 10
    
    def test_salmon_run_of_four(self):
        """Run of 4 salmon = 12 points"""
        tiles = [create_tile(i, 0, AnimalType.SALMON) for i in range(4)]
        board = create_board_with_tiles(1, tiles)
        
        scorer = SalmonScorer()
        score, details = scorer.score(board)
        
        assert score == 12
    
    def test_salmon_run_of_five_plus(self):
        """Run of 5+ salmon = 15 points"""
        tiles = [create_tile(i, 0, AnimalType.SALMON) for i in range(7)]
        board = create_board_with_tiles(1, tiles)
        
        scorer = SalmonScorer()
        score, details = scorer.score(board)
        
        assert score == 15


# ============================================================================
# Hawk Scoring Tests
# ============================================================================

class TestHawkScoring:
    """Tests for hawk scoring rules - Card B"""
    
    def test_single_isolated_hawk_no_score(self):
        """Single isolated hawk = 0 points (Card B needs 2+)"""
        tiles = [
            create_tile(0, 0, AnimalType.HAWK),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = HawkScorer()
        score, details = scorer.score(board)
        
        assert score == 0
        assert details['isolated_hawks'] == 1
    
    def test_two_isolated_hawks(self):
        """2 isolated hawks = 5 points"""
        tiles = [
            create_tile(0, 0, AnimalType.HAWK),
            create_tile(5, 5, AnimalType.HAWK),  # Far away = isolated
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = HawkScorer()
        score, details = scorer.score(board)
        
        assert score == 5
        assert details['isolated_hawks'] == 2
    
    def test_three_isolated_hawks(self):
        """3 isolated hawks = 9 points"""
        tiles = [
            create_tile(0, 0, AnimalType.HAWK),
            create_tile(5, 5, AnimalType.HAWK),
            create_tile(10, 10, AnimalType.HAWK),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = HawkScorer()
        score, details = scorer.score(board)
        
        assert score == 9
        assert details['isolated_hawks'] == 3
    
    def test_adjacent_hawks_dont_score(self):
        """Adjacent hawks don't score"""
        tiles = [
            create_tile(0, 0, AnimalType.HAWK),
            create_tile(1, 0, AnimalType.HAWK),  # Adjacent
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = HawkScorer()
        score, details = scorer.score(board)
        
        assert score == 0
        assert details['isolated_hawks'] == 0


# ============================================================================
# Fox Scoring Tests
# ============================================================================

class TestFoxScoring:
    """Tests for fox scoring rules"""
    
    def test_fox_with_no_neighbors(self):
        """Fox with no animal neighbors = 0"""
        tiles = [create_tile(0, 0, AnimalType.FOX)]
        board = create_board_with_tiles(1, tiles)
        
        scorer = FoxScorer()
        score, details = scorer.score(board)
        
        assert score == 0
    
    def test_fox_with_diverse_neighbors(self):
        """Fox with 4 different neighbor types = 4 points"""
        tiles = [
            create_tile(0, 0, AnimalType.FOX),
            create_tile(1, 0, AnimalType.BEAR),
            create_tile(0, 1, AnimalType.ELK),
            create_tile(-1, 1, AnimalType.SALMON),
            create_tile(-1, 0, AnimalType.HAWK),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = FoxScorer()
        score, details = scorer.score(board)
        
        assert score == 4
    
    def test_fox_duplicate_neighbor_types(self):
        """Multiple neighbors of same type only count once"""
        tiles = [
            create_tile(0, 0, AnimalType.FOX),
            create_tile(1, 0, AnimalType.BEAR),
            create_tile(0, 1, AnimalType.BEAR),  # Same type
            create_tile(-1, 0, AnimalType.ELK),
        ]
        board = create_board_with_tiles(1, tiles)
        
        scorer = FoxScorer()
        score, details = scorer.score(board)
        
        assert score == 2  # Only bear and elk count


# ============================================================================
# Integration Tests
# ============================================================================

class TestScoringEngine:
    """Integration tests for complete scoring"""
    
    def test_complete_scoring(self):
        """Test full scoring with mixed animals - Card B/C rules"""
        tiles = [
            # Bear group of 3 (Card B scores 10)
            create_tile(0, 0, AnimalType.BEAR),
            create_tile(1, 0, AnimalType.BEAR),
            create_tile(0, 1, AnimalType.BEAR),
            # Elk group of 3 (9 points)
            create_tile(5, 0, AnimalType.ELK),
            create_tile(6, 0, AnimalType.ELK),
            create_tile(7, 0, AnimalType.ELK),
            # 2 Isolated hawks (Card B: 5 points)
            create_tile(10, 10, AnimalType.HAWK),
            create_tile(-5, -5, AnimalType.HAWK),
            # Salmon run of 3 (Card C: 10 points)
            create_tile(0, 5, AnimalType.SALMON),
            create_tile(1, 5, AnimalType.SALMON),
            create_tile(2, 5, AnimalType.SALMON),
            # Fox with neighbors
            create_tile(0, 4, AnimalType.FOX),
        ]
        board = create_board_with_tiles(1, tiles)
        
        engine = ScoringEngine()
        breakdown = engine.calculate_score(board)
        
        assert breakdown.bear_score == 10   # 1 group of 3
        assert breakdown.elk_score == 9     # 3 elk
        assert breakdown.hawk_score == 5    # 2 isolated
        assert breakdown.salmon_score == 10 # run of 3
        assert breakdown.fox_score == 1     # 1 adjacent type (salmon)
        assert breakdown.total_score == 10 + 9 + 5 + 10 + 1  # 35 total
    
    def test_reproducibility(self):
        """Same input should always give same output"""
        tiles = [
            create_tile(0, 0, AnimalType.BEAR),
            create_tile(1, 0, AnimalType.BEAR),
            create_tile(5, 5, AnimalType.SALMON),
            create_tile(6, 5, AnimalType.SALMON),
            create_tile(7, 5, AnimalType.SALMON),
        ]
        board = create_board_with_tiles(1, tiles)
        
        engine = ScoringEngine()
        
        # Run multiple times
        scores = [engine.calculate_score(board).total_score for _ in range(5)]
        
        # All should be identical
        assert len(set(scores)) == 1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
