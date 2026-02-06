"""
Phase 4: Rule-Based Scoring Engine

This module implements the EXACT Cascadia scoring rules.
ALL scoring logic is deterministic Python - no VLM involvement.

Scoring Rules Implemented (Based on Challenge Image Cards):
- Bear (Card B - Mother and Cubs): Groups of exactly 3 = 10 points
- Elk (Card B): Contiguous line formations (1=2, 2=5, 3=9, 4=13)
- Salmon (Card C): Non-branching runs (3=10, 4=12, 5+=15)
- Hawk (Card B): Isolated hawks (2=5, 3=9, 4=12, 5=16, 6=20, 7=24)
- Fox: Points based on unique adjacent animal types

Design Principles:
- Pure Python, no ML/VLM
- Graph algorithms from Phase 3
- Fully deterministic and reproducible
- Detailed logging for verification
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from src.models import (
    AnimalType, HabitatType, Tile, PlayerBoard, ScoreBreakdown
)
from src.phase3_graph import BoardGraph, HexGrid
from src.utils import get_logger, load_scoring_rules, get_output_dir, ensure_dir

logger = get_logger(__name__)


# ============================================================================
# Scoring Tables (from challenge image scoring cards)
# ============================================================================

# Bear Card B - "Mother and Cubs"
# Only groups of exactly 3 score (10 points each)
BEAR_SCORING = {
    1: 0,   # Single bear = 0
    2: 0,   # Pair = 0
    3: 10,  # Group of 3 = 10 points
    4: 0,   # Group of 4+ = 0
}

# Elk Card B
ELK_SCORING = {
    1: 2,
    2: 5,
    3: 9,
    4: 13,
}

# Salmon Card C
# Minimum 3 salmon to score
SALMON_SCORING = {
    1: 0,
    2: 0,
    3: 10,
    4: 12,
    5: 15,  # 5+ all score 15
}

# Hawk Card B
# Score based on number of isolated hawks
HAWK_SCORING = {
    1: 0,
    2: 5,
    3: 9,
    4: 12,
    5: 16,
    6: 20,
    7: 24,
}

FOX_SCORING = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}


# ============================================================================
# Individual Animal Scorers
# ============================================================================

class BearScorer:
    """
    Scores bears based on Card B - "Mother and Cubs".
    
    Rules (from scoring card):
    - Only groups of EXACTLY 3 adjacent bears score
    - Each group of 3 = 10 points
    - Single bears = 0 points
    - Pairs = 0 points
    - Groups of 4+ = 0 points
    """
    
    def __init__(self):
        self.scoring_table = BEAR_SCORING
    
    def score(self, board_graph: BoardGraph) -> Tuple[int, Dict]:
        """
        Calculate bear score based on groups of exactly 3.
        
        Returns:
            Tuple of (score, details_dict)
        """
        # Find all connected components of bears
        components = board_graph.find_connected_components(AnimalType.BEAR)
        
        # Count groups of exactly 3
        valid_groups = []
        singles = []
        pairs = []
        too_large = []
        
        for group in components:
            if len(group) == 3:
                valid_groups.append(group)
            elif len(group) == 1:
                singles.append(group[0])
            elif len(group) == 2:
                pairs.append(group)
            else:
                too_large.append(group)
        
        num_valid = len(valid_groups)
        score = num_valid * 10  # Each group of 3 = 10 points
        
        details = {
            "num_groups_of_3": num_valid,
            "valid_groups": [
                {"tiles": [f"({t.coord.q},{t.coord.r})" for t in grp]}
                for grp in valid_groups
            ],
            "singles": [f"({t.coord.q},{t.coord.r})" for t in singles],
            "pairs": [
                {"tiles": [f"({t.coord.q},{t.coord.r})" for t in p]}
                for p in pairs
            ],
            "groups_too_large": [
                {"size": len(g), "tiles": [f"({t.coord.q},{t.coord.r})" for t in g]}
                for g in too_large
            ],
            "scoring_explanation": f"{num_valid} group(s) of 3 bears = {score} points"
        }
        
        logger.info(f"Bear scoring (Card B): {num_valid} groups of 3 = {score} points")
        if singles:
            logger.info(f"  (Singles not scoring: {len(singles)})")
        if pairs:
            logger.info(f"  (Pairs not scoring: {len(pairs)})")
        if too_large:
            logger.info(f"  (Groups of 4+ not scoring: {len(too_large)})")
        
        return score, details


class ElkScorer:
    """
    Scores elk based on LINE FORMATIONS.
    
    Rules (from scoring card):
    - Elk score based on contiguous line formations
    - Lines can be straight or slightly curved
    - Each elk group scores based on its size
    - Score: 1=2, 2=5, 3=9, 4=13
    """
    
    def __init__(self):
        self.scoring_table = ELK_SCORING
    
    def score(self, board_graph: BoardGraph) -> Tuple[int, Dict]:
        """Calculate elk score based on line formations."""
        components = board_graph.find_connected_components(AnimalType.ELK)
        
        total_score = 0
        group_details = []
        
        for group in components:
            group_size = len(group)
            
            # Score based on size, capped at 4
            group_score = self.scoring_table.get(
                min(group_size, 4),
                self.scoring_table[4]
            )
            total_score += group_score
            
            group_details.append({
                "size": group_size,
                "score": group_score,
                "tiles": [f"({t.coord.q},{t.coord.r})" for t in group]
            })
        
        details = {
            "num_groups": len(components),
            "groups": group_details,
            "scoring_explanation": f"Sum of group scores = {total_score} points"
        }
        
        logger.info(f"Elk scoring: {len(components)} groups, total = {total_score} points")
        for g in group_details:
            logger.info(f"  Group of {g['size']} = {g['score']} points")
        
        return total_score, details


class SalmonScorer:
    """
    Scores salmon based on RUNS (non-branching chains) - Card C.
    
    Rules (from scoring card):
    - Salmon score based on the length of their run
    - A run is a connected chain with NO BRANCHES
    - Minimum 3 salmon needed to score
    - Score: 1-2=0, 3=10, 4=12, 5+=15
    """
    
    def __init__(self):
        self.scoring_table = SALMON_SCORING
    
    def score(self, board_graph: BoardGraph) -> Tuple[int, Dict]:
        """Calculate salmon score based on run lengths."""
        # Find chains (non-branching paths)
        chains = board_graph.find_chains(AnimalType.SALMON)
        
        total_score = 0
        run_details = []
        
        for chain in chains:
            run_length = len(chain)
            
            # Score based on length (Card C rules)
            if run_length <= 2:
                run_score = 0
            elif run_length == 3:
                run_score = 10
            elif run_length == 4:
                run_score = 12
            else:  # 5+
                run_score = 15
            
            total_score += run_score
            
            run_details.append({
                "length": run_length,
                "score": run_score,
                "tiles": [f"({t.coord.q},{t.coord.r})" for t in chain]
            })
        
        details = {
            "num_runs": len(chains),
            "runs": run_details,
            "scoring_explanation": f"Sum of run scores = {total_score} points"
        }
        
        logger.info(f"Salmon scoring (Card C): {len(chains)} runs, total = {total_score} points")
        for r in run_details:
            logger.info(f"  Run of {r['length']} = {r['score']} points")
        
        return total_score, details


class HawkScorer:
    """
    Scores hawks based on ISOLATION - Card B.
    
    Rules (from scoring card):
    - Only ISOLATED hawks score (not adjacent to other hawks)
    - Hawks adjacent to other hawks score 0
    - Need at least 2 isolated hawks to score
    - Score: 1=0, 2=5, 3=9, 4=12, 5=16, 6=20, 7=24
    """
    
    def __init__(self):
        self.scoring_table = HAWK_SCORING
    
    def score(self, board_graph: BoardGraph) -> Tuple[int, Dict]:
        """Calculate hawk score based on isolated hawks."""
        # Find isolated hawks
        isolated_hawks = board_graph.find_isolated_tiles(AnimalType.HAWK)
        num_isolated = len(isolated_hawks)
        
        # Find non-isolated hawks
        all_hawks = board_graph.get_tiles_by_animal(AnimalType.HAWK)
        non_isolated = [h for h in all_hawks if h not in isolated_hawks]
        
        # Score based on count (Card B rules)
        if num_isolated <= 1:
            score = 0
        elif num_isolated == 2:
            score = 5
        elif num_isolated == 3:
            score = 9
        elif num_isolated == 4:
            score = 12
        elif num_isolated == 5:
            score = 16
        elif num_isolated == 6:
            score = 20
        else:  # 7+
            score = 24
        
        details = {
            "total_hawks": len(all_hawks),
            "isolated_hawks": num_isolated,
            "non_isolated_hawks": len(non_isolated),
            "isolated_positions": [f"({t.coord.q},{t.coord.r})" for t in isolated_hawks],
            "non_isolated_positions": [f"({t.coord.q},{t.coord.r})" for t in non_isolated],
            "scoring_explanation": f"{num_isolated} isolated hawk(s) = {score} points"
        }
        
        logger.info(f"Hawk scoring: {num_isolated} isolated (of {len(all_hawks)} total) = {score} points")
        
        return score, details


class FoxScorer:
    """
    Scores foxes based on ADJACENT DIVERSITY.
    
    Rules (from scoring card):
    - Each fox scores points equal to the number of UNIQUE 
      OTHER animal types adjacent to it
    - A fox adjacent to bear, elk, salmon would score 3
    - Maximum is 4 (all other animal types)
    """
    
    def __init__(self):
        self.scoring_table = FOX_SCORING
    
    def score(self, board_graph: BoardGraph) -> Tuple[int, Dict]:
        """Calculate fox score based on adjacent diversity."""
        foxes = board_graph.get_tiles_by_animal(AnimalType.FOX)
        
        total_score = 0
        fox_details = []
        
        for fox in foxes:
            # Count unique adjacent animals (excluding fox itself and NONE)
            adjacent_animals = board_graph.get_adjacent_animals(fox)
            adjacent_animals.discard(AnimalType.FOX)
            adjacent_animals.discard(AnimalType.NONE)
            
            num_unique = len(adjacent_animals)
            fox_score = self.scoring_table.get(num_unique, num_unique)
            total_score += fox_score
            
            fox_details.append({
                "position": f"({fox.coord.q},{fox.coord.r})",
                "adjacent_types": [a.value for a in adjacent_animals],
                "unique_count": num_unique,
                "score": fox_score
            })
        
        details = {
            "num_foxes": len(foxes),
            "foxes": fox_details,
            "scoring_explanation": f"Sum of fox scores = {total_score} points"
        }
        
        logger.info(f"Fox scoring: {len(foxes)} foxes, total = {total_score} points")
        for f in fox_details:
            logger.info(f"  Fox at {f['position']}: {f['unique_count']} types = {f['score']} points")
        
        return total_score, details


# ============================================================================
# Habitat Scoring (Bonus)
# ============================================================================

class HabitatScorer:
    """
    Scores habitat bonuses (optional).
    
    Rules:
    - Largest contiguous area of each habitat type earns bonus
    - Typically compared between players
    """
    
    def score(self, board_graph: BoardGraph) -> Tuple[int, Dict]:
        """Calculate habitat bonus score."""
        # This would require habitat data which may not be fully detected
        # Returning 0 for now - can be implemented if habitat detection is added
        
        details = {
            "note": "Habitat scoring not implemented - requires habitat detection",
            "habitat_areas": {}
        }
        
        return 0, details


# ============================================================================
# Complete Scoring Engine
# ============================================================================

class ScoringEngine:
    """
    Complete scoring engine that calculates all scores for a player.
    """
    
    def __init__(self):
        self.bear_scorer = BearScorer()
        self.elk_scorer = ElkScorer()
        self.salmon_scorer = SalmonScorer()
        self.hawk_scorer = HawkScorer()
        self.fox_scorer = FoxScorer()
        self.habitat_scorer = HabitatScorer()
    
    def calculate_score(self, board_graph: BoardGraph, 
                        nature_tokens: int = 0) -> ScoreBreakdown:
        """
        Calculate complete score breakdown for a player.
        
        Args:
            board_graph: The player's board graph
            nature_tokens: Number of leftover nature tokens
            
        Returns:
            ScoreBreakdown with all scores and details
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Scoring Player {board_graph.player_id}")
        logger.info(f"{'='*50}")
        
        # Calculate each animal score
        bear_score, bear_details = self.bear_scorer.score(board_graph)
        elk_score, elk_details = self.elk_scorer.score(board_graph)
        salmon_score, salmon_details = self.salmon_scorer.score(board_graph)
        hawk_score, hawk_details = self.hawk_scorer.score(board_graph)
        fox_score, fox_details = self.fox_scorer.score(board_graph)
        
        # Habitat bonus
        habitat_score, habitat_details = self.habitat_scorer.score(board_graph)
        
        # Nature tokens
        nature_token_score = nature_tokens * 1  # 1 point per token
        
        # Create breakdown
        breakdown = ScoreBreakdown(
            bear_score=bear_score,
            elk_score=elk_score,
            salmon_score=salmon_score,
            hawk_score=hawk_score,
            fox_score=fox_score,
            habitat_bonus=habitat_score,
            nature_token_score=nature_token_score,
            bear_details=bear_details,
            elk_details=elk_details,
            salmon_details=salmon_details,
            hawk_details=hawk_details,
            fox_details=fox_details
        )
        
        # Log summary
        logger.info(f"\nScore Summary for Player {board_graph.player_id}:")
        logger.info(f"  Bear:    {bear_score:3d}")
        logger.info(f"  Elk:     {elk_score:3d}")
        logger.info(f"  Salmon:  {salmon_score:3d}")
        logger.info(f"  Hawk:    {hawk_score:3d}")
        logger.info(f"  Fox:     {fox_score:3d}")
        logger.info(f"  Habitat: {habitat_score:3d}")
        logger.info(f"  Nature:  {nature_token_score:3d}")
        logger.info(f"  ----------------")
        logger.info(f"  TOTAL:   {breakdown.total_score:3d}")
        
        return breakdown


# ============================================================================
# Main Scoring Pipeline
# ============================================================================

def score_all_players(board_graphs: Dict[int, BoardGraph],
                      nature_tokens: Optional[Dict[int, int]] = None,
                      output_dir: Optional[Path] = None) -> Dict[int, ScoreBreakdown]:
    """
    Calculate scores for all players.
    
    Args:
        board_graphs: Dictionary mapping player_id to BoardGraph
        nature_tokens: Optional dict mapping player_id to nature token count
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary mapping player_id to ScoreBreakdown
    """
    output_dir = output_dir or get_output_dir()
    ensure_dir(output_dir)
    
    engine = ScoringEngine()
    nature_tokens = nature_tokens or {}
    
    scores = {}
    
    for player_id, board_graph in board_graphs.items():
        tokens = nature_tokens.get(player_id, 0)
        breakdown = engine.calculate_score(board_graph, tokens)
        scores[player_id] = breakdown
    
    # Save detailed results
    results = {
        f"Player {pid}": breakdown.to_dict()
        for pid, breakdown in scores.items()
    }
    
    results_path = output_dir / "score_breakdown.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nSaved score breakdown to: {results_path}")
    
    return scores


# Example usage and testing
if __name__ == "__main__":
    print("Phase 4: Scoring Engine Module")
    print("="*50)
    print("\nThis module implements Cascadia scoring rules.")
    print("\nScoring implemented (Challenge Image Cards):")
    print("  - Bear (Card B): Groups of exactly 3")
    print("  - Elk (Card B): Line formations")
    print("  - Salmon (Card C): Non-branching runs (min 3)")
    print("  - Hawk (Card B): Isolated hawks (min 2)")
    print("  - Fox: Adjacent diversity")
    print("\nUsage:")
    print("  from phase4_scoring import score_all_players")
    print("  scores = score_all_players(board_graphs)")
