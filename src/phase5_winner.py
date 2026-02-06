"""
Phase 5: Winner Determination

This module handles:
- Comparing total scores across all players
- Handling tie-breaker rules
- Recording final standings and margins

Cascadia Tie-breaker Rules:
1. Player with most wildlife tokens wins ties
2. If still tied, players share the victory
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from src.models import ScoreBreakdown, PlayerBoard, GameResult, PlayerResult
from src.phase3_graph import BoardGraph
from src.utils import get_logger, get_output_dir, ensure_dir, generate_game_id

logger = get_logger(__name__)


@dataclass
class PlayerStanding:
    """Represents a player's final standing"""
    player_id: int
    player_name: Optional[str]
    total_score: int
    token_count: int  # For tie-breaking
    rank: int
    is_winner: bool
    tied_with: List[int]  # Other player IDs if tied


class WinnerDeterminator:
    """
    Determines the winner based on scores and tie-breaker rules.
    """
    
    def determine_winner(self, 
                        scores: Dict[int, ScoreBreakdown],
                        player_boards: Optional[Dict[int, PlayerBoard]] = None,
                        player_names: Optional[Dict[int, str]] = None) -> Tuple[List[PlayerStanding], int]:
        """
        Determine winner and rankings.
        
        Args:
            scores: Dictionary mapping player_id to ScoreBreakdown
            player_boards: Optional player boards for tie-breaking (token count)
            player_names: Optional player names
            
        Returns:
            Tuple of (list of PlayerStanding objects, winner_id)
        """
        player_names = player_names or {}
        
        # Calculate totals and token counts
        player_data = []
        for player_id, breakdown in scores.items():
            total = breakdown.total_score
            
            # Get token count for tie-breaking
            if player_boards and player_id in player_boards:
                token_count = len(player_boards[player_id].tiles)
            else:
                # Estimate from score details
                token_count = self._estimate_token_count(breakdown)
            
            player_data.append({
                'player_id': player_id,
                'player_name': player_names.get(player_id),
                'total_score': total,
                'token_count': token_count
            })
        
        # Sort by score (descending), then by token count (descending) for ties
        player_data.sort(key=lambda x: (-x['total_score'], -x['token_count']))
        
        # Assign ranks and identify ties
        standings = []
        current_rank = 1
        
        for i, data in enumerate(player_data):
            # Check for ties with previous player
            tied_with = []
            if i > 0:
                prev = player_data[i - 1]
                if (data['total_score'] == prev['total_score'] and 
                    data['token_count'] == prev['token_count']):
                    # True tie (same score AND same token count)
                    # Keep same rank
                    current_rank = standings[-1].rank
                    tied_with = [prev['player_id']]
                    standings[-1].tied_with.append(data['player_id'])
            
            standing = PlayerStanding(
                player_id=data['player_id'],
                player_name=data['player_name'],
                total_score=data['total_score'],
                token_count=data['token_count'],
                rank=current_rank,
                is_winner=(current_rank == 1),
                tied_with=tied_with
            )
            standings.append(standing)
            
            # Update rank for next player (accounting for ties)
            if i + 1 < len(player_data):
                next_data = player_data[i + 1]
                if (next_data['total_score'] != data['total_score'] or
                    next_data['token_count'] != data['token_count']):
                    current_rank = i + 2
        
        # Get winner(s)
        winners = [s for s in standings if s.is_winner]
        winner_id = winners[0].player_id if winners else None
        
        # Log results
        self._log_standings(standings)
        
        return standings, winner_id
    
    def _estimate_token_count(self, breakdown: ScoreBreakdown) -> int:
        """Estimate token count from score breakdown details"""
        count = 0
        
        # Try to extract from details
        details = breakdown.bear_details
        if 'num_groups_of_3' in details:
            count += details['num_groups_of_3'] * 3
            count += len(details.get('singles', []))
            count += len(details.get('pairs', [])) * 2
            for group in details.get('groups_too_large', []):
                count += group.get('size', 0)
        
        # Add other animals similarly
        for animal_details in [breakdown.elk_details, breakdown.salmon_details]:
            for group in animal_details.get('groups', animal_details.get('runs', [])):
                count += group.get('size', group.get('length', 0))
        
        if 'total_hawks' in breakdown.hawk_details:
            count += breakdown.hawk_details['total_hawks']
        
        if 'num_foxes' in breakdown.fox_details:
            count += breakdown.fox_details['num_foxes']
        
        return count
    
    def _log_standings(self, standings: List[PlayerStanding]):
        """Log final standings"""
        logger.info("\n" + "="*50)
        logger.info("FINAL STANDINGS")
        logger.info("="*50)
        
        for standing in standings:
            name = standing.player_name or f"Player {standing.player_id}"
            winner_marker = " <- WINNER" if standing.is_winner else ""
            tie_note = f" (tied with Player {standing.tied_with})" if standing.tied_with else ""
            
            logger.info(f"  #{standing.rank}: {name} - {standing.total_score} points{winner_marker}{tie_note}")


def determine_game_winner(scores: Dict[int, ScoreBreakdown],
                          player_boards: Optional[List[PlayerBoard]] = None,
                          board_graphs: Optional[Dict[int, BoardGraph]] = None,
                          output_dir: Optional[Path] = None) -> GameResult:
    """
    Main function to determine game winner and create final result.
    
    Args:
        scores: Dictionary mapping player_id to ScoreBreakdown
        player_boards: Optional list of PlayerBoard objects
        board_graphs: Optional dictionary of BoardGraph objects
        output_dir: Output directory for results
        
    Returns:
        GameResult object with complete game results
    """
    output_dir = output_dir or get_output_dir()
    ensure_dir(output_dir)
    
    # Convert player_boards list to dict if needed
    boards_dict = None
    if player_boards:
        boards_dict = {pb.player_id: pb for pb in player_boards}
    
    # Determine winner
    determinator = WinnerDeterminator()
    standings, winner_id = determinator.determine_winner(scores, boards_dict)
    
    # Find winner details
    winner_standing = next((s for s in standings if s.player_id == winner_id), None)
    
    # Calculate margin of victory
    if len(standings) > 1:
        second_place_score = standings[1].total_score
        margin = winner_standing.total_score - second_place_score
    else:
        margin = 0
    
    # Create PlayerResult objects
    player_results = []
    for standing in standings:
        pb = boards_dict.get(standing.player_id) if boards_dict else None
        if pb is None:
            # Create minimal PlayerBoard if not available
            pb = PlayerBoard(player_id=standing.player_id, tiles=[])
        
        player_result = PlayerResult(
            player_id=standing.player_id,
            player_name=standing.player_name,
            board=pb,
            scores=scores[standing.player_id]
        )
        player_results.append(player_result)
    
    # Create GameResult
    game_result = GameResult(
        game_id=generate_game_id(),
        players=player_results,
        winner_id=winner_id,
        winner_name=winner_standing.player_name if winner_standing else None,
        winning_score=winner_standing.total_score if winner_standing else 0,
        margin=margin
    )
    
    # Save results
    results_path = output_dir / "game_result.json"
    with open(results_path, 'w') as f:
        f.write(game_result.to_json())
    
    logger.info(f"\nSaved game results to: {results_path}")
    
    # Print summary
    print_game_summary(game_result)
    
    return game_result


def print_game_summary(game_result: GameResult):
    """Print a formatted game summary"""
    print("\n" + "="*60)
    print("                    GAME RESULTS")
    print("="*60)
    
    # Scores table
    print("\n  PLAYER           | BEAR | ELK | SAL | HAWK | FOX | TOTAL")
    print("  " + "-"*58)
    
    for player in sorted(game_result.players, key=lambda p: -p.scores.total_score):
        name = player.player_name or f"Player {player.player_id}"
        s = player.scores
        
        winner_mark = "* " if player.player_id == game_result.winner_id else "  "
        
        print(f"  {winner_mark}{name:15s} | {s.bear_score:4d} | {s.elk_score:3d} | {s.salmon_score:3d} | "
              f"{s.hawk_score:4d} | {s.fox_score:3d} | {s.total_score:5d}")
    
    print("  " + "-"*58)
    
    # Winner announcement
    winner_name = game_result.winner_name or f"Player {game_result.winner_id}"
    print(f"\n  WINNER: {winner_name}")
    print(f"     Score: {game_result.winning_score} points")
    if game_result.margin > 0:
        print(f"     Margin of victory: {game_result.margin} points")
    
    print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    print("Phase 5: Winner Determination Module")
    print("="*50)
    print("\nThis module determines the game winner.")
    print("\nFeatures:")
    print("  - Score comparison")
    print("  - Tie-breaker handling")
    print("  - Final standings")
    print("  - Game result output")
    print("\nUsage:")
    print("  from phase5_winner import determine_game_winner")
    print("  result = determine_game_winner(scores, player_boards)")
