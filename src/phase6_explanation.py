"""
Phase 6: Explanation & Verification

This module handles:
- Generating annotated images with scoring formations highlighted
- Using VLM to create human-readable explanations
- Creating verification reports

Key Purpose:
- Meet "clarity of reasoning" requirement
- Make results verifiable
- Provide visual proof of scoring
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.patches as mpatches

from src.models import (
    AnimalType, ScoreBreakdown, GameResult, PlayerResult, PlayerBoard
)
from src.phase3_graph import BoardGraph, HexGrid
from src.phase2_detection import VLMClient, get_vlm_client
from src.utils import get_logger, ensure_dir, get_output_dir

logger = get_logger(__name__)


# ============================================================================
# Annotated Image Generation
# ============================================================================

class ScoringAnnotator:
    """
    Creates annotated images highlighting scoring formations.
    """
    
    ANIMAL_COLORS = {
        AnimalType.BEAR: (139, 69, 19),      # Saddle brown
        AnimalType.ELK: (210, 180, 140),      # Tan
        AnimalType.SALMON: (250, 128, 114),   # Salmon
        AnimalType.HAWK: (70, 130, 180),      # Steel blue
        AnimalType.FOX: (255, 99, 71),        # Tomato
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or get_output_dir()
        ensure_dir(self.output_dir)
    
    def create_scoring_diagram(self, 
                                board_graph: BoardGraph,
                                score_breakdown: ScoreBreakdown,
                                player_id: int) -> Path:
        """
        Create a diagram showing all scoring formations.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Player {player_id} Scoring Breakdown (Total: {score_breakdown.total_score})', 
                     fontsize=16, fontweight='bold')
        
        animals = [
            (AnimalType.BEAR, score_breakdown.bear_score, score_breakdown.bear_details, 'Bear Groups'),
            (AnimalType.ELK, score_breakdown.elk_score, score_breakdown.elk_details, 'Elk Lines'),
            (AnimalType.SALMON, score_breakdown.salmon_score, score_breakdown.salmon_details, 'Salmon Runs'),
            (AnimalType.HAWK, score_breakdown.hawk_score, score_breakdown.hawk_details, 'Isolated Hawks'),
            (AnimalType.FOX, score_breakdown.fox_score, score_breakdown.fox_details, 'Fox Diversity'),
        ]
        
        for idx, (animal, score, details, title) in enumerate(animals):
            ax = axes[idx // 3, idx % 3]
            self._draw_animal_scoring(ax, board_graph, animal, details, title, score)
        
        # Summary in last subplot
        ax = axes[1, 2]
        self._draw_score_summary(ax, score_breakdown)
        
        plt.tight_layout()
        
        # Save
        diagrams_dir = self.output_dir / "diagrams"
        ensure_dir(diagrams_dir)
        filepath = diagrams_dir / f"player_{player_id}_scoring_diagram.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved scoring diagram: {filepath}")
        return filepath
    
    def _draw_animal_scoring(self, ax, board_graph: BoardGraph, 
                              animal: AnimalType, details: Dict,
                              title: str, score: int):
        """Draw scoring visualization for one animal type"""
        ax.set_title(f'{title}: {score} pts', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        
        # Get all tiles of this animal
        animal_tiles = board_graph.get_tiles_by_animal(animal)
        
        if not animal_tiles:
            ax.text(0.5, 0.5, 'No tokens', ha='center', va='center', 
                    fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return
        
        # Calculate bounds
        qs = [t.coord.q for t in animal_tiles]
        rs = [t.coord.r for t in animal_tiles]
        
        # Draw hexes
        color = np.array(self.ANIMAL_COLORS.get(animal, (128, 128, 128))) / 255.0
        
        for tile in animal_tiles:
            x, y = HexGrid.axial_to_pixel(tile.coord, 1.0)
            
            hex_patch = RegularPolygon(
                (x, -y), numVertices=6, radius=0.9,
                orientation=np.pi/6,
                facecolor=color, edgecolor='black', linewidth=2
            )
            ax.add_patch(hex_patch)
            
            # Add coordinate label
            ax.text(x, -y, f'{tile.coord.q},{tile.coord.r}', 
                    ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold')
        
        # Draw connections for relevant animals
        if animal in [AnimalType.BEAR, AnimalType.SALMON, AnimalType.ELK]:
            subgraph = board_graph.get_animal_subgraph(animal)
            for edge in subgraph.edges():
                p1 = HexGrid.axial_to_pixel(
                    board_graph.coord_to_tile[edge[0]].coord, 1.0)
                p2 = HexGrid.axial_to_pixel(
                    board_graph.coord_to_tile[edge[1]].coord, 1.0)
                ax.plot([p1[0], p2[0]], [-p1[1], -p2[1]], 
                        'k-', linewidth=3, zorder=0)
        
        # Auto-scale
        all_tiles = list(board_graph.coord_to_tile.values())
        if all_tiles:
            all_x = [HexGrid.axial_to_pixel(t.coord, 1.0)[0] for t in all_tiles]
            all_y = [-HexGrid.axial_to_pixel(t.coord, 1.0)[1] for t in all_tiles]
            margin = 2
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        ax.axis('off')
    
    def _draw_score_summary(self, ax, breakdown: ScoreBreakdown):
        """Draw score summary table"""
        ax.axis('off')
        ax.set_title('Score Summary', fontsize=12, fontweight='bold')
        
        # Create table data
        data = [
            ['Animal', 'Score'],
            ['Bear', str(breakdown.bear_score)],
            ['Elk', str(breakdown.elk_score)],
            ['Salmon', str(breakdown.salmon_score)],
            ['Hawk', str(breakdown.hawk_score)],
            ['Fox', str(breakdown.fox_score)],
            ['Habitat', str(breakdown.habitat_bonus)],
            ['Nature', str(breakdown.nature_token_score)],
            ['TOTAL', str(breakdown.total_score)],
        ]
        
        # Draw as text
        y_pos = 0.9
        for i, row in enumerate(data):
            weight = 'bold' if i == 0 or i == len(data) - 1 else 'normal'
            ax.text(0.2, y_pos, row[0], fontsize=11, fontweight=weight,
                    transform=ax.transAxes)
            ax.text(0.7, y_pos, row[1], fontsize=11, fontweight=weight,
                    ha='right', transform=ax.transAxes)
            y_pos -= 0.1


# ============================================================================
# Natural Language Explanation Generation
# ============================================================================

class ExplanationGenerator:
    """
    Generates human-readable explanations of scoring.
    Can optionally use VLM for more natural language.
    """
    
    def __init__(self, use_vlm: bool = False, vlm_provider: str = "openai"):
        self.use_vlm = use_vlm
        if use_vlm:
            try:
                self.vlm_client = get_vlm_client(vlm_provider)
            except Exception as e:
                logger.warning(f"Could not initialize VLM: {e}. Using template-based explanations.")
                self.use_vlm = False
    
    def generate_player_explanation(self, 
                                    player_id: int,
                                    score_breakdown: ScoreBreakdown) -> str:
        """Generate explanation for a player's scores"""
        
        if self.use_vlm:
            return self._generate_vlm_explanation(player_id, score_breakdown)
        else:
            return self._generate_template_explanation(player_id, score_breakdown)
    
    def _generate_template_explanation(self, player_id: int, 
                                        breakdown: ScoreBreakdown) -> str:
        """Generate explanation using templates"""
        lines = [
            f"## Player {player_id} Score Explanation",
            f"**Total Score: {breakdown.total_score} points**\n",
        ]
        
        # Bear explanation (Card B - Mother and Cubs: groups of 3)
        bear = breakdown.bear_details
        if bear.get('num_groups_of_3', 0) > 0:
            lines.append(f"### Bear Score: {breakdown.bear_score} points")
            lines.append(f"Player {player_id} formed {bear['num_groups_of_3']} group(s) of exactly 3 bears.")
            if bear.get('singles'):
                lines.append(f"({len(bear['singles'])} single bear(s) did not score)")
            if bear.get('pairs'):
                lines.append(f"({len(bear['pairs'])} pair(s) did not score)")
            lines.append("")
        else:
            lines.append(f"### Bear Score: 0 points")
            lines.append("No valid groups of 3 bears were formed.\n")
        
        # Elk explanation
        elk = breakdown.elk_details
        if elk.get('groups'):
            lines.append(f"### Elk Score: {breakdown.elk_score} points")
            for i, group in enumerate(elk['groups'], 1):
                lines.append(f"- Group {i}: {group['size']} elk in a line = {group['score']} pts")
            lines.append("")
        else:
            lines.append(f"### Elk Score: 0 points\n")
        
        # Salmon explanation
        salmon = breakdown.salmon_details
        if salmon.get('runs'):
            lines.append(f"### Salmon Score: {breakdown.salmon_score} points")
            for i, run in enumerate(salmon['runs'], 1):
                lines.append(f"- Run {i}: {run['length']} salmon in a chain = {run['score']} pts")
            lines.append("")
        else:
            lines.append(f"### Salmon Score: 0 points\n")
        
        # Hawk explanation
        hawk = breakdown.hawk_details
        lines.append(f"### Hawk Score: {breakdown.hawk_score} points")
        isolated = hawk.get('isolated_hawks', 0)
        total = hawk.get('total_hawks', 0)
        lines.append(f"{isolated} out of {total} hawks were isolated (not adjacent to other hawks).")
        lines.append("")
        
        # Fox explanation
        fox = breakdown.fox_details
        if fox.get('foxes'):
            lines.append(f"### Fox Score: {breakdown.fox_score} points")
            for f in fox['foxes']:
                types = ', '.join(f['adjacent_types']) if f['adjacent_types'] else 'none'
                lines.append(f"- Fox at {f['position']}: {f['unique_count']} unique adjacent types ({types}) = {f['score']} pts")
            lines.append("")
        else:
            lines.append(f"### Fox Score: 0 points\n")
        
        return '\n'.join(lines)
    
    def _generate_vlm_explanation(self, player_id: int, 
                                   breakdown: ScoreBreakdown) -> str:
        """Use VLM to generate natural language explanation"""
        prompt = f"""Based on this Cascadia board game scoring data, write a clear, 
natural explanation of how Player {player_id} scored their points.

Scoring Data:
{json.dumps(breakdown.to_dict(), indent=2)}

Write 2-3 sentences for each animal type explaining:
1. How many scoring formations they created
2. Why those formations scored (reference the rules)
3. The points earned

Keep the tone informative but engaging. End with a brief overall assessment.
"""
        
        try:
            # Note: VLM doesn't need an image for this, but we'd need a text-only endpoint
            # For now, fall back to template
            return self._generate_template_explanation(player_id, breakdown)
        except Exception as e:
            logger.warning(f"VLM explanation failed: {e}")
            return self._generate_template_explanation(player_id, breakdown)
    
    def generate_game_summary(self, game_result: GameResult) -> str:
        """Generate summary explanation for entire game"""
        lines = [
            "# Cascadia Game Analysis Report",
            f"\n**Game ID:** {game_result.game_id}",
            f"\n## Final Results\n"
        ]
        
        # Rankings
        sorted_players = sorted(game_result.players, 
                                key=lambda p: -p.scores.total_score)
        
        for rank, player in enumerate(sorted_players, 1):
            name = player.player_name or f"Player {player.player_id}"
            winner = " 🏆" if player.player_id == game_result.winner_id else ""
            lines.append(f"**#{rank} {name}{winner}**: {player.scores.total_score} points")
        
        # Winner announcement
        winner_name = game_result.winner_name or f"Player {game_result.winner_id}"
        lines.append(f"\n## Winner: {winner_name}")
        lines.append(f"Winning score: {game_result.winning_score} points")
        if game_result.margin > 0:
            lines.append(f"Margin of victory: {game_result.margin} points")
        
        # Individual breakdowns
        lines.append("\n---\n")
        for player in game_result.players:
            explanation = self.generate_player_explanation(
                player.player_id, player.scores)
            lines.append(explanation)
            lines.append("\n---\n")
        
        return '\n'.join(lines)


# ============================================================================
# Verification Report
# ============================================================================

class VerificationReportGenerator:
    """
    Generates a complete verification report with all evidence.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or get_output_dir()
        ensure_dir(self.output_dir)
        self.annotator = ScoringAnnotator(self.output_dir)
        self.explainer = ExplanationGenerator(use_vlm=False)
    
    def generate_report(self, 
                        game_result: GameResult,
                        board_graphs: Dict[int, BoardGraph]) -> Path:
        """
        Generate complete verification report.
        
        Returns:
            Path to the generated report
        """
        report_dir = self.output_dir / "report"
        ensure_dir(report_dir)
        
        # Generate diagrams for each player
        diagram_paths = []
        for player in game_result.players:
            if player.player_id in board_graphs:
                path = self.annotator.create_scoring_diagram(
                    board_graphs[player.player_id],
                    player.scores,
                    player.player_id
                )
                diagram_paths.append(path)
        
        # Generate text report
        report_content = self.explainer.generate_game_summary(game_result)
        
        # Add diagram references
        report_content += "\n\n## Visual Evidence\n\n"
        for path in diagram_paths:
            report_content += f"- [{path.name}]({path.name})\n"
        
        # Save report
        report_path = report_dir / "verification_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Also save as JSON for programmatic access
        json_path = report_dir / "verification_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(game_result.to_json())
        
        logger.info(f"Generated verification report: {report_path}")
        
        return report_path


# ============================================================================
# Main Pipeline
# ============================================================================

def generate_explanations(game_result: GameResult,
                          board_graphs: Dict[int, BoardGraph],
                          output_dir: Optional[Path] = None) -> Path:
    """
    Generate all explanations and verification materials.
    
    Args:
        game_result: The complete game result
        board_graphs: Dictionary of board graphs
        output_dir: Output directory
        
    Returns:
        Path to the verification report
    """
    output_dir = output_dir or get_output_dir()
    
    report_generator = VerificationReportGenerator(output_dir)
    report_path = report_generator.generate_report(game_result, board_graphs)
    
    return report_path


# Example usage
if __name__ == "__main__":
    print("Phase 6: Explanation & Verification Module")
    print("="*50)
    print("\nThis module generates:")
    print("  - Annotated scoring diagrams")
    print("  - Natural language explanations")
    print("  - Verification reports")
    print("\nUsage:")
    print("  from phase6_explanation import generate_explanations")
    print("  report_path = generate_explanations(game_result, board_graphs)")
