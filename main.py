"""
Cascadia Board Game Scorer - Main Pipeline

This is the main entry point that orchestrates all phases:
1. Image Preprocessing
2. Token Detection (VLM)
3. Graph Construction
4. Rule-Based Scoring
5. Winner Determination
6. Explanation Generation

Usage:
    python main.py --input game_board.png
    python main.py --input player1.png player2.png player3.png
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.utils import (
    get_logger, load_config, ensure_dir, get_output_dir, 
    get_input_dir, generate_game_id, Timer
)
from src.models import GameResult

logger = get_logger(__name__)


def run_pipeline(image_paths: List[str],
                 vlm_provider: str = "openai",
                 output_dir: Optional[Path] = None,
                 skip_vlm: bool = False,
                 manual_data_path: Optional[str] = None) -> GameResult:
    """
    Run the complete Cascadia scoring pipeline.
    
    Args:
        image_paths: List of image paths (either one combined image or separate per player)
        vlm_provider: VLM provider to use ("openai" or "google")
        output_dir: Output directory for results
        skip_vlm: If True, skip VLM detection and use manual data
        manual_data_path: Path to manual detection data JSON
        
    Returns:
        GameResult with complete scoring
    """
    output_dir = output_dir or get_output_dir()
    ensure_dir(output_dir)
    
    game_id = generate_game_id()
    logger.info(f"\n{'='*60}")
    logger.info(f"CASCADIA SCORING PIPELINE")
    logger.info(f"Game ID: {game_id}")
    logger.info(f"{'='*60}\n")
    
    # =========================================================================
    # Phase 1: Image Preprocessing
    # =========================================================================
    with Timer("Phase 1: Image Preprocessing"):
        from src.phase1_preprocessing import (
            preprocess_game_image, 
            load_separate_player_images
        )
        
        if len(image_paths) == 1:
            # Single combined image
            logger.info(f"Processing combined game image: {image_paths[0]}")
            player_board_images = preprocess_game_image(
                image_paths[0],
                output_dir=output_dir
            )
        else:
            # Separate images per player
            logger.info(f"Processing {len(image_paths)} separate player images")
            player_board_images = load_separate_player_images(
                image_paths,
                output_dir=output_dir
            )
    
    # =========================================================================
    # Phase 2: Token Detection
    # =========================================================================
    with Timer("Phase 2: Token Detection"):
        from src.phase2_detection import detect_all_players
        
        if skip_vlm and manual_data_path:
            # Load manual detection data
            logger.info(f"Loading manual detection data from: {manual_data_path}")
            from src.models import PlayerBoard, Tile, AxialCoord, AnimalType
            
            with open(manual_data_path) as f:
                manual_data = json.load(f)
            
            player_boards = []
            for player_data in manual_data['players']:
                tiles = []
                for tile_data in player_data['tiles']:
                    tile = Tile(
                        id=tile_data.get('id', f"tile_{len(tiles)}"),
                        animal=AnimalType(tile_data['animal']),
                        habitat=tile_data.get('habitat', 'unknown'),
                        coord=AxialCoord(
                            q=tile_data['coord']['q'],
                            r=tile_data['coord']['r']
                        )
                    )
                    tiles.append(tile)
                
                player_board = PlayerBoard(
                    player_id=player_data['player_id'],
                    tiles=tiles,
                    nature_tokens=player_data.get('nature_tokens', 0)
                )
                player_boards.append(player_board)
        else:
            # Use VLM for detection
            logger.info(f"Using {vlm_provider} VLM for token detection")
            player_boards = detect_all_players(
                player_board_images,
                vlm_provider=vlm_provider,
                output_dir=output_dir
            )
    
    # =========================================================================
    # Phase 3: Graph Construction
    # =========================================================================
    with Timer("Phase 3: Graph Construction"):
        from src.phase3_graph import build_player_graphs
        
        board_graphs = build_player_graphs(
            player_boards,
            output_dir=output_dir,
            visualize=True
        )
    
    # =========================================================================
    # Phase 4: Rule-Based Scoring
    # =========================================================================
    with Timer("Phase 4: Scoring"):
        from src.phase4_scoring import score_all_players
        
        nature_tokens = {pb.player_id: pb.nature_tokens for pb in player_boards}
        scores = score_all_players(
            board_graphs,
            nature_tokens=nature_tokens,
            output_dir=output_dir
        )
    
    # =========================================================================
    # Phase 5: Winner Determination
    # =========================================================================
    with Timer("Phase 5: Winner Determination"):
        from src.phase5_winner import determine_game_winner
        
        game_result = determine_game_winner(
            scores,
            player_boards=player_boards,
            board_graphs=board_graphs,
            output_dir=output_dir
        )
        game_result.game_id = game_id
    
    # =========================================================================
    # Phase 6: Generate Explanations
    # =========================================================================
    with Timer("Phase 6: Explanations"):
        from src.phase6_explanation import generate_explanations
        
        report_path = generate_explanations(
            game_result,
            board_graphs,
            output_dir=output_dir
        )
    
    # =========================================================================
    # Final Output
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Verification report: {report_path}")
    
    return game_result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Cascadia Board Game Scorer using VLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single combined game image
  python main.py --input game_board.png
  
  # Process separate player images
  python main.py --input player1.png player2.png player3.png
  
  # Use Google Gemini instead of OpenAI
  python main.py --input game_board.png --vlm google
  
  # Use manual detection data (skip VLM)
  python main.py --input game_board.png --manual-data detections.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        required=True,
        help='Input image(s). Either one combined image or separate per-player images.'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--vlm',
        type=str,
        choices=['openai', 'google'],
        default='openai',
        help='VLM provider to use (default: openai)'
    )
    
    parser.add_argument(
        '--manual-data',
        type=str,
        default=None,
        help='Path to manual detection data JSON (skips VLM)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for path in args.input:
        if not Path(path).exists():
            logger.error(f"Input file not found: {path}")
            sys.exit(1)
    
    # Set output directory
    output_dir = Path(args.output) if args.output else None
    
    # Run pipeline
    try:
        skip_vlm = args.manual_data is not None
        
        result = run_pipeline(
            image_paths=args.input,
            vlm_provider=args.vlm,
            output_dir=output_dir,
            skip_vlm=skip_vlm,
            manual_data_path=args.manual_data
        )
        
        # Print final result
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(result.to_json())
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
