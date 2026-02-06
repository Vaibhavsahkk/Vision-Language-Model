"""
Phase 2: Tile & Token Detection (Vision → Data)

This module handles:
- Using VLM to identify animals and habitats on each tile
- Converting visual information into structured board state
- Optional: OpenCV-assisted token center detection

Design Decision:
- VLM is used for IDENTIFICATION (what animal is on each tile)
- We rely on VLM's spatial understanding to get positions
- Results are validated through visual overlays

Supported VLM Providers:
- OpenAI GPT-4 Vision
- Google Gemini Vision
"""

import os
import json
import base64
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time

import cv2
import numpy as np
from PIL import Image

from src.models import (
    AnimalType, HabitatType, AxialCoord, PixelCoord, 
    Tile, PlayerBoard
)
from src.utils import get_logger, load_config, ensure_dir, get_output_dir
from src.phase1_preprocessing import PlayerBoardImage

logger = get_logger(__name__)


# ============================================================================
# VLM Client Abstraction
# ============================================================================

class VLMClient:
    """Base class for VLM API clients"""
    
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Send image to VLM and get response"""
        raise NotImplementedError


class OpenAIVisionClient(VLMClient):
    """OpenAI GPT-4 Vision client"""
    
    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # or "gpt-4-vision-preview"
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Send image to GPT-4 Vision"""
        base64_image = self._encode_image(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.1  # Low temperature for consistent results
        )
        
        return response.choices[0].message.content


class GeminiVisionClient(VLMClient):
    """Google Gemini Vision client"""
    
    def __init__(self, api_key: Optional[str] = None):
        import google.generativeai as genai
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Send image to Gemini Vision"""
        image = Image.open(image_path)
        
        response = self.model.generate_content(
            [prompt, image],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 4096
            }
        )
        
        return response.text


def get_vlm_client(provider: str = "openai") -> VLMClient:
    """Factory function to get VLM client"""
    if provider.lower() == "openai":
        return OpenAIVisionClient()
    elif provider.lower() in ["google", "gemini"]:
        return GeminiVisionClient()
    else:
        raise ValueError(f"Unknown VLM provider: {provider}")


# ============================================================================
# Detection Prompts
# ============================================================================

DETECTION_PROMPT = """You are analyzing a Cascadia board game player board image.

TASK: Identify ALL animal tokens and their positions on this hexagonal game board.

ANIMALS TO DETECT:
- Bear (brown/dark colored token)
- Elk (tan/beige colored token)  
- Salmon (pink/orange colored token)
- Hawk (blue/gray colored token)
- Fox (orange/red colored token)

INSTRUCTIONS:
1. Look at the hexagonal board carefully
2. Identify EVERY animal token you can see
3. Assign each token a position using a grid system

POSITION SYSTEM:
- Use row (r) and column (c) coordinates
- Row 0 is the TOP row of hexes
- Column 0 is the LEFTMOST hex in each row
- For hexagonal offset: odd rows are shifted right by half a hex

OUTPUT FORMAT (JSON only, no other text):
{
    "total_tokens": <number>,
    "tokens": [
        {
            "animal": "<bear|elk|salmon|hawk|fox>",
            "row": <row_number>,
            "col": <col_number>,
            "confidence": "<high|medium|low>",
            "notes": "<any observations>"
        }
    ],
    "board_dimensions": {
        "rows": <estimated_rows>,
        "cols": <estimated_cols>
    },
    "analysis_notes": "<any challenges or uncertainties>"
}

Be thorough - count EVERY token you can see. It's better to include uncertain detections with low confidence than to miss tokens.
"""

DETAILED_DETECTION_PROMPT = """You are a precise game state analyzer for the Cascadia board game.

TASK: Create a complete inventory of ALL animal tokens on this player's hexagonal board.

GAME CONTEXT:
- Cascadia uses hexagonal tiles arranged in a connected pattern
- Each tile MAY have an animal token placed on it (or be empty)
- Animal tokens are circular wooden pieces
- The 5 animal types are: Bear, Elk, Salmon, Hawk, Fox

TOKEN VISUAL IDENTIFICATION:
- BEAR: Brown/dark brown circular token, sometimes with bear silhouette
- ELK: Tan/beige/light brown circular token, sometimes with elk silhouette
- SALMON: Pink/salmon/orange-pink circular token, sometimes with fish silhouette
- HAWK: Blue/gray/slate circular token, sometimes with bird silhouette
- FOX: Orange/red-orange circular token, sometimes with fox silhouette

COORDINATE SYSTEM:
I need you to assign positions using axial coordinates (q, r):
- q increases going right
- r increases going down-right
- The center/reference hex should be (0, 0)
- Adjacent hexes differ by 1 in their coordinates

CRITICAL: Scan the ENTIRE board systematically:
1. Start from the top-left area
2. Move row by row
3. Count every single token
4. Note empty tiles too if relevant

OUTPUT (JSON only):
{
    "player_board_analysis": {
        "total_animal_tokens": <exact_count>,
        "tokens_by_type": {
            "bear": <count>,
            "elk": <count>,
            "salmon": <count>,
            "hawk": <count>,
            "fox": <count>
        },
        "detailed_tokens": [
            {
                "id": "<unique_id>",
                "animal": "<bear|elk|salmon|hawk|fox>",
                "q": <axial_q>,
                "r": <axial_r>,
                "confidence": <0.0-1.0>,
                "visual_description": "<brief description of what you see>"
            }
        ],
        "empty_tile_positions": [
            {"q": <q>, "r": <r>}
        ],
        "board_shape_notes": "<description of board shape/extent>",
        "detection_challenges": "<any issues encountered>"
    }
}

IMPORTANT: 
- Be EXHAUSTIVE - every token matters for scoring
- If unsure about a token type, include it with lower confidence
- Double-check your count matches the tokens_by_type totals
"""


HABITAT_DETECTION_PROMPT = """Analyze the habitat types on this Cascadia board.

HABITAT TYPES IN CASCADIA:
- Forest (green, trees)
- Mountain (gray/white, peaks)
- Prairie (yellow/tan, grass)
- Wetland (blue-green, water plants)
- River (blue, flowing water)

Note: Each hexagonal tile has ONE OR TWO habitat types painted on it.

For each tile position, identify the PRIMARY habitat type.

OUTPUT (JSON only):
{
    "habitats": [
        {
            "q": <axial_q>,
            "r": <axial_r>,
            "primary_habitat": "<forest|mountain|prairie|wetland|river>",
            "secondary_habitat": "<type or null>",
            "confidence": <0.0-1.0>
        }
    ]
}
"""


# ============================================================================
# Token Detector
# ============================================================================

class TokenDetector:
    """
    Detects animal tokens and habitats using VLM.
    """
    
    def __init__(self, vlm_provider: str = "openai", output_dir: Optional[Path] = None):
        self.vlm_client = get_vlm_client(vlm_provider)
        self.output_dir = output_dir or get_output_dir()
        ensure_dir(self.output_dir)
        self.detection_cache: Dict[str, Any] = {}
        
    def detect_tokens(self, player_board: PlayerBoardImage, 
                      use_detailed_prompt: bool = True) -> Dict:
        """
        Detect all animal tokens on a player's board.
        
        Args:
            player_board: PlayerBoardImage object
            use_detailed_prompt: Use more detailed prompt for better accuracy
            
        Returns:
            Dictionary with detection results
        """
        # Ensure we have a saved image to send to VLM
        if player_board.filepath is None:
            temp_path = self.output_dir / f"temp_player_{player_board.player_id}.png"
            cv2.imwrite(str(temp_path), player_board.image)
            image_path = str(temp_path)
        else:
            image_path = str(player_board.filepath)
        
        prompt = DETAILED_DETECTION_PROMPT if use_detailed_prompt else DETECTION_PROMPT
        
        logger.info(f"Sending Player {player_board.player_id} board to VLM for token detection...")
        
        try:
            response = self.vlm_client.analyze_image(image_path, prompt)
            result = self._parse_detection_response(response)
            
            # Cache result
            self.detection_cache[f"player_{player_board.player_id}"] = result
            
            # Save raw response for debugging
            self._save_raw_response(player_board.player_id, response, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Token detection failed: {e}")
            raise
    
    def detect_habitats(self, player_board: PlayerBoardImage) -> Dict:
        """
        Detect habitat types on tiles (optional, for bonus scoring).
        """
        if player_board.filepath is None:
            temp_path = self.output_dir / f"temp_player_{player_board.player_id}.png"
            cv2.imwrite(str(temp_path), player_board.image)
            image_path = str(temp_path)
        else:
            image_path = str(player_board.filepath)
        
        logger.info(f"Detecting habitats for Player {player_board.player_id}...")
        
        try:
            response = self.vlm_client.analyze_image(image_path, HABITAT_DETECTION_PROMPT)
            result = self._parse_detection_response(response)
            return result
        except Exception as e:
            logger.error(f"Habitat detection failed: {e}")
            return {"habitats": []}
    
    def _parse_detection_response(self, response: str) -> Dict:
        """Parse JSON from VLM response"""
        # Try to extract JSON from response
        try:
            # First, try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object pattern
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning("Could not parse JSON from VLM response, returning raw text")
        return {"raw_response": response, "parse_error": True}
    
    def _save_raw_response(self, player_id: int, raw_response: str, parsed: Dict):
        """Save VLM response for debugging"""
        debug_dir = self.output_dir / "debug" / "vlm_responses"
        ensure_dir(debug_dir)
        
        # Save raw response
        raw_path = debug_dir / f"player_{player_id}_raw.txt"
        with open(raw_path, 'w') as f:
            f.write(raw_response)
        
        # Save parsed JSON
        parsed_path = debug_dir / f"player_{player_id}_parsed.json"
        with open(parsed_path, 'w') as f:
            json.dump(parsed, f, indent=2)
        
        logger.debug(f"Saved VLM responses to {debug_dir}")
    
    def convert_to_player_board(self, player_id: int, detection_result: Dict) -> PlayerBoard:
        """
        Convert VLM detection results to PlayerBoard data model.
        """
        tiles = []
        
        # Handle the detailed detection format
        if "player_board_analysis" in detection_result:
            analysis = detection_result["player_board_analysis"]
            detailed_tokens = analysis.get("detailed_tokens", [])
            
            for token in detailed_tokens:
                tile = Tile(
                    id=token.get("id", f"tile_{len(tiles)}"),
                    animal=AnimalType(token["animal"].lower()),
                    habitat=HabitatType.UNKNOWN,  # Will be filled by habitat detection
                    coord=AxialCoord(
                        q=token.get("q", 0),
                        r=token.get("r", 0)
                    ),
                    confidence=token.get("confidence", 0.8)
                )
                tiles.append(tile)
        
        # Handle simpler detection format
        elif "tokens" in detection_result:
            for i, token in enumerate(detection_result["tokens"]):
                # Convert row/col to axial if needed
                row = token.get("row", token.get("r", 0))
                col = token.get("col", token.get("q", 0))
                q, r = self._rowcol_to_axial(row, col)
                
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                conf = token.get("confidence", "medium")
                if isinstance(conf, str):
                    conf = confidence_map.get(conf.lower(), 0.7)
                
                tile = Tile(
                    id=f"tile_{i}",
                    animal=AnimalType(token["animal"].lower()),
                    habitat=HabitatType.UNKNOWN,
                    coord=AxialCoord(q=q, r=r),
                    confidence=conf
                )
                tiles.append(tile)
        
        player_board = PlayerBoard(
            player_id=player_id,
            tiles=tiles
        )
        
        logger.info(f"Converted detection to PlayerBoard: {len(tiles)} tiles")
        return player_board
    
    def _rowcol_to_axial(self, row: int, col: int) -> Tuple[int, int]:
        """Convert row/column to axial coordinates"""
        # Offset coordinates to axial
        # For odd-r offset: q = col - (row - (row & 1)) / 2
        q = col - (row - (row & 1)) // 2
        r = row
        return q, r


# ============================================================================
# Visual Verification
# ============================================================================

class DetectionVisualizer:
    """Creates visual overlays to verify detection results"""
    
    # Colors for each animal type (BGR format)
    ANIMAL_COLORS = {
        AnimalType.BEAR: (51, 51, 139),      # Dark brown
        AnimalType.ELK: (153, 204, 255),      # Tan
        AnimalType.SALMON: (147, 112, 219),   # Pink
        AnimalType.HAWK: (173, 142, 102),     # Slate blue
        AnimalType.FOX: (0, 128, 255),        # Orange
        AnimalType.NONE: (128, 128, 128),     # Gray
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or get_output_dir()
        ensure_dir(self.output_dir)
    
    def create_detection_overlay(self, 
                                 player_board_image: PlayerBoardImage,
                                 player_board: PlayerBoard,
                                 hex_size: int = 50) -> np.ndarray:
        """
        Create an overlay image showing detected tokens.
        
        This helps verify that detections are correct.
        """
        image = player_board_image.image.copy()
        
        # We need to map axial coordinates to pixel positions
        # This is approximate - would need calibration for accuracy
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        for tile in player_board.tiles:
            # Convert axial to pixel (approximate)
            px, py = self._axial_to_pixel(tile.coord, hex_size, center_x, center_y)
            
            # Ensure within bounds
            px = max(hex_size, min(w - hex_size, px))
            py = max(hex_size, min(h - hex_size, py))
            
            # Draw circle for token
            color = self.ANIMAL_COLORS.get(tile.animal, (255, 255, 255))
            cv2.circle(image, (int(px), int(py)), hex_size // 2, color, 3)
            
            # Draw label
            label = tile.animal.value[0].upper()  # First letter
            cv2.putText(image, label, (int(px) - 10, int(py) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw coordinate
            coord_text = f"({tile.coord.q},{tile.coord.r})"
            cv2.putText(image, coord_text, (int(px) - 20, int(py) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image
    
    def _axial_to_pixel(self, coord: AxialCoord, size: int, 
                        center_x: int, center_y: int) -> Tuple[float, float]:
        """Convert axial coordinates to pixel position"""
        # Hex width and height
        hex_width = size * 2
        hex_height = size * np.sqrt(3)
        
        # Axial to pixel
        x = center_x + size * (3/2 * coord.q)
        y = center_y + size * (np.sqrt(3)/2 * coord.q + np.sqrt(3) * coord.r)
        
        return x, y
    
    def save_overlay(self, overlay: np.ndarray, player_id: int) -> Path:
        """Save overlay image"""
        overlay_dir = self.output_dir / "overlays"
        ensure_dir(overlay_dir)
        
        filepath = overlay_dir / f"player_{player_id}_detection_overlay.png"
        cv2.imwrite(str(filepath), overlay)
        
        logger.info(f"Saved detection overlay: {filepath}")
        return filepath
    
    def create_summary_image(self, player_board: PlayerBoard) -> np.ndarray:
        """Create a summary visualization of token counts"""
        # Create a simple summary image
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Title
        cv2.putText(img, f"Player {player_board.player_id} Token Summary",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Count tokens by type
        counts = {}
        for animal in AnimalType:
            if animal != AnimalType.NONE:
                counts[animal] = len(player_board.get_tiles_by_animal(animal))
        
        # Draw bars
        y = 60
        for animal, count in counts.items():
            color = self.ANIMAL_COLORS.get(animal, (128, 128, 128))
            
            # Animal name
            cv2.putText(img, f"{animal.value.capitalize()}: {count}",
                        (20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Bar
            bar_width = count * 20
            cv2.rectangle(img, (150, y), (150 + bar_width, y + 20), color, -1)
            
            y += 40
        
        # Total
        total = sum(counts.values())
        cv2.putText(img, f"Total: {total}", (20, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return img


# ============================================================================
# Main Detection Pipeline
# ============================================================================

def detect_all_players(player_boards: List[PlayerBoardImage],
                       vlm_provider: str = "openai",
                       output_dir: Optional[Path] = None,
                       create_overlays: bool = True) -> List[PlayerBoard]:
    """
    Run token detection for all players.
    
    Args:
        player_boards: List of PlayerBoardImage from Phase 1
        vlm_provider: "openai" or "google"
        output_dir: Output directory
        create_overlays: Whether to create visual verification overlays
        
    Returns:
        List of PlayerBoard objects with detected tokens
    """
    output_dir = output_dir or get_output_dir()
    detector = TokenDetector(vlm_provider, output_dir)
    visualizer = DetectionVisualizer(output_dir)
    
    results = []
    
    for player_board_img in player_boards:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Player {player_board_img.player_id}")
        logger.info(f"{'='*50}")
        
        # Detect tokens
        detection_result = detector.detect_tokens(player_board_img)
        
        # Convert to PlayerBoard model
        player_board = detector.convert_to_player_board(
            player_board_img.player_id,
            detection_result
        )
        player_board.source_image_path = str(player_board_img.filepath)
        
        # Create verification overlay
        if create_overlays:
            overlay = visualizer.create_detection_overlay(player_board_img, player_board)
            visualizer.save_overlay(overlay, player_board.player_id)
            
            summary = visualizer.create_summary_image(player_board)
            summary_path = output_dir / "overlays" / f"player_{player_board.player_id}_summary.png"
            cv2.imwrite(str(summary_path), summary)
        
        # Log summary
        logger.info(f"Player {player_board.player_id} detection complete:")
        for animal in AnimalType:
            if animal != AnimalType.NONE:
                count = len(player_board.get_tiles_by_animal(animal))
                if count > 0:
                    logger.info(f"  {animal.value}: {count}")
        
        results.append(player_board)
        
        # Small delay between API calls
        time.sleep(1)
    
    return results


# Example usage
if __name__ == "__main__":
    import sys
    from src.phase1_preprocessing import preprocess_game_image, load_separate_player_images
    
    print("Phase 2: Token Detection Module")
    print("="*50)
    print("\nThis module uses VLM to detect animal tokens on Cascadia boards.")
    print("\nUsage:")
    print("  1. First run Phase 1 to preprocess images")
    print("  2. Then call detect_all_players() with the results")
    print("\nExample:")
    print("  from phase1_preprocessing import preprocess_game_image")
    print("  from phase2_detection import detect_all_players")
    print("  ")
    print("  boards = preprocess_game_image('game_image.png')")
    print("  player_boards = detect_all_players(boards)")
