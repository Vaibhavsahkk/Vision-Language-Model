"""
Phase 1: Image Preprocessing & Player Segmentation

This module handles:
- Loading the final board image
- Rotating/aligning if needed
- Splitting into individual player boards
- Basic image enhancement

Design Decision:
- We keep preprocessing minimal to avoid losing information
- Player board segmentation can be manual, semi-automatic, or VLM-assisted
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from src.utils import get_logger, ensure_dir, get_output_dir

logger = get_logger(__name__)


@dataclass
class PlayerBoardImage:
    """Container for a player's board image with metadata"""
    player_id: int
    image: np.ndarray
    original_region: Tuple[int, int, int, int]  # x, y, w, h in original image
    filepath: Optional[Path] = None
    
    @property
    def height(self) -> int:
        return self.image.shape[0]
    
    @property
    def width(self) -> int:
        return self.image.shape[1]


class ImagePreprocessor:
    """
    Handles image loading, preprocessing, and player board segmentation.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or get_output_dir()
        ensure_dir(self.output_dir)
        self.debug_mode = True
        
    def load_image(self, image_path: str | Path) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array in BGR format (OpenCV standard)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
        return image
    
    def load_image_pil(self, image_path: str | Path) -> Image.Image:
        """Load image using PIL (useful for VLM APIs)"""
        image_path = Path(image_path)
        return Image.open(image_path)
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
        logger.info(f"Rotated image by {angle} degrees")
        return rotated
    
    def enhance_image(self, image: np.ndarray, 
                      brightness: float = 1.0,
                      contrast: float = 1.0,
                      sharpen: bool = False) -> np.ndarray:
        """
        Apply basic image enhancements.
        
        Args:
            image: Input image
            brightness: Brightness multiplier (1.0 = no change)
            contrast: Contrast multiplier (1.0 = no change)
            sharpen: Whether to apply sharpening
            
        Returns:
            Enhanced image
        """
        result = image.copy()
        
        # Apply brightness and contrast
        if brightness != 1.0 or contrast != 1.0:
            result = cv2.convertScaleAbs(result, alpha=contrast, beta=(brightness - 1) * 127)
        
        # Apply sharpening
        if sharpen:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            result = cv2.filter2D(result, -1, kernel)
        
        return result
    
    def auto_adjust_levels(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically adjust image levels for better contrast.
        Uses CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def segment_player_boards_manual(self, 
                                     image: np.ndarray,
                                     regions: List[Tuple[int, int, int, int]]) -> List[PlayerBoardImage]:
        """
        Manually segment player boards using provided regions.
        
        Args:
            image: Full board image
            regions: List of (x, y, width, height) tuples for each player
            
        Returns:
            List of PlayerBoardImage objects
        """
        player_boards = []
        
        for i, (x, y, w, h) in enumerate(regions, start=1):
            # Extract region
            board_img = image[y:y+h, x:x+w].copy()
            
            player_board = PlayerBoardImage(
                player_id=i,
                image=board_img,
                original_region=(x, y, w, h)
            )
            player_boards.append(player_board)
            logger.info(f"Extracted Player {i} board: {w}x{h} at ({x}, {y})")
        
        return player_boards
    
    def segment_player_boards_auto(self, 
                                   image: np.ndarray,
                                   num_players: int = 3,
                                   layout: str = "horizontal") -> List[PlayerBoardImage]:
        """
        Automatically segment player boards assuming equal division.
        
        Args:
            image: Full board image
            num_players: Number of players (default 3)
            layout: "horizontal" or "vertical" arrangement
            
        Returns:
            List of PlayerBoardImage objects
        """
        h, w = image.shape[:2]
        
        if layout == "horizontal":
            # Split horizontally (side by side)
            board_width = w // num_players
            regions = [(i * board_width, 0, board_width, h) for i in range(num_players)]
        else:
            # Split vertically (stacked)
            board_height = h // num_players
            regions = [(0, i * board_height, w, board_height) for i in range(num_players)]
        
        return self.segment_player_boards_manual(image, regions)
    
    def segment_single_board(self, image: np.ndarray, player_id: int = 1) -> PlayerBoardImage:
        """
        Treat the entire image as a single player's board.
        Use this when each player's board is already a separate image.
        """
        h, w = image.shape[:2]
        return PlayerBoardImage(
            player_id=player_id,
            image=image.copy(),
            original_region=(0, 0, w, h)
        )
    
    def crop_to_content(self, image: np.ndarray, padding: int = 10) -> np.ndarray:
        """
        Crop image to content area (remove excessive borders).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find content
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get bounding box of all contours
        x_min, y_min = image.shape[1], image.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        return image[y_min:y_max, x_min:x_max]
    
    def save_player_board(self, player_board: PlayerBoardImage, 
                          output_dir: Optional[Path] = None,
                          prefix: str = "player") -> Path:
        """
        Save a player board image to file.
        """
        output_dir = output_dir or self.output_dir
        ensure_dir(output_dir)
        
        filename = f"{prefix}_{player_board.player_id}_board.png"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), player_board.image)
        player_board.filepath = filepath
        
        logger.info(f"Saved Player {player_board.player_id} board to: {filepath}")
        return filepath
    
    def save_debug_image(self, image: np.ndarray, name: str) -> Path:
        """Save a debug image"""
        if not self.debug_mode:
            return None
        
        debug_dir = self.output_dir / "debug"
        ensure_dir(debug_dir)
        
        filepath = debug_dir / f"{name}.png"
        cv2.imwrite(str(filepath), image)
        logger.debug(f"Saved debug image: {filepath}")
        return filepath
    
    def visualize_segmentation(self, image: np.ndarray, 
                               regions: List[Tuple[int, int, int, int]],
                               save: bool = True) -> np.ndarray:
        """
        Draw rectangles showing player board regions on the image.
        """
        vis_image = image.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
        
        for i, (x, y, w, h) in enumerate(regions):
            color = colors[i % len(colors)]
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(vis_image, f"Player {i + 1}", (x + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        if save:
            self.save_debug_image(vis_image, "segmentation_regions")
        
        return vis_image


def preprocess_game_image(image_path: str | Path,
                          player_regions: Optional[List[Tuple[int, int, int, int]]] = None,
                          num_players: int = 3,
                          layout: str = "horizontal",
                          enhance: bool = False,
                          output_dir: Optional[Path] = None) -> List[PlayerBoardImage]:
    """
    Main preprocessing function - loads image and segments into player boards.
    
    Args:
        image_path: Path to the full game board image
        player_regions: Manual regions as (x, y, w, h) tuples. If None, uses auto-segmentation.
        num_players: Number of players (used for auto-segmentation)
        layout: "horizontal" or "vertical" (used for auto-segmentation)
        enhance: Whether to apply auto-enhancement
        output_dir: Directory to save output images
        
    Returns:
        List of PlayerBoardImage objects
    """
    preprocessor = ImagePreprocessor(output_dir)
    
    # Load image
    image = preprocessor.load_image(image_path)
    
    # Enhance if requested
    if enhance:
        image = preprocessor.auto_adjust_levels(image)
    
    # Segment into player boards
    if player_regions:
        player_boards = preprocessor.segment_player_boards_manual(image, player_regions)
        preprocessor.visualize_segmentation(image, player_regions)
    else:
        player_boards = preprocessor.segment_player_boards_auto(image, num_players, layout)
        # Calculate regions for visualization
        regions = [pb.original_region for pb in player_boards]
        preprocessor.visualize_segmentation(image, regions)
    
    # Save individual boards
    for pb in player_boards:
        preprocessor.save_player_board(pb)
    
    return player_boards


def load_separate_player_images(image_paths: List[str | Path],
                                output_dir: Optional[Path] = None) -> List[PlayerBoardImage]:
    """
    Load player boards from separate image files.
    Use this when each player's board is already a separate image.
    
    Args:
        image_paths: List of paths to player board images (one per player)
        output_dir: Directory to save processed images
        
    Returns:
        List of PlayerBoardImage objects
    """
    preprocessor = ImagePreprocessor(output_dir)
    player_boards = []
    
    for i, path in enumerate(image_paths, start=1):
        image = preprocessor.load_image(path)
        player_board = preprocessor.segment_single_board(image, player_id=i)
        player_board.filepath = Path(path)
        player_boards.append(player_board)
    
    return player_boards


# Example usage for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Try auto-segmentation first
        boards = preprocess_game_image(
            image_path,
            num_players=3,
            layout="horizontal",
            enhance=True
        )
        
        print(f"\nProcessed {len(boards)} player boards:")
        for board in boards:
            print(f"  Player {board.player_id}: {board.width}x{board.height}")
            if board.filepath:
                print(f"    Saved to: {board.filepath}")
    else:
        print("Usage: python phase1_preprocessing.py <image_path>")
        print("\nThis module provides image preprocessing for Cascadia board game scoring.")
