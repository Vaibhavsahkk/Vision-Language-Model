"""
Phase 3: Graph Construction

This module handles:
- Building a hexagonal grid graph per player
- Nodes = tiles with animal/habitat metadata
- Edges = adjacent hexes (6-directional)

Key Design:
- Uses NetworkX for graph operations
- Axial coordinate system for clean neighbor calculation
- Graph enables all scoring algorithms in Phase 4
"""

import networkx as nx
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.models import (
    AnimalType, HabitatType, AxialCoord, Tile, PlayerBoard
)
from src.utils import get_logger, ensure_dir, get_output_dir

logger = get_logger(__name__)


# ============================================================================
# Hexagonal Grid Utilities
# ============================================================================

class HexGrid:
    """
    Utilities for hexagonal grid operations using axial coordinates.
    
    Axial Coordinate System:
    - Uses (q, r) coordinates
    - 6 neighbors for each hex
    - Efficient distance and neighbor calculations
    
    Neighbor Directions (axial):
        (+1, 0), (+1, -1), (0, -1),
        (-1, 0), (-1, +1), (0, +1)
    """
    
    # The 6 axial direction vectors for hex neighbors
    DIRECTIONS = [
        (1, 0),   # East
        (1, -1),  # Northeast
        (0, -1),  # Northwest
        (-1, 0),  # West
        (-1, 1),  # Southwest
        (0, 1),   # Southeast
    ]
    
    @staticmethod
    def get_neighbors(coord: AxialCoord) -> List[AxialCoord]:
        """Get all 6 neighboring coordinates"""
        return [
            AxialCoord(coord.q + dq, coord.r + dr)
            for dq, dr in HexGrid.DIRECTIONS
        ]
    
    @staticmethod
    def distance(a: AxialCoord, b: AxialCoord) -> int:
        """Calculate hex distance between two coordinates"""
        return (abs(a.q - b.q) 
                + abs(a.q + a.r - b.q - b.r) 
                + abs(a.r - b.r)) // 2
    
    @staticmethod
    def are_adjacent(a: AxialCoord, b: AxialCoord) -> bool:
        """Check if two coordinates are adjacent"""
        return HexGrid.distance(a, b) == 1
    
    @staticmethod
    def axial_to_cube(coord: AxialCoord) -> Tuple[int, int, int]:
        """Convert axial to cube coordinates"""
        x = coord.q
        z = coord.r
        y = -x - z
        return (x, y, z)
    
    @staticmethod
    def cube_to_axial(x: int, y: int, z: int) -> AxialCoord:
        """Convert cube to axial coordinates"""
        return AxialCoord(x, z)
    
    @staticmethod
    def axial_to_pixel(coord: AxialCoord, size: float) -> Tuple[float, float]:
        """Convert axial coordinates to pixel position for visualization"""
        x = size * (3/2 * coord.q)
        y = size * (np.sqrt(3)/2 * coord.q + np.sqrt(3) * coord.r)
        return (x, y)


# ============================================================================
# Board Graph
# ============================================================================

class BoardGraph:
    """
    Represents a player's board as a graph.
    
    - Nodes: Tiles (with animal and habitat data)
    - Edges: Adjacent tiles (automatically computed)
    
    This structure enables efficient scoring algorithms:
    - Finding connected components (bears, elk herds)
    - Finding paths/chains (salmon runs)
    - Counting isolated nodes (hawks)
    - Checking neighbors (foxes)
    """
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.graph = nx.Graph()
        self.coord_to_tile: Dict[Tuple[int, int], Tile] = {}
        
    def add_tile(self, tile: Tile):
        """Add a tile as a node in the graph"""
        coord_key = (tile.coord.q, tile.coord.r)
        
        # Add node with attributes
        self.graph.add_node(
            coord_key,
            tile=tile,
            animal=tile.animal,
            habitat=tile.habitat,
            confidence=tile.confidence
        )
        
        self.coord_to_tile[coord_key] = tile
        
        # Connect to existing adjacent tiles
        for neighbor_coord in HexGrid.get_neighbors(tile.coord):
            neighbor_key = (neighbor_coord.q, neighbor_coord.r)
            if neighbor_key in self.coord_to_tile:
                self.graph.add_edge(coord_key, neighbor_key)
    
    def build_from_player_board(self, player_board: PlayerBoard):
        """Build graph from a PlayerBoard object"""
        logger.info(f"Building graph for Player {player_board.player_id}...")
        
        for tile in player_board.tiles:
            self.add_tile(tile)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges")
    
    def get_tile(self, q: int, r: int) -> Optional[Tile]:
        """Get tile at coordinate"""
        return self.coord_to_tile.get((q, r))
    
    def get_tiles_by_animal(self, animal: AnimalType) -> List[Tile]:
        """Get all tiles with a specific animal"""
        tiles = []
        for coord_key, data in self.graph.nodes(data=True):
            if data.get('animal') == animal:
                tiles.append(data['tile'])
        return tiles
    
    def get_animal_subgraph(self, animal: AnimalType) -> nx.Graph:
        """Get subgraph containing only tiles with specific animal"""
        animal_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('animal') == animal
        ]
        return self.graph.subgraph(animal_nodes).copy()
    
    def get_neighbors(self, tile: Tile) -> List[Tile]:
        """Get neighboring tiles"""
        coord_key = (tile.coord.q, tile.coord.r)
        neighbors = []
        
        for neighbor_key in self.graph.neighbors(coord_key):
            if neighbor_key in self.coord_to_tile:
                neighbors.append(self.coord_to_tile[neighbor_key])
        
        return neighbors
    
    def get_adjacent_animals(self, tile: Tile) -> Set[AnimalType]:
        """Get set of unique animal types adjacent to a tile"""
        neighbors = self.get_neighbors(tile)
        return {n.animal for n in neighbors if n.animal != AnimalType.NONE}
    
    def find_connected_components(self, animal: AnimalType) -> List[List[Tile]]:
        """
        Find connected groups of tiles with the same animal.
        Returns list of tile groups (each group is a list of connected tiles).
        """
        subgraph = self.get_animal_subgraph(animal)
        components = []
        
        for component_nodes in nx.connected_components(subgraph):
            tiles = [self.coord_to_tile[node] for node in component_nodes]
            components.append(tiles)
        
        return components
    
    def find_chains(self, animal: AnimalType) -> List[List[Tile]]:
        """
        Find non-branching chains (paths) of tiles with the same animal.
        Used for salmon scoring.
        
        A chain has no branching - each internal node has exactly 2 neighbors
        of the same animal type.
        """
        subgraph = self.get_animal_subgraph(animal)
        chains = []
        visited = set()
        
        for component in nx.connected_components(subgraph):
            component_subgraph = subgraph.subgraph(component)
            
            # Check if this component is a simple path (chain) or has branches
            degrees = dict(component_subgraph.degree())
            
            # Find endpoints (degree 1) and branch points (degree > 2)
            endpoints = [n for n, d in degrees.items() if d == 1]
            branch_points = [n for n, d in degrees.items() if d > 2]
            
            if not branch_points:
                # No branches - this is a valid chain
                if len(component) == 1:
                    # Single node
                    chain_tiles = [self.coord_to_tile[n] for n in component]
                    chains.append(chain_tiles)
                elif len(endpoints) == 2:
                    # Simple path from one endpoint to another
                    path = nx.shortest_path(component_subgraph, endpoints[0], endpoints[1])
                    chain_tiles = [self.coord_to_tile[n] for n in path]
                    chains.append(chain_tiles)
                elif len(endpoints) == 0:
                    # It's a cycle - still counts as a valid formation
                    chain_tiles = [self.coord_to_tile[n] for n in component]
                    chains.append(chain_tiles)
            else:
                # Has branches - need to extract individual chain segments
                # Find the longest path without branching
                chain_tiles = self._extract_longest_chain(component_subgraph)
                if chain_tiles:
                    chains.append(chain_tiles)
        
        return chains
    
    def _extract_longest_chain(self, subgraph: nx.Graph) -> List[Tile]:
        """Extract the longest non-branching chain from a subgraph with branches"""
        if subgraph.number_of_nodes() == 0:
            return []
        
        # Find all simple paths and return the longest
        longest_path = []
        nodes = list(subgraph.nodes())
        
        # For small graphs, try all pairs
        if len(nodes) <= 20:
            for i, start in enumerate(nodes):
                for end in nodes[i+1:]:
                    try:
                        for path in nx.all_simple_paths(subgraph, start, end):
                            if len(path) > len(longest_path):
                                longest_path = path
                    except nx.NetworkXNoPath:
                        continue
        else:
            # For larger graphs, use approximation
            # Start from low-degree nodes
            degrees = dict(subgraph.degree())
            start_nodes = sorted(degrees.keys(), key=lambda x: degrees[x])[:5]
            
            for start in start_nodes:
                # BFS to find longest path from this start
                visited = {start}
                current_path = [start]
                
                while True:
                    current = current_path[-1]
                    next_nodes = [n for n in subgraph.neighbors(current) 
                                  if n not in visited]
                    if not next_nodes:
                        break
                    # Choose neighbor with lowest degree (tends to extend paths)
                    next_node = min(next_nodes, key=lambda x: degrees[x])
                    current_path.append(next_node)
                    visited.add(next_node)
                
                if len(current_path) > len(longest_path):
                    longest_path = current_path
        
        return [self.coord_to_tile[n] for n in longest_path]
    
    def find_isolated_tiles(self, animal: AnimalType) -> List[Tile]:
        """
        Find tiles that have no adjacent tiles with the same animal.
        Used for hawk scoring.
        """
        animal_tiles = self.get_tiles_by_animal(animal)
        isolated = []
        
        for tile in animal_tiles:
            neighbors = self.get_neighbors(tile)
            same_animal_neighbors = [n for n in neighbors if n.animal == animal]
            
            if len(same_animal_neighbors) == 0:
                isolated.append(tile)
        
        return isolated
    
    def count_unique_adjacent_animals(self, tile: Tile) -> int:
        """
        Count unique animal types adjacent to a tile.
        Used for fox scoring.
        """
        adjacent_animals = self.get_adjacent_animals(tile)
        # Remove the tile's own animal type and NONE
        adjacent_animals.discard(tile.animal)
        adjacent_animals.discard(AnimalType.NONE)
        return len(adjacent_animals)


# ============================================================================
# Graph Visualization
# ============================================================================

class GraphVisualizer:
    """Visualize board graphs for debugging and verification"""
    
    ANIMAL_COLORS = {
        AnimalType.BEAR: '#8B4513',    # Brown
        AnimalType.ELK: '#DEB887',      # Tan
        AnimalType.SALMON: '#FA8072',   # Salmon
        AnimalType.HAWK: '#4682B4',     # Steel blue
        AnimalType.FOX: '#FF6347',      # Tomato
        AnimalType.NONE: '#D3D3D3',     # Light gray
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or get_output_dir()
        ensure_dir(self.output_dir)
    
    def visualize_board_graph(self, board_graph: BoardGraph, 
                              title: Optional[str] = None,
                              save: bool = True,
                              show: bool = False) -> Optional[Path]:
        """Create a visual representation of the board graph"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Calculate positions using hex coordinates
        pos = {}
        for node in board_graph.graph.nodes():
            q, r = node
            x, y = HexGrid.axial_to_pixel(AxialCoord(q, r), size=1.0)
            pos[node] = (x, -y)  # Flip y for display
        
        # Get node colors based on animal type
        node_colors = []
        for node in board_graph.graph.nodes():
            data = board_graph.graph.nodes[node]
            animal = data.get('animal', AnimalType.NONE)
            node_colors.append(self.ANIMAL_COLORS.get(animal, '#D3D3D3'))
        
        # Draw the graph
        nx.draw(
            board_graph.graph,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=500,
            edge_color='#888888',
            width=2,
            with_labels=False
        )
        
        # Add labels with animal first letter
        labels = {}
        for node in board_graph.graph.nodes():
            data = board_graph.graph.nodes[node]
            animal = data.get('animal', AnimalType.NONE)
            if animal != AnimalType.NONE:
                labels[node] = animal.value[0].upper()
        
        nx.draw_networkx_labels(
            board_graph.graph,
            pos,
            labels,
            ax=ax,
            font_size=10,
            font_weight='bold'
        )
        
        # Title
        title = title or f"Player {board_graph.player_id} Board Graph"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        legend_elements = []
        for animal, color in self.ANIMAL_COLORS.items():
            if animal != AnimalType.NONE:
                from matplotlib.patches import Patch
                legend_elements.append(
                    Patch(facecolor=color, label=animal.value.capitalize())
                )
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            graph_dir = self.output_dir / "graphs"
            ensure_dir(graph_dir)
            filepath = graph_dir / f"player_{board_graph.player_id}_graph.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved graph visualization: {filepath}")
            
            if not show:
                plt.close()
            
            return filepath
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return None
    
    def visualize_animal_subgraph(self, board_graph: BoardGraph, 
                                   animal: AnimalType,
                                   highlight_groups: bool = True) -> Optional[Path]:
        """Visualize subgraph for a specific animal type"""
        subgraph = board_graph.get_animal_subgraph(animal)
        
        if subgraph.number_of_nodes() == 0:
            logger.warning(f"No {animal.value} tokens found")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Calculate positions
        pos = {}
        for node in subgraph.nodes():
            q, r = node
            x, y = HexGrid.axial_to_pixel(AxialCoord(q, r), size=1.0)
            pos[node] = (x, -y)
        
        # Color by connected component if highlighting groups
        if highlight_groups:
            import matplotlib.cm as cm
            components = list(nx.connected_components(subgraph))
            colors = cm.tab10(np.linspace(0, 1, max(len(components), 1)))
            
            node_colors = {}
            for i, component in enumerate(components):
                for node in component:
                    node_colors[node] = colors[i]
            
            color_list = [node_colors[n] for n in subgraph.nodes()]
        else:
            color_list = self.ANIMAL_COLORS[animal]
        
        # Draw
        nx.draw(
            subgraph,
            pos,
            ax=ax,
            node_color=color_list,
            node_size=700,
            edge_color='#444444',
            width=3,
            with_labels=False
        )
        
        # Add coordinate labels
        labels = {node: f"({node[0]},{node[1]})" for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, ax=ax, font_size=8)
        
        ax.set_title(f"Player {board_graph.player_id} - {animal.value.capitalize()} Tokens",
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        graph_dir = self.output_dir / "graphs"
        ensure_dir(graph_dir)
        filepath = graph_dir / f"player_{board_graph.player_id}_{animal.value}_subgraph.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {animal.value} subgraph: {filepath}")
        return filepath


# ============================================================================
# Main Graph Building Pipeline
# ============================================================================

def build_player_graphs(player_boards: List[PlayerBoard],
                        output_dir: Optional[Path] = None,
                        visualize: bool = True) -> Dict[int, BoardGraph]:
    """
    Build graphs for all players.
    
    Args:
        player_boards: List of PlayerBoard objects from Phase 2
        output_dir: Output directory for visualizations
        visualize: Whether to create graph visualizations
        
    Returns:
        Dictionary mapping player_id to BoardGraph
    """
    output_dir = output_dir or get_output_dir()
    visualizer = GraphVisualizer(output_dir) if visualize else None
    
    graphs = {}
    
    for player_board in player_boards:
        logger.info(f"\n{'='*50}")
        logger.info(f"Building graph for Player {player_board.player_id}")
        logger.info(f"{'='*50}")
        
        # Create and build graph
        board_graph = BoardGraph(player_board.player_id)
        board_graph.build_from_player_board(player_board)
        
        # Log graph statistics
        logger.info(f"Graph statistics:")
        logger.info(f"  Nodes (tiles): {board_graph.graph.number_of_nodes()}")
        logger.info(f"  Edges (adjacencies): {board_graph.graph.number_of_edges()}")
        
        # Log animal distribution
        for animal in AnimalType:
            if animal != AnimalType.NONE:
                tiles = board_graph.get_tiles_by_animal(animal)
                if tiles:
                    components = board_graph.find_connected_components(animal)
                    logger.info(f"  {animal.value}: {len(tiles)} tiles in {len(components)} groups")
        
        # Visualize
        if visualizer:
            visualizer.visualize_board_graph(board_graph)
        
        graphs[player_board.player_id] = board_graph
    
    return graphs


# Example usage
if __name__ == "__main__":
    print("Phase 3: Graph Construction Module")
    print("="*50)
    print("\nThis module builds hexagonal grid graphs from detected tiles.")
    print("\nKey features:")
    print("  - Axial coordinate system for hex grids")
    print("  - NetworkX-based graph operations")
    print("  - Connected component detection")
    print("  - Path/chain finding")
    print("  - Neighbor analysis")
    print("\nUsage:")
    print("  from phase3_graph import build_player_graphs")
    print("  graphs = build_player_graphs(player_boards)")
