import glob
import json
import random

import elements
import embodied
import numpy as np


class ARC(embodied.Env):
    
    # ARC color palette (RGB values for colors 0-9)
    COLOR_MAP = {
        0: [0, 0, 0],           # Black
        1: [0, 116, 217],       # Blue
        2: [255, 65, 54],       # Red
        3: [46, 204, 64],       # Green
        4: [255, 220, 0],       # Yellow
        5: [170, 170, 170],     # Gray
        6: [240, 18, 190],      # Magenta
        7: [255, 133, 27],      # Orange
        8: [127, 219, 255],     # Light Blue
        9: [135, 12, 37],       # Maroon
    }
    
    def __init__(self, task, puzzle_dir='./arc-data/', version='V2', split='training', length=100, size=64):
        """
        Args:
            task: Not used, but required by interface
            puzzle_dir: Path to arc-data directory (default: './arc-data/')
            version: 'V1' or 'V2' (default: 'V2')
            split: 'training' or 'evaluation' (default: 'training')
            length: Maximum steps per episode
            size: Size to pad grids to (default 64×64)
        """
        self.puzzle_dir = puzzle_dir
        self.version = version
        self.split = split
        self.length = length
        self.size = size
        
        # Construct the full path to puzzles
        self.full_puzzle_path = f"{puzzle_dir}/{version}/data/{split}"
        self.puzzles = self._load_puzzles()
        
        if len(self.puzzles) == 0:
            raise ValueError(f"No puzzles found in {self.full_puzzle_path}. Please check the path.")
        
        print(f"Loaded {len(self.puzzles)} ARC puzzles from {self.full_puzzle_path} ({version}/{split})")
        
        # Current episode state
        self.current_puzzle = None
        self.train_inputs = []
        self.train_outputs = []
        self.test_input = None
        self.test_output = None
        self.current_output = None
        self.step_count = 0
    
    def _load_puzzles(self):
        """Load ARC JSON files from the specified version and split directory."""
        puzzles = []
        pattern = f'{self.full_puzzle_path}/*.json'
        
        for filepath in glob.glob(pattern):
            try:
                with open(filepath) as f:
                    puzzle = json.load(f)
                    # Validate puzzle has required structure
                    if 'train' in puzzle and 'test' in puzzle:
                        puzzles.append(puzzle)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        
        return puzzles
    
    @property
    def obs_space(self):
        """Define observation space with paired images."""
        pair_shape = (self.size, self.size * 2, 3)  # input|output side-by-side
        
        return {
            'pair_1': elements.Space(np.uint8, pair_shape),
            'pair_2': elements.Space(np.uint8, pair_shape),
            'pair_3': elements.Space(np.uint8, pair_shape),
            'pair_4': elements.Space(np.uint8, pair_shape),
            'test_pair': elements.Space(np.uint8, pair_shape),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
    
    @property
    def act_space(self):
        """Define action space for grid editing."""
        return {
            'action_type': elements.Space(np.int32, (), 0, 3),  # 0:paint, 1:copy, 2:done
            'x': elements.Space(np.int32, (), 0, 29),
            'y': elements.Space(np.int32, (), 0, 29),
            'color': elements.Space(np.int32, (), 0, 9),
            'reset': elements.Space(bool),
        }
    
    def step(self, action):
        """Execute action and return observation."""
        # Handle reset
        if action['reset'] or self.current_puzzle is None:
            return self._reset()
        
        # Execute action on the grid
        self._execute_action(action)
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        is_done = (
            action['action_type'] == 2 or  # "done" action
            self.step_count >= self.length
        )
        
        # Generate observation
        obs = self._get_observation()
        obs['reward'] = np.float32(reward)
        obs['is_first'] = False
        obs['is_last'] = is_done
        obs['is_terminal'] = is_done
        
        return obs
    
    def _reset(self):
        """Start a new episode with a random puzzle."""
        # Pick random puzzle
        self.current_puzzle = random.choice(self.puzzles)
        
        # Extract training examples
        train = self.current_puzzle['train']
        self.train_inputs = [np.array(ex['input'], dtype=np.uint8) for ex in train]
        self.train_outputs = [np.array(ex['output'], dtype=np.uint8) for ex in train]
        
        # Pad to always have 4 examples (repeat last if fewer)
        while len(self.train_inputs) < 4:
            idx = len(self.train_inputs) - 1
            self.train_inputs.append(self.train_inputs[idx].copy())
            self.train_outputs.append(self.train_outputs[idx].copy())
        
        # Take only first 4 if more
        self.train_inputs = self.train_inputs[:4]
        self.train_outputs = self.train_outputs[:4]
        
        # Get test case (use first test example)
        test = self.current_puzzle['test'][0]
        self.test_input = np.array(test['input'], dtype=np.uint8)
        self.test_output = np.array(test['output'], dtype=np.uint8)
        
        # Initialize blank working grid (same size as test output)
        self.current_output = np.zeros_like(self.test_output)
        self.step_count = 0
        
        # Return initial observation
        obs = self._get_observation()
        obs['reward'] = np.float32(0)
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        
        return obs
    
    def _execute_action(self, action):
        """Modify the current_output grid based on action."""
        action_type = action['action_type']
        x, y = action['x'], action['y']
        
        # Bounds check
        h, w = self.current_output.shape
        if not (0 <= x < h and 0 <= y < w):
            return  # Invalid position, ignore
        
        if action_type == 0:  # Paint
            color = action['color']
            self.current_output[x, y] = color
        
        elif action_type == 1:  # Copy from input
            if x < self.test_input.shape[0] and y < self.test_input.shape[1]:
                self.current_output[x, y] = self.test_input[x, y]
        
        # action_type == 2 is "done", no grid modification
    
    def _calculate_reward(self):
        """Reward based on similarity to ground truth."""
        # Reward: percentage of correct cells
        correct = (self.current_output == self.test_output)
        accuracy = correct.sum() / self.test_output.size
        
        return float(accuracy)
    
    def _get_observation(self):
        """Generate observation dict with all paired images."""
        obs = {}
        
        # Create training pairs
        for i in range(4):
            pair = self._make_pair(self.train_inputs[i], self.train_outputs[i])
            obs[f'pair_{i+1}'] = pair
        
        # Create test pair (input | current work)
        test_pair = self._make_pair(self.test_input, self.current_output)
        obs['test_pair'] = test_pair
        
        return obs
    
    def _make_pair(self, input_grid, output_grid):
        """Convert two grids into a single image (input|output side-by-side)."""
        # Pad and colorize each grid
        inp_rgb = self._pad_and_colorize(input_grid)  # size×size×3
        out_rgb = self._pad_and_colorize(output_grid)  # size×size×3
        
        # Concatenate horizontally
        pair = np.concatenate([inp_rgb, out_rgb], axis=1)  # size×(size*2)×3
        
        return pair
    
    def _pad_and_colorize(self, grid):
        """Pad grid to size×size and convert to RGB."""
        h, w = grid.shape
        
        # Calculate padding
        pad_h = (self.size - h) // 2
        pad_w = (self.size - w) // 2
        
        # Ensure we don't exceed size
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        
        # Pad
        padded = np.pad(
            grid,
            ((pad_h, self.size - h - pad_h), (pad_w, self.size - w - pad_w)),
            mode='constant',
            constant_values=0
        )
        
        # Ensure exactly size×size (crop if grid was too large)
        padded = padded[:self.size, :self.size]
        
        # Convert to RGB
        rgb = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for color_idx, rgb_val in self.COLOR_MAP.items():
            mask = padded == color_idx
            rgb[mask] = rgb_val
        
        return rgb