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
        self.num_valid_pairs = 0
        
        # Store last completed episode for visualization
        self.last_episode_data = None
        
        # Track actions during episode
        self.action_history = []
        
        # Track which actions have been used to prevent repeats
        self.has_copied = False
        self.has_resized = False
        self.painted_positions = set()  # Set of (x, y) tuples
    
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
            'pair_5': elements.Space(np.uint8, pair_shape),
            'test_pair': elements.Space(np.uint8, pair_shape),
            'num_valid_pairs': elements.Space(np.int32, (), 0, 5),  # How many of pair_1-5 are real (0-5)
            'can_copy': elements.Space(bool),  # Whether copy action is still available
            'can_resize': elements.Space(bool),  # Whether resize action is still available
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
    
    @property
    def act_space(self):
        """Define action space for grid editing."""
        return {
            'action_type': elements.Space(np.int32, (), 0, 4),  # 0:paint, 1:copy, 2:resize, 3:done
            'x': elements.Space(np.int32, (), 0, 29),
            'y': elements.Space(np.int32, (), 0, 29),
            'color': elements.Space(np.int32, (), 0, 9),
            'width': elements.Space(np.int32, (), 0, 30),   # Target width for resize
            'height': elements.Space(np.int32, (), 0, 30),  # Target height for resize
            'reset': elements.Space(bool),
        }
    
    def step(self, action):
        """Execute action and return observation."""
        # Handle reset
        if action['reset'] or self.current_puzzle is None:
            return self._reset()
        
        # Store action in history before executing
        action_record = {
            'step': self.step_count,
            'action_type': int(action['action_type']),
            'x': int(action['x']),
            'y': int(action['y']),
            'color': int(action['color']),
            'width': int(action['width']),
            'height': int(action['height']),
        }
        self.action_history.append(action_record)
        
        # Execute action on the grid
        self._execute_action(action)
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        is_done = (
            action['action_type'] == 3 or  # "done" action
            self.step_count >= self.length
        )
        
        # Save episode data when episode ends
        if is_done:
            self._save_episode_data(reward)
        
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
        
        # Store the actual number of examples (before padding)
        self.num_valid_pairs = min(len(self.train_inputs), 5)
        
        # Zero-pad to always have 5 examples
        while len(self.train_inputs) < 5:
            # Create blank grids (use first example's shape as reference)
            blank = np.zeros_like(self.train_inputs[0])
            self.train_inputs.append(blank)
            self.train_outputs.append(blank)
        
        # Take only first 5 if more
        self.train_inputs = self.train_inputs[:5]
        self.train_outputs = self.train_outputs[:5]
        
        # Get test case (use first test example)
        test = self.current_puzzle['test'][0]
        self.test_input = np.array(test['input'], dtype=np.uint8)
        self.test_output = np.array(test['output'], dtype=np.uint8)
        
        # Initialize blank 3x3 working grid
        self.current_output = np.zeros((3, 3), dtype=np.uint8)
        self.step_count = 0
        
        # Clear action history for new episode
        self.action_history = []
        
        # Reset action tracking to allow all actions in new episode
        self.has_copied = False
        self.has_resized = False
        self.painted_positions = set()
        
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
        x, y = int(action['x']), int(action['y'])
        
        if action_type == 0:  # Paint
            h, w = self.current_output.shape
            if 0 <= x < h and 0 <= y < w:
                # Check if this position has already been painted
                if (x, y) in self.painted_positions:
                    # Skip this action - position already painted
                    return
                
                # Paint the cell and mark it as painted
                color = action['color']
                self.current_output[x, y] = color
                self.painted_positions.add((x, y))
        
        elif action_type == 1:  # Copy entire input
            # Check if copy has already been used
            if self.has_copied:
                # Skip this action - already copied once
                return
            
            # Perform copy and mark as used
            self.current_output = self.test_input.copy()
            self.has_copied = True
        
        elif action_type == 2:  # Resize
            # Check if resize has already been used
            if self.has_resized:
                # Skip this action - already resized once
                return
            
            # Clip to valid range [1, 30], treating 0 as 1
            new_height = int(np.clip(action['height'], 1, 30))
            new_width = int(np.clip(action['width'], 1, 30))
            
            # Create new grid
            new_grid = np.zeros((new_height, new_width), dtype=np.uint8)
            
            # Copy existing content where it fits (preserve content)
            old_h, old_w = self.current_output.shape
            copy_h = min(old_h, new_height)
            copy_w = min(old_w, new_width)
            new_grid[:copy_h, :copy_w] = self.current_output[:copy_h, :copy_w]
            
            self.current_output = new_grid
            self.has_resized = True
            
            # Note: When resizing, we clear the painted_positions set since
            # the grid dimensions have changed and positions may no longer be valid.
            # This allows repainting at the same coordinates in the new grid size.
            self.painted_positions.clear()
        
        # action_type == 3 is "done", no grid modification
    
    def _calculate_reward(self):
        """
        Reward based on similarity to ground truth and size matching.
        
        Reward components:
        1. Size accuracy: Reward for getting closer to correct dimensions
        2. Content accuracy: Percentage of correct cells (in overlapping region)
        3. Bonus: Extra reward for exact size match
        """
        target_h, target_w = self.test_output.shape
        current_h, current_w = self.current_output.shape
        
        # Component 1: Size accuracy (0 to 0.3)
        # Distance from correct size (Manhattan distance normalized)
        h_diff = abs(current_h - target_h)
        w_diff = abs(current_w - target_w)
        max_diff = 30 + 30  # Maximum possible difference
        size_distance = (h_diff + w_diff) / max_diff
        size_accuracy = (1.0 - size_distance) * 0.3
        
        # Component 2: Content accuracy (0 to 0.6)
        # Calculate accuracy on the overlapping region
        min_h = min(current_h, target_h)
        min_w = min(current_w, target_w)
        
        if min_h > 0 and min_w > 0:
            overlap_correct = (
                self.current_output[:min_h, :min_w] == 
                self.test_output[:min_h, :min_w]
            ).sum()
            # Normalize by target size (not overlap size) to penalize wrong dimensions
            content_accuracy = (overlap_correct / self.test_output.size) * 0.6
        else:
            content_accuracy = 0.0
        
        # Component 3: Exact size bonus (0 or 0.1)
        exact_size_bonus = 0.1 if (current_h == target_h and current_w == target_w) else 0.0
        
        # Total reward
        reward = size_accuracy + content_accuracy + exact_size_bonus
        
        return float(reward)
    
    def _get_observation(self):
        """Generate observation dict with all paired images."""
        obs = {}
        
        # Create training pairs
        for i in range(5):
            pair = self._make_pair(self.train_inputs[i], self.train_outputs[i])
            obs[f'pair_{i+1}'] = pair
        
        # Create test pair (input | current work)
        test_pair = self._make_pair(self.test_input, self.current_output)
        obs['test_pair'] = test_pair
        
        # Add mask information
        obs['num_valid_pairs'] = np.int32(self.num_valid_pairs)
        
        # Add action availability masks
        obs['can_copy'] = not self.has_copied
        obs['can_resize'] = not self.has_resized
        
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
    
    def _save_episode_data(self, final_reward):
        """Save the completed episode data for visualization."""
        self.last_episode_data = {
            'test_input': self.test_input.tolist(),
            'test_output': self.test_output.tolist(),  # Ground truth
            'agent_output': self.current_output.tolist(),  # Agent's final answer
            'final_reward': float(final_reward),
            'steps': int(self.step_count),
            'actions': self.action_history,  # Complete action history
        }
        
        # Write to a fixed file for web app to read
        try:
            import os
            output_file = os.path.join(os.getcwd(), 'latest_episode.json')
            with open(output_file, 'w') as f:
                json.dump(self.last_episode_data, f)
        except Exception as e:
            print(f"Warning: Could not save episode visualization data: {e}")
    
    def get_last_episode_visualization(self):
        """Get the last episode's grids for visualization."""
        if self.last_episode_data is None:
            return None
        return self.last_episode_data