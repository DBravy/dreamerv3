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
    
    def __init__(self, task, puzzle_dir='./arc-data/', version='V2', split='training', length=100, size=64, max_puzzles=None, repeat_single=False, puzzle_index=None, invalid_penalty=0.1):
        """
        Args:
            task: Not used, but required by interface
            puzzle_dir: Path to arc-data directory (default: './arc-data/')
            version: 'V1' or 'V2' (default: 'V2')
            split: 'training' or 'evaluation' (default: 'training')
            length: Maximum steps per episode
            size: Size to pad grids to (default 64×64)
            max_puzzles: If set, limit the number of loaded puzzles to this many (use first N)
            repeat_single: If True, select a single puzzle on first reset and repeat it every episode
            puzzle_index: Optional explicit index into the loaded puzzles to always use (overrides random)
            invalid_penalty: Penalty for invalid actions (default: 0.1)
        """
        self.puzzle_dir = puzzle_dir
        self.version = version
        self.split = split
        self.length = length
        self.size = size
        self.max_puzzles = max_puzzles
        self.repeat_single = repeat_single
        self.puzzle_index = puzzle_index
        self.invalid_penalty = invalid_penalty
        
        # Construct the full path to puzzles
        self.full_puzzle_path = f"{puzzle_dir}/{version}/data/{split}"
        self.puzzles = self._load_puzzles()
        
        if len(self.puzzles) == 0:
            raise ValueError(f"No puzzles found in {self.full_puzzle_path}. Please check the path.")
        
        print(f"Loaded {len(self.puzzles)} ARC puzzles from {self.full_puzzle_path} ({version}/{split}); max_puzzles={self.max_puzzles}, repeat_single={self.repeat_single}, puzzle_index={self.puzzle_index}, invalid_penalty={self.invalid_penalty}")
        
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
        self.fixed_puzzle = None  # When repeat_single is True, hold onto the chosen puzzle
        
        # NEW: Track action validity
        self.last_action_valid = True
        self.invalid_action_count = 0
        self.invalid_action_types = {'paint_duplicate': 0, 'copy_duplicate': 0, 'resize_duplicate': 0, 'paint_oob': 0}
    
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
        
        # Optionally limit the number of puzzles
        if self.max_puzzles is not None:
            try:
                maxn = int(self.max_puzzles)
            except Exception:
                maxn = None
            if maxn is not None and maxn > 0:
                puzzles = puzzles[:maxn]
        
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
            'test_pair': elements.Space(np.uint8, pair_shape),  # test_input | current_work (for policy to see current state)
            'target_pair': elements.Space(np.uint8, pair_shape),  # test_input | ground_truth (for world model to learn complete solution)
            'num_valid_pairs': elements.Space(np.int32, (), 0, 5),  # How many of pair_1-5 are real (0-5)
            'valid_actions': elements.Space(np.int32, (4,), 0, 1),  # Mask for [paint, copy, resize, done]
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            'valid_actions': elements.Space(np.int32, (4,), 0, 1),  # Mask for [paint, copy, resize, done]
            'valid_positions': elements.Space(np.int32, (30, 30), 0, 1),  # NEW: Spatial mask for paintable positions
        }
    
    @property
    def act_space(self):
        """Define action space for grid editing."""
        return {
            'action_type': elements.Space(np.int32, (), 0, 4),  # 0:paint, 1:copy, 2:resize, 3:done
            'x': elements.Space(np.int32, (), 0, 30),
            'y': elements.Space(np.int32, (), 0, 30),
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
        
        # Execute action on the grid (this sets self.last_action_valid)
        self._execute_action(action)
        self.step_count += 1
        
        # Calculate reward (includes penalty for invalid actions)
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
        # Pick puzzle according to settings
        if self.repeat_single:
            # Initialize fixed puzzle on first reset
            if self.fixed_puzzle is None:
                if self.puzzle_index is not None:
                    try:
                        idx = int(self.puzzle_index)
                    except Exception:
                        idx = 0
                    # Treat negative indices or -1 as unset -> choose randomly
                    if idx < 0:
                        self.fixed_puzzle = random.choice(self.puzzles)
                    else:
                        idx = max(0, min(idx, len(self.puzzles) - 1))
                        self.fixed_puzzle = self.puzzles[idx]
                else:
                    self.fixed_puzzle = random.choice(self.puzzles)
            self.current_puzzle = self.fixed_puzzle
        else:
            self.current_puzzle = random.choice(self.puzzles)
        
        # Extract training examples
        train_pairs = self.current_puzzle['train']
        
        # Pad training pairs to exactly 5 (ARC has varying amounts)
        self.train_inputs = []
        self.train_outputs = []
        for i in range(5):
            if i < len(train_pairs):
                inp = np.array(train_pairs[i]['input'], dtype=np.uint8)
                out = np.array(train_pairs[i]['output'], dtype=np.uint8)
            else:
                # Use empty placeholder
                inp = np.zeros((1, 1), dtype=np.uint8)
                out = np.zeros((1, 1), dtype=np.uint8)
            self.train_inputs.append(inp)
            self.train_outputs.append(out)
        self.num_valid_pairs = min(len(train_pairs), 5)
        
        # Extract test case (always use the first test case)
        test_pair = self.current_puzzle['test'][0]
        self.test_input = np.array(test_pair['input'], dtype=np.uint8)
        self.test_output = np.array(test_pair['output'], dtype=np.uint8)
        
        # Initialize current output as empty grid (same size as test input)
        self.current_output = np.zeros_like(self.test_input)
        
        # Reset episode state
        self.step_count = 0
        self.action_history = []
        self.has_copied = False
        self.has_resized = False
        self.painted_positions = set()
        
        # NEW: Reset validity tracking
        self.last_action_valid = True
        self.invalid_action_count = 0
        self.invalid_action_types = {'paint_duplicate': 0, 'copy_duplicate': 0, 'resize_duplicate': 0, 'paint_oob': 0}
        
        # Return initial observation
        obs = self._get_observation()
        obs['reward'] = np.float32(0)
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        
        return obs
    
    def _execute_action(self, action):
        """Modify the current_output grid based on action."""
        # Convert all action values to Python integers at the start
        action_type = int(action['action_type'])
        x = int(action['x'])
        y = int(action['y'])
        
        # Assume action is valid until proven otherwise
        self.last_action_valid = True
        
        if action_type == 0:  # Paint
            # Check if position is out of bounds
            h, w = self.current_output.shape
            if x >= w or y >= h or x < 0 or y < 0:
                self.last_action_valid = False
                self.invalid_action_count += 1
                self.invalid_action_types['paint_oob'] += 1
                return
            
            # NEW: Check if position already painted
            if (x, y) in self.painted_positions:
                self.last_action_valid = False
                self.invalid_action_count += 1
                self.invalid_action_types['paint_duplicate'] += 1
                return
            
            # Valid paint action - execute it
            color = int(action['color'])
            self.current_output[y, x] = color  # NumPy arrays are [row, col] = [y, x]
            self.painted_positions.add((x, y))
        
        elif action_type == 1:  # Copy entire input
            # Check if copy has already been used
            if self.has_copied:
                self.last_action_valid = False
                self.invalid_action_count += 1
                self.invalid_action_types['copy_duplicate'] += 1
                return
            
            # Valid copy action - execute it
            self.current_output = self.test_input.copy()
            self.has_copied = True
        
        elif action_type == 2:  # Resize
            # Check if resize has already been used
            if self.has_resized:
                self.last_action_valid = False
                self.invalid_action_count += 1
                self.invalid_action_types['resize_duplicate'] += 1
                return
            
            # Valid resize action - execute it
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
        
        # action_type == 3 is "done", no grid modification (always valid)
    
    def _calculate_reward(self):
        """
        Reward based on similarity to ground truth and size matching.
        
        Reward components:
        1. Size accuracy: Reward for getting closer to correct dimensions
        2. Content accuracy: Percentage of correct cells (in overlapping region)
        3. Bonus: Extra reward for exact size match
        4. NEW: Penalty for invalid actions
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
        
        # Component 4: Invalid action penalty (NEW)
        invalid_penalty = -self.invalid_penalty if not self.last_action_valid else 0.0
        
        # Total reward
        reward = size_accuracy + content_accuracy + exact_size_bonus + invalid_penalty
        
        return float(reward)
    
    def _get_observation(self):
        """Generate observation dict with all paired images."""
        obs = {}
        
        # Create training pairs
        for i in range(5):
            pair = self._make_pair(self.train_inputs[i], self.train_outputs[i])
            obs[f'pair_{i+1}'] = pair
        
        # Create test pair (input | current work) - for policy to see current state
        test_pair = self._make_pair(self.test_input, self.current_output)
        obs['test_pair'] = test_pair
        
        # Create target pair (input | ground truth) - for world model to learn complete solution
        target_pair = self._make_pair(self.test_input, self.test_output)
        obs['target_pair'] = target_pair
        
        # Add mask information
        obs['num_valid_pairs'] = np.int32(self.num_valid_pairs)

        # Action availability mask: [paint, copy, resize, done]
        obs['valid_actions'] = np.array([
            1,                          # paint always allowed
            0 if self.has_copied else 1,
            0 if self.has_resized else 1,
            1,                          # done always allowed
        ], dtype=np.int32)

        # NEW: Spatial mask for valid paint positions
        valid_positions = np.ones((30, 30), dtype=np.int32)
        for x, y in self.painted_positions:
            if 0 <= x < 30 and 0 <= y < 30:
                valid_positions[x, y] = 0
        obs['valid_positions'] = valid_positions
        
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
            # NEW: Invalid action statistics
            'invalid_action_count': int(self.invalid_action_count),
            'invalid_action_types': self.invalid_action_types,
            'invalid_action_rate': float(self.invalid_action_count / max(1, self.step_count)),
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