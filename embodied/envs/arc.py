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
    
    def __init__(self, task, puzzle_dir='./arc-data/', version='V2', split='training', length=50, size=64, max_puzzles=None, repeat_single=False, puzzle_index=None, invalid_penalty=0.1, size_reward_exponent=2.0, min_target_height=None, max_target_height=None, min_target_width=None, max_target_width=None):
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
            size_reward_exponent: Exponent for grid size reward curve (default: 2.0)
                - 1.0 = linear (equal reward for each step closer)
                - 2.0 = quadratic (being close to target matters more)
                - Higher values = even steeper curve (more emphasis on being very close)
            min_target_height: Optional minimum height for all grids in puzzle (filters puzzles)
            max_target_height: Optional maximum height for all grids in puzzle (filters puzzles)
            min_target_width: Optional minimum width for all grids in puzzle (filters puzzles)
            max_target_width: Optional maximum width for all grids in puzzle (filters puzzles)
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
        self.size_reward_exponent = size_reward_exponent
        
        # Convert 0 or negative values to None (meaning no filter)
        self.min_target_height = min_target_height if (min_target_height and min_target_height > 0) else None
        self.max_target_height = max_target_height if (max_target_height and max_target_height > 0) else None
        self.min_target_width = min_target_width if (min_target_width and min_target_width > 0) else None
        self.max_target_width = max_target_width if (max_target_width and max_target_width > 0) else None
        
        # Construct the full path to puzzles
        self.full_puzzle_path = f"{puzzle_dir}/{version}/data/{split}"
        self.puzzles = self._load_puzzles()
        
        if len(self.puzzles) == 0:
            raise ValueError(f"No puzzles found in {self.full_puzzle_path}. Please check the path.")
        
        filter_info = []
        if self.min_target_height is not None or self.max_target_height is not None:
            filter_info.append(f"height={self.min_target_height or 0}-{self.max_target_height or '∞'}")
        if self.min_target_width is not None or self.max_target_width is not None:
            filter_info.append(f"width={self.min_target_width or 0}-{self.max_target_width or '∞'}")
        filter_str = f", filters=[{', '.join(filter_info)}]" if filter_info else ""
        
        print(f"Loaded {len(self.puzzles)} ARC puzzles from {self.full_puzzle_path} ({version}/{split}); max_puzzles={self.max_puzzles}, repeat_single={self.repeat_single}, puzzle_index={self.puzzle_index}, invalid_penalty={self.invalid_penalty}, size_reward_exponent={self.size_reward_exponent}{filter_str}")
        
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
        self.has_resized = False
        self.painted_positions = set()  # Set of (x, y) tuples
        self.fixed_puzzle = None  # When repeat_single is True, hold onto the chosen puzzle
        
        # NEW: Track current selected color
        self.current_color = 0  # Default to black (color 0)
        
        # NEW: Track which colors have been selected (for reward purposes)
        self.selected_colors = set()  # Colors that have been chosen at least once
        
        # Track action validity
        self.last_action_valid = True
        self.invalid_action_count = 0
        self.invalid_action_types = {
            'paint_duplicate': 0, 
            'resize_duplicate': 0, 
            'paint_oob': 0,
            'paint_before_resize': 0
        }
        
        # Track previous accuracy for delta rewards (separate components)
        self.previous_grid_size_accuracy = 0.0
        self.previous_content_accuracy = 0.0
    
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
                        # Apply grid size filters if specified
                        if self._puzzle_matches_filters(puzzle):
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
    
    def _puzzle_matches_filters(self, puzzle):
        """Check if a puzzle matches the grid size filters."""
        # If no filters are set, accept all puzzles
        if (self.min_target_height is None and self.max_target_height is None and
            self.min_target_width is None and self.max_target_width is None):
            return True
        
        try:
            # Check ALL grids in the puzzle (training pairs and test)
            grids_to_check = []
            
            # Add all training pair inputs and outputs
            for train_pair in puzzle.get('train', []):
                grids_to_check.append(train_pair.get('input'))
                grids_to_check.append(train_pair.get('output'))
            
            # Add test input and output (first test case)
            if 'test' in puzzle and len(puzzle['test']) > 0:
                grids_to_check.append(puzzle['test'][0].get('input'))
                grids_to_check.append(puzzle['test'][0].get('output'))
            
            # Check each grid against the filters
            for grid in grids_to_check:
                if grid is None:
                    continue
                
                height = len(grid)
                width = len(grid[0]) if height > 0 else 0
                
                # Check height constraints
                if self.min_target_height is not None and height < self.min_target_height:
                    return False
                if self.max_target_height is not None and height > self.max_target_height:
                    return False
                
                # Check width constraints
                if self.min_target_width is not None and width < self.min_target_width:
                    return False
                if self.max_target_width is not None and width > self.max_target_width:
                    return False
            
            return True
        except (KeyError, IndexError, TypeError):
            # If we can't get dimensions, reject the puzzle
            return False
    
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
            'valid_actions': elements.Space(np.int32, (4,), 0, 1),  # Mask for [paint, resize, done, set_color] matching action_type indices
            'current_color': elements.Space(np.int32, (), 0, 9),  # Currently selected color (0-9)
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            'valid_positions': elements.Space(np.int32, (30, 30), 0, 1),  # Spatial mask [y, x] for paintable positions (1=valid, 0=invalid/painted/out-of-bounds)
        }
    
    @property
    def act_space(self):
        """Define action space for grid editing."""
        return {
            'action_type': elements.Space(np.int32, (), 0, 4),  # 0:paint, 1:resize, 2:done, 3:set_color
            'x': elements.Space(np.int32, (), 0, 30),
            'y': elements.Space(np.int32, (), 0, 30),
            'color': elements.Space(np.int32, (), 0, 9),  # Used only for set_color action
            'width': elements.Space(np.int32, (), 0, 30),   # Target width for resize
            'height': elements.Space(np.int32, (), 0, 30),  # Target height for resize
            'reset': elements.Space(bool),
        }
    
    def step(self, action):
        """Execute action and return observation."""
        # Handle reset
        if action['reset'] or self.current_puzzle is None:
            return self._reset()
        
        # Track the previous set of selected colors (before action execution)
        previous_selected_colors = self.selected_colors.copy()
        
        # Store action in history before executing
        action_record = {
            'step': self.step_count,
            'action_type': int(action['action_type']),
            'x': int(action['x']),
            'y': int(action['y']),
            'color': int(action['color']),
            'width': int(action['width']),
            'height': int(action['height']),
            'current_color': int(self.current_color),  # Track what color was active
        }
        self.action_history.append(action_record)
        
        # Execute action on the grid (this sets self.last_action_valid)
        self._execute_action(action)
        self.step_count += 1
        
        # Check if this was a new useful color selection
        new_useful_color = False
        if action['action_type'] == 3:  # Set color action
            selected_color = int(action['color'])
            # Check if this is a new color (not previously selected)
            if selected_color not in previous_selected_colors:
                # Check if this color appears in the target output
                if np.any(self.test_output == selected_color):
                    new_useful_color = True
        
        # Calculate reward (delta-based: improvement + one-time bonuses/penalties)
        reward, current_grid_size_accuracy, current_content_accuracy = self._calculate_reward(new_useful_color)
        
        # Add reward to the action record
        action_record['reward'] = float(reward)
        action_record['grid_size_accuracy'] = float(current_grid_size_accuracy)
        action_record['content_accuracy'] = float(current_content_accuracy)
        
        # Update tracked accuracy for next step's delta calculation
        self.previous_grid_size_accuracy = current_grid_size_accuracy
        self.previous_content_accuracy = current_content_accuracy
        
        # Check if done
        is_done = (
            action['action_type'] == 2 or  # "done" action
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
        
        # Initialize current output as blank 3x3 grid
        self.current_output = np.zeros((3, 3), dtype=np.uint8)
        
        # Reset episode state
        self.step_count = 0
        self.action_history = []
        self.has_resized = False
        self.painted_positions = set()
        
        # NEW: Reset color to default (black)
        self.current_color = 0
        self.selected_colors = set()  # Reset selected colors tracking
        self.selected_colors.add(0)  # Black is selected by default
        
        # Reset validity tracking
        self.last_action_valid = True
        self.invalid_action_count = 0
        self.invalid_action_types = {
            'paint_duplicate': 0, 
            'resize_duplicate': 0, 
            'paint_oob': 0,
            'paint_before_resize': 0
        }
        
        # Initialize accuracy tracking to starting grid state
        # This ensures the first action is rewarded based on improvement from initial state
        _, initial_grid_size_accuracy, initial_content_accuracy = self._calculate_reward(new_useful_color=False)
        self.previous_grid_size_accuracy = initial_grid_size_accuracy
        self.previous_content_accuracy = initial_content_accuracy
        
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
        
        if action_type == 0:  # Paint (using current_color)
            # Check if resize has been done first
            if not self.has_resized:
                self.last_action_valid = False
                self.invalid_action_count += 1
                if 'paint_before_resize' not in self.invalid_action_types:
                    self.invalid_action_types['paint_before_resize'] = 0
                self.invalid_action_types['paint_before_resize'] += 1
                return
            
            # Check if position is out of bounds
            h, w = self.current_output.shape
            if x >= w or y >= h or x < 0 or y < 0:
                self.last_action_valid = False
                self.invalid_action_count += 1
                self.invalid_action_types['paint_oob'] += 1
                return
            
            # Check if position already painted
            if (x, y) in self.painted_positions:
                self.last_action_valid = False
                self.invalid_action_count += 1
                self.invalid_action_types['paint_duplicate'] += 1
                return
            
            # Valid paint action - execute it using current_color
            self.current_output[y, x] = self.current_color  # NumPy arrays are [row, col] = [y, x]
            self.painted_positions.add((x, y))
        
        elif action_type == 1:  # Resize
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
        
        elif action_type == 3:  # Set color
            # Update the current color
            color = int(action['color'])
            new_color = np.clip(color, 0, 9)  # Ensure color is in valid range
            self.current_color = new_color
            
            # Track this color as selected (for reward calculation)
            self.selected_colors.add(new_color)
        
        # action_type == 2 is "done", no grid modification (always valid)
    
    def _calculate_reward(self, new_useful_color=False):
        """
        Reward based on similarity to ground truth and size matching.
        
        DELTA REWARD SYSTEM WITH SEPARATE COMPONENTS:
        - Grid size accuracy and content accuracy are tracked SEPARATELY
        - Each component gives independent delta rewards (improvement from previous step)
        - Both components are scaled to 0-1.0 range for equal importance
        - This allows the agent to be rewarded for improvements in either dimension
        
        Reward components:
        1. Grid size accuracy (0 to 1.0): Dense reward with non-linear curve
           - Height accuracy: (1.0 - |current_h - target_h| / 30) ** exponent
           - Width accuracy: (1.0 - |current_w - target_w| / 30) ** exponent
           - Combined: (height_accuracy + width_accuracy) / 2
           - Non-linear curve (default exponent=2.0) makes being close to target more valuable:
             * Far from target (e.g. 30→29): small reward improvement
             * Close to target (e.g. 4→3): large reward improvement
           - Getting exact height OR exact width gives 0.5 accuracy (half credit)
           - Getting both gives 1.0 accuracy (full credit)
        
        2. Content accuracy (0 to 1.0): Percentage of correct cells in overlapping region
           - Full credit for correct position + correct color
           - Partial credit (40%) for correct color in wrong position
           - Scaled to 0-1.0 range
        
        3. Bonus: Small reward for selecting a new useful color (one-time)
        4. Penalty: For invalid actions only (not for wrong grid size)
        
        Returns:
            tuple: (reward, current_grid_size_accuracy, current_content_accuracy)
        """
        target_h, target_w = self.test_output.shape
        current_h, current_w = self.current_output.shape
        
        # ===== Component 1: Grid Size Accuracy (0 to 1.0) =====
        # Dense reward that judges height and width separately
        # Uses a non-linear curve to make being close to the target more valuable
        # Each dimension contributes 0.5 to the total (so 0.5 + 0.5 = 1.0 max)
        
        # Calculate linear accuracy (1.0 at exact match, 0.0 at max distance)
        height_linear = max(0.0, 1.0 - abs(current_h - target_h) / 30.0)
        width_linear = max(0.0, 1.0 - abs(current_w - target_w) / 30.0)
        
        # Apply non-linear curve using the exponent
        # exponent = 1.0: linear (each step equally valuable)
        # exponent = 2.0: quadratic (being close is much more valuable)
        # exponent > 2.0: even steeper (heavily rewards being very close)
        height_accuracy = height_linear ** self.size_reward_exponent
        width_accuracy = width_linear ** self.size_reward_exponent
        
        # Combined grid size accuracy (average of height and width)
        # This means: exact height only = 0.5, exact width only = 0.5, both = 1.0
        grid_size_accuracy = (height_accuracy + width_accuracy) / 2.0
        
        # ===== Component 2: Content Accuracy (0 to 1.0) =====
        # Calculate accuracy on the overlapping region
        min_h = min(current_h, target_h)
        min_w = min(current_w, target_w)
        
        if min_h > 0 and min_w > 0:
            # Full credit: correct position AND correct color
            overlap_correct = (
                self.current_output[:min_h, :min_w] == 
                self.test_output[:min_h, :min_w]
            ).sum()
            
            # Partial credit: correct color but wrong position
            # Count how many of each color appear in both grids
            current_overlap = self.current_output[:min_h, :min_w]
            target_overlap = self.test_output[:min_h, :min_w]
            
            # For each color, find the minimum count between current and target
            # This gives us the maximum number of correct color matches possible
            color_matches = 0
            for color in range(10):  # Colors 0-9
                current_count = (current_overlap == color).sum()
                target_count = (target_overlap == color).sum()
                color_matches += min(current_count, target_count)
            
            # Subtract the exact matches to get wrong-position matches
            wrong_position_matches = color_matches - overlap_correct
            
            # Reward: full credit for exact matches, 40% credit for color-only matches
            # Scaled to 0-1.0 range (previously was 0-0.7)
            exact_match_reward = (overlap_correct / self.test_output.size)
            color_match_reward = (wrong_position_matches / self.test_output.size) * 0.4
            content_accuracy = exact_match_reward + color_match_reward
        else:
            content_accuracy = 0.0
        
        # ===== Delta Rewards for Each Component =====
        # Each component gives reward based on improvement from previous step
        grid_size_improvement = grid_size_accuracy - self.previous_grid_size_accuracy
        content_improvement = content_accuracy - self.previous_content_accuracy
        
        # ===== One-Time Bonuses and Penalties =====
        # Component 3: Useful color selection bonus (0 or 0.02) - ONE-TIME BONUS
        # Small reward for selecting a color that appears in the target
        color_selection_bonus = 0.02 if new_useful_color else 0.0
        
        # Component 4: Invalid action penalty - ONE-TIME PENALTY
        # Only for invalid actions, NOT for wrong grid size
        invalid_penalty = -self.invalid_penalty if not self.last_action_valid else 0.0
        
        # ===== Final Reward =====
        # Sum of both improvement components plus bonuses/penalties
        # Both grid size and content can contribute equally to reward
        reward = grid_size_improvement + content_improvement + color_selection_bonus + invalid_penalty
        
        return float(reward), float(grid_size_accuracy), float(content_accuracy)
    
    def _get_observation(self):
        """
        Generate observation dict with all paired images.
        
        Action masking system:
        - valid_actions: [4] array masking action types matching action_type indices [0:paint, 1:resize, 2:done, 3:set_color]
          - paint (index 0) is only available AFTER resize is performed (0 before resize, 1 after)
          - resize (index 1) can only be used once as the first action (1 on first step, 0 after)
          - done (index 2) is only available after 10 steps (0 before step 10, 1 after)
          - set_color (index 3) is always available (1)
        - valid_positions: [30, 30] array masking spatial positions for painting
          - 1 = valid position (inside grid boundaries AND not yet painted)
          - 0 = invalid position (outside grid boundaries OR already painted)
          - Coordinates are [y, x] = [row, col] to match NumPy convention
          - Updates dynamically when grid is resized
        - current_color: scalar indicating the currently selected color (0-9)
        """
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

        # Action availability mask: [paint, resize, done, set_color] matching action_type indices [0, 1, 2, 3]
        # Enforce: resize must be first action, then only paint/set_color/done allowed
        obs['valid_actions'] = np.array([
            1 if self.has_resized else 0,  # [0] paint only allowed AFTER resize
            1 if (self.step_count == 0 and not self.has_resized) else 0,  # [1] resize ONLY on step 0
            1 if self.step_count >= 10 else 0,  # [2] done only allowed after 10 steps
            1,                              # [3] set_color always allowed
        ], dtype=np.int32)
        
        # Add current color to observation
        obs['current_color'] = np.int32(self.current_color)

        # Spatial mask for valid paint positions
        # Start with all positions marked as invalid (0)
        valid_positions = np.zeros((30, 30), dtype=np.int32)
        
        # Get current grid dimensions
        current_h, current_w = self.current_output.shape
        
        # Mark positions within the current grid boundaries as valid (1)
        # Only if they haven't been painted yet
        # Note: painted_positions stores (x, y) where x=col, y=row
        # valid_positions is indexed as [row, col] = [y, x] to match NumPy convention
        for y in range(min(current_h, 30)):  # rows
            for x in range(min(current_w, 30)):  # cols
                if (x, y) not in self.painted_positions:
                    valid_positions[y, x] = 1
        
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
        # Calculate total reward by summing all step rewards
        total_reward = sum(action.get('reward', 0.0) for action in self.action_history)
        
        self.last_episode_data = {
            'test_input': self.test_input.tolist(),
            'test_output': self.test_output.tolist(),  # Ground truth
            'agent_output': self.current_output.tolist(),  # Agent's final answer
            'final_reward': float(final_reward),
            'total_reward': float(total_reward),  # Sum of all step rewards
            'steps': int(self.step_count),
            'actions': self.action_history,  # Complete action history
            # Invalid action statistics
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