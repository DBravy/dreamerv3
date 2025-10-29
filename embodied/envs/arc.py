import glob
import json
import random

import elements
import embodied
import numpy as np


class ARC(embodied.Env):
    """
    ARC Environment with enforced phase-based task structure and constrained color painting.
    
    PHASE SYSTEM:
    The environment enforces a structured workflow that mirrors human problem-solving:
    
    1. SETUP Phase:
       - Only 'resize' action is valid
       - Agent must set the output grid dimensions
       - Transitions to COLOR_SELECT after successful resize
    
    2. COLOR_SELECT Phase:
       - 'set_color' action: Pick a color to work on → transitions to PAINT
         * Agent must provide count predictions (count_0 through count_9) with set_color action
         * The count for the selected color determines how many times to paint
       - 'done' action: Submit answer and end episode (voluntary stopping)
    
    3. PAINT Phase:
       - 'paint' action: Paint with the currently selected color
       - AUTOMATIC TRANSITION: After painting N times (where N = count prediction for current color),
         automatically returns to COLOR_SELECT
       - 'done' action: Also allowed for backward compatibility (manual transition)
    
    CONSTRAINED PAINTING:
    The policy is constrained by the color count selection heads. When a color is selected,
    the environment uses the count prediction for that color (from count_0 to count_9 heads)
    to determine exactly how many times the agent should paint with that color before
    automatically transitioning back to COLOR_SELECT. This removes the burden on the
    policy head to learn when to stop painting each color.
    
    DONE ACTION BEHAVIOR:
    - Done in PAINT phase → Return to COLOR_SELECT (work on another color) [optional, auto-transitions]
    - Done in COLOR_SELECT phase → Submit answer and end episode
    
    This structure ensures agents work on one color at a time with a predetermined count,
    preventing pathological behavior and making learning more tractable.
    
    Observations include:
    - 'current_phase' (0=SETUP, 1=COLOR_SELECT, 2=PAINT)
    - 'current_color' (0-9)
    - 'remaining_paint_count' (how many more paints needed for current color)
    """
    
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
    
    def __init__(self, task, puzzle_dir='./arc-data/', version='V2', split='training', length=50, size=64, max_puzzles=None, repeat_single=False, puzzle_index=None, invalid_penalty=0.1, min_target_height=None, max_target_height=None, min_target_width=None, max_target_width=None):
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
        
        print(f"Loaded {len(self.puzzles)} ARC puzzles from {self.full_puzzle_path} ({version}/{split}); max_puzzles={self.max_puzzles}, repeat_single={self.repeat_single}, puzzle_index={self.puzzle_index}, invalid_penalty={self.invalid_penalty}{filter_str}")
        
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
        # Require explicit color selection after resize before painting
        self.has_selected_color = False
        self.painted_positions = set()  # Set of (x, y) tuples
        self.fixed_puzzle = None  # When repeat_single is True, hold onto the chosen puzzle
        
        # NEW: Track current selected color
        self.current_color = 1  # Default to blue (color 1) - black is disabled
        
        # NEW: Track which colors have been selected (for reward purposes)
        self.selected_colors = set()  # Colors that have been chosen at least once
        
        # PHASE SYSTEM: Track current phase of task execution
        # Phases: 'SETUP' (resize) → 'COLOR_SELECT' (pick color) → 'PAINT' (paint with color) → repeat COLOR_SELECT/PAINT
        self.phase = 'SETUP'  # Start in setup phase
        self.paints_in_current_phase = 0  # Count paints in current PAINT phase
        
        # NEW: Track expected paint count for current color
        self.expected_paint_count = 0  # How many times we should paint with current color
        self.remaining_paint_count = 0  # How many paints are left for current color
        
        # Track action validity
        self.last_action_valid = True
        self.invalid_action_count = 0
        self.invalid_action_types = {
            'paint_duplicate': 0, 
            'resize_duplicate': 0, 
            'paint_oob': 0,
            'paint_before_resize': 0,
            'paint_before_color': 0,
            'first_step_not_resize': 0,
            'second_step_not_set_color': 0,
            'set_color_before_resize': 0,
            'paint_same_color': 0
        }
        
        # Track previous accuracy for delta rewards (separate components)
        self.previous_grid_size_accuracy = 0.0
        self.previous_content_accuracy = 0.0
        
        # Track color counts for reward limiting
        self.target_color_counts = {}  # How many of each color in target
        self.rewarded_color_counts = {}  # How many times we've rewarded each color
    
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
            # Grid dimensions for each training pair (uses output grid dimensions)
            'pair_1_height': elements.Space(np.int32, (), 0, 30),
            'pair_1_width': elements.Space(np.int32, (), 0, 30),
            'pair_2_height': elements.Space(np.int32, (), 0, 30),
            'pair_2_width': elements.Space(np.int32, (), 0, 30),
            'pair_3_height': elements.Space(np.int32, (), 0, 30),
            'pair_3_width': elements.Space(np.int32, (), 0, 30),
            'pair_4_height': elements.Space(np.int32, (), 0, 30),
            'pair_4_width': elements.Space(np.int32, (), 0, 30),
            'pair_5_height': elements.Space(np.int32, (), 0, 30),
            'pair_5_width': elements.Space(np.int32, (), 0, 30),
            # Grid dimensions for the test target grid (ground truth)
            'test_grid_height': elements.Space(np.int32, (), 0, 30),
            'test_grid_width': elements.Space(np.int32, (), 0, 30),
            'num_valid_pairs': elements.Space(np.int32, (), 0, 5),  # How many of pair_1-5 are real (0-5)
        'valid_actions': elements.Space(np.int32, (4,), 0, 1),  # Mask for [paint, resize, done, set_color] matching action_type indices
        'current_color': elements.Space(np.int32, (), 0, 9),  # Currently selected color (0-9)
        'current_phase': elements.Space(np.int32, (), 0, 2),  # Current phase: 0=SETUP, 1=COLOR_SELECT, 2=PAINT
        'remaining_paint_count': elements.Space(np.int32, (), 0, 900),  # How many more paints needed for current color (0-900, though typically 0-30)
        'valid_colors': elements.Space(np.int32, (10,), 0, 1),  # Mask for colors 0-9 (0=invalid, 1=valid)
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
            'width': elements.Space(np.int32, (), 0, 30),   # Target width for resize (0-29 output, maps to 1-30 actual)
            'height': elements.Space(np.int32, (), 0, 30),  # Target height for resize (0-29 output, maps to 1-30 actual)
            # Count predictions for each color (0-30) - used to determine how many times to paint each color
            'count_0': elements.Space(np.int32, (), 0, 5),  # Black count
            'count_1': elements.Space(np.int32, (), 0, 5),  # Blue count
            'count_2': elements.Space(np.int32, (), 0, 5),  # Red count
            'count_3': elements.Space(np.int32, (), 0, 5),  # Green count
            'count_4': elements.Space(np.int32, (), 0, 5),  # Yellow count
            'count_5': elements.Space(np.int32, (), 0, 5),  # Gray count
            'count_6': elements.Space(np.int32, (), 0, 5),  # Magenta count
            'count_7': elements.Space(np.int32, (), 0, 5),  # Orange count
            'count_8': elements.Space(np.int32, (), 0, 5),  # Light Blue count
            'count_9': elements.Space(np.int32, (), 0, 5),  # Maroon count
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
        
        # Extract count predictions if present (for constrained color painting)
        count_predictions = {}
        for i in range(10):
            count_key = f'count_{i}'
            if count_key in action:
                count_predictions[i] = int(action[count_key])
        
        self.action_history.append(action_record)
        
        # Execute action on the grid (this sets self.last_action_valid)
        self._execute_action(action, count_predictions)
        self.step_count += 1
        
        # Check if this was a new useful color selection
        new_useful_color = False
        if action['action_type'] == 3:  # Set color action
            selected_color = int(action['color'])
            # Check if this is a new color (not previously selected)
            if selected_color not in previous_selected_colors:
                # Check if this color appears in the target output
                # Do not count black (0) as a useful new color
                if selected_color != 0 and np.any(self.test_output == selected_color):
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
        # Episode ends when:
        # 1. Time limit reached, OR
        # 2. Done action pressed from COLOR_SELECT phase (submit answer)
        
        is_done = (
            self.step_count >= self.length or  # Time limit
            (action['action_type'] == 2 and self.phase == 'COLOR_SELECT')  # Done in COLOR_SELECT = submit
        )
        
        # Note: Done in PAINT phase transitions to COLOR_SELECT (handled in _execute_action)
        # This allows agent to finish one color and pick another
        # Done in COLOR_SELECT submits the final answer and ends the episode
        
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
        self.has_selected_color = False
        self.painted_positions = set()
        
        # NEW: Reset color to default (blue)
        self.current_color = 1  # Blue - black is disabled
        self.selected_colors = set()  # Reset selected colors tracking
        self.selected_colors.add(1)  # Blue is selected by default
        
        # PHASE SYSTEM: Reset to SETUP phase
        self.phase = 'SETUP'
        self.paints_in_current_phase = 0
        self.expected_paint_count = 0
        self.remaining_paint_count = 0
        
        # Count how many of each color appear in the target grid
        self.target_color_counts = {}
        self.rewarded_color_counts = {}
        for color in range(10):  # Colors 0-9
            count = int(np.sum(self.test_output == color))
            self.target_color_counts[color] = count
            self.rewarded_color_counts[color] = 0
        
        # Reset validity tracking
        self.last_action_valid = True
        self.invalid_action_count = 0
        self.invalid_action_types = {
            'paint_duplicate': 0, 
            'resize_duplicate': 0, 
            'paint_oob': 0,
            'paint_before_resize': 0,
            'paint_before_color': 0,
            'first_step_not_resize': 0,
            'second_step_not_set_color': 0,
            'set_color_before_resize': 0,
            'paint_same_color': 0
        }
        
        # Initialize accuracy tracking to starting grid state
        # This ensures the first action is rewarded based on improvement from initial state
        _, initial_grid_size_accuracy, initial_content_accuracy = self._calculate_reward(new_useful_color=False)
        self.previous_grid_size_accuracy = 0.0  # Start from zero
        self.previous_content_accuracy = 0.0
        
        # Return initial observation
        obs = self._get_observation()
        obs['reward'] = np.float32(0)
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        
        return obs
    
    def _execute_action(self, action, count_predictions=None):
        """
        Modify the current_output grid based on action.
        
        PHASE SYSTEM:
        - SETUP phase: Only resize action is valid. After resize → COLOR_SELECT phase.
        - COLOR_SELECT phase: Only set_color or done actions are valid. After set_color → PAINT phase.
        - PAINT phase: Only paint action is valid. Agent must paint with current color.
          When remaining_paint_count reaches 0 → automatically transition to COLOR_SELECT.
        
        Args:
            action: Action dict containing action_type, x, y, color, width, height
            count_predictions: Dict mapping color_idx -> predicted count for that color
        """
        if count_predictions is None:
            count_predictions = {}
        
        # Convert all action values to Python integers at the start
        action_type = int(action['action_type'])
        x = int(action['x'])
        y = int(action['y'])
        
        # Assume action is valid until proven otherwise
        self.last_action_valid = True
        
        # PHASE-BASED ACTION VALIDATION
        if self.phase == 'SETUP':
            # In SETUP phase: only resize is allowed
            if action_type != 1:  # Not resize
                self.last_action_valid = False
                self.invalid_action_count += 1
                if 'wrong_action_in_setup' not in self.invalid_action_types:
                    self.invalid_action_types['wrong_action_in_setup'] = 0
                self.invalid_action_types['wrong_action_in_setup'] += 1
                return
            
            # Valid resize action - execute it
            # Agent outputs 0-29, add 1 to get actual dimensions 1-30
            new_height = int(np.clip(action['height'] + 1, 1, 30))
            new_width = int(np.clip(action['width'] + 1, 1, 30))
            
            # Create new grid
            new_grid = np.zeros((new_height, new_width), dtype=np.uint8)
            
            # Copy existing content where it fits (preserve content)
            old_h, old_w = self.current_output.shape
            copy_h = min(old_h, new_height)
            copy_w = min(old_w, new_width)
            new_grid[:copy_h, :copy_w] = self.current_output[:copy_h, :copy_w]
            
            self.current_output = new_grid
            self.has_resized = True
            self.painted_positions.clear()  # Clear painted positions on resize
            
            # TRANSITION: SETUP → COLOR_SELECT
            self.phase = 'COLOR_SELECT'
        
        elif self.phase == 'COLOR_SELECT':
            # In COLOR_SELECT phase: set_color or done (if no more work) is allowed
            if action_type == 3:  # Set color
                # Valid set_color action - execute it
                color = int(action['color'])
                new_color = np.clip(color, 0, 9) 
                        # CHECK: Prevent re-selecting already used colors
                if new_color in self.selected_colors:
                    self.last_action_valid = False
                    self.invalid_action_count += 1
                    if 'color_reselection' not in self.invalid_action_types:
                        self.invalid_action_types['color_reselection'] = 0
                    self.invalid_action_types['color_reselection'] += 1
                    return # Ensure color is in valid range
                self.current_color = new_color
                
                # Track this color as selected (for reward calculation)
                self.selected_colors.add(new_color)
                self.has_selected_color = True
                
                # Set expected paint count from count prediction head
                if new_color in count_predictions:
                    self.expected_paint_count = count_predictions[new_color]
                    self.remaining_paint_count = count_predictions[new_color]
                else:
                    # Default to 1 if no prediction available
                    self.expected_paint_count = 1
                    self.remaining_paint_count = 1
                
                # TRANSITION: COLOR_SELECT → PAINT
                self.phase = 'PAINT'
                self.paints_in_current_phase = 0  # Reset paint counter
            
            elif action_type == 2:  # Done action
                # Done is only valid if no more positions to paint
                # This will end the episode (handled in step() method)
                # No phase transition needed - episode ends
                pass
            
            else:
                # Wrong action type in COLOR_SELECT phase
                self.last_action_valid = False
                self.invalid_action_count += 1
                if 'wrong_action_in_color_select' not in self.invalid_action_types:
                    self.invalid_action_types['wrong_action_in_color_select'] = 0
                self.invalid_action_types['wrong_action_in_color_select'] += 1
                return
        
        elif self.phase == 'PAINT':
            # In PAINT phase: only paint action is allowed
            if action_type == 0:  # Paint action
                # Check if position is out of bounds
                h, w = self.current_output.shape
                if x >= w or y >= h or x < 0 or y < 0:
                    self.last_action_valid = False
                    self.invalid_action_count += 1
                    if 'paint_oob' not in self.invalid_action_types:
                        self.invalid_action_types['paint_oob'] = 0
                    self.invalid_action_types['paint_oob'] += 1
                    return
                
                # Check if position already painted
                if (x, y) in self.painted_positions:
                    self.last_action_valid = False
                    self.invalid_action_count += 1
                    if 'paint_duplicate' not in self.invalid_action_types:
                        self.invalid_action_types['paint_duplicate'] = 0
                    self.invalid_action_types['paint_duplicate'] += 1
                    return
                
                # Check if painting the same color that's already at this position
                if self.current_output[y, x] == self.current_color:
                    self.last_action_valid = False
                    self.invalid_action_count += 1
                    if 'paint_same_color' not in self.invalid_action_types:
                        self.invalid_action_types['paint_same_color'] = 0
                    self.invalid_action_types['paint_same_color'] += 1
                    return
                
                # Valid paint action - execute it using current_color
                self.current_output[y, x] = self.current_color  # NumPy arrays are [row, col] = [y, x]
                self.painted_positions.add((x, y))
                self.paints_in_current_phase += 1
                
                # Decrement remaining paint count
                self.remaining_paint_count -= 1
                
                # AUTO-TRANSITION: When remaining count reaches 0, return to COLOR_SELECT
                if self.remaining_paint_count <= 0:
                    self.phase = 'COLOR_SELECT'
                    self.remaining_paint_count = 0  # Reset to 0 for clean state
            
            elif action_type == 2:  # Done action - no longer needed, but handle gracefully
                # Allow done action to also transition back to COLOR_SELECT
                # This provides backward compatibility if needed
                if self.paints_in_current_phase == 0:
                    self.last_action_valid = False
                    self.invalid_action_count += 1
                    if 'done_before_painting' not in self.invalid_action_types:
                        self.invalid_action_types['done_before_painting'] = 0
                    self.invalid_action_types['done_before_painting'] += 1
                    return
                
                # TRANSITION: PAINT → COLOR_SELECT
                self.phase = 'COLOR_SELECT'
                self.remaining_paint_count = 0  # Reset to 0 for clean state
            
            else:
                # Wrong action type in PAINT phase
                self.last_action_valid = False
                self.invalid_action_count += 1
                if 'wrong_action_in_paint' not in self.invalid_action_types:
                    self.invalid_action_types['wrong_action_in_paint'] = 0
                self.invalid_action_types['wrong_action_in_paint'] += 1
                return
        
        # action_type == 2 outside PAINT phase is handled by allowing episode termination
        # in the step() method
    
    def _calculate_reward(self, new_useful_color=False):
        """
        Reward based on similarity to ground truth and size matching.
        
        PAINTING REWARD SYSTEM:
        - When agent paints, reward is ONLY given when BOTH:
          1. Correct color at the correct position: reward = 1.0
          2. Otherwise (wrong color, wrong position, or both): reward = 0.0
        
        For non-paint actions (resize, set_color, done):
        - Grid size accuracy and content accuracy are tracked SEPARATELY
        - Each component gives independent delta rewards (improvement from previous step)
        
        Returns:
            tuple: (reward, current_grid_size_accuracy, current_content_accuracy)
        """
        target_h, target_w = self.test_output.shape
        current_h, current_w = self.current_output.shape
        
        # Check if the last action was a paint action
        was_paint_action = False
        paint_reward = 0.0
        
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            if last_action['action_type'] == 0:  # Paint action
                was_paint_action = True
                
                # If the last paint action was invalid, apply invalid penalty and do not grant paint reward
                if not self.last_action_valid:
                    paint_reward = -self.invalid_penalty
                else:
                    # Get the painted position and color
                    painted_x = last_action['x']
                    painted_y = last_action['y']
                    painted_color = last_action['current_color']
                    
                    # Only reward if BOTH color AND position are correct
                    # Check if position is within target grid bounds
                    if painted_y < target_h and painted_x < target_w and painted_y >= 0 and painted_x >= 0:
                        # Check if the color at this position matches the target
                        if self.test_output[painted_y, painted_x] == painted_color:
                            paint_reward = 1.0
                        else:
                            paint_reward = 0.0
                    else:
                        # Position is out of bounds of target grid
                        paint_reward = 0.0
                
                # COMMENTED OUT: Previous reward logic with distance-based rewards
                # # Never reward painting with black (color 0)
                # if painted_color == 0:
                #     paint_reward = 0.0
                # # Check if this color exists in the target grid
                # elif not np.any(self.test_output == painted_color):
                #     # Wrong color (not in target) → 0 reward
                #     paint_reward = 0.0
                # # Check if we've already rewarded this color too many times
                # elif self.rewarded_color_counts[painted_color] >= self.target_color_counts[painted_color]:
                #     # Already painted this color as many times as it appears in target → 0 reward
                #     paint_reward = 0.0
                # else:
                #     # Correct color (exists in target) and still under the limit → base 0.5 reward
                #     paint_reward = 0.5
                #     
                #     # Increment the rewarded count for this color
                #     self.rewarded_color_counts[painted_color] += 1
                #     
                #     # Find the nearest position in target grid with this color
                #     # Get all positions with this color in the target
                #     target_positions = np.argwhere(self.test_output == painted_color)
                #     
                #     if len(target_positions) > 0:
                #         # Calculate Manhattan distance to each target position
                #         distances = np.abs(target_positions - np.array([painted_y, painted_x])).sum(axis=1)
                #         min_distance = distances.min()
                #         
                #         # Exponential distance reward: 0.5 * e^(-lambda * distance)
                #         # lambda controls the steepness (higher = steeper curve)
                #         # We want: far away = small reward, close = large reward
                #         # At distance 0: e^0 = 1, so reward = 0.5 * 1 = 0.5 additional
                #         # At large distance: e^(-large) ≈ 0, so reward ≈ 0 additional
                #         
                #         # Use lambda = 0.3 for moderate exponential decay
                #         # This makes each step closer significantly more valuable when close
                #         distance_lambda = 0.3
                #         distance_reward = 0.5 * np.exp(-distance_lambda * min_distance)
                #         
                #         paint_reward += distance_reward
        
        # ===== Component 1: Grid Size Accuracy (0 to 1.0) =====
        # SPARSE reward: Only give credit when BOTH height AND width are exactly correct
        # No partial credit for being close
        
        height_diff = abs(current_h - target_h)
        width_diff = abs(current_w - target_w)
        
        # Only give reward if BOTH dimensions are exact
        if height_diff == 0 and width_diff == 0:
            grid_size_accuracy = 1.0
        else:
            grid_size_accuracy = 0.0
        
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
            current_overlap = self.current_output[:min_h, :min_w]
            target_overlap = self.test_output[:min_h, :min_w]
            
            color_matches = 0
            for color in range(10):  # Colors 0-9
                current_count = (current_overlap == color).sum()
                target_count = (target_overlap == color).sum()
                color_matches += min(current_count, target_count)
            
            wrong_position_matches = color_matches - overlap_correct
            
            # Reward: full credit for exact matches, 40% credit for color-only matches
            exact_match_reward = (overlap_correct / self.test_output.size)
            color_match_reward = (wrong_position_matches / self.test_output.size) * 0.4
            content_accuracy = exact_match_reward + color_match_reward
        else:
            content_accuracy = 0.0
        
        # ===== Calculate Final Reward =====
        if was_paint_action:
            # For paint actions, use the direct paint reward
            reward = paint_reward
        else:
            # For non-paint actions, only reward the relevant component
            grid_size_improvement = grid_size_accuracy - self.previous_grid_size_accuracy
            content_improvement = content_accuracy - self.previous_content_accuracy
            
            color_selection_bonus = 0.02 if new_useful_color else 0.0
            invalid_penalty = -self.invalid_penalty if not self.last_action_valid else 0.0
            
            # Determine which action type was just performed
            last_action_type = self.action_history[-1]['action_type'] if len(self.action_history) > 0 else None
            
            if last_action_type == 1:  # Resize action
                # Only reward grid size improvement (resize doesn't affect content)
                reward = grid_size_improvement + invalid_penalty
            elif last_action_type == 3:  # Set color action
                # Only reward color selection bonus (doesn't affect grid or content)
                reward = color_selection_bonus + invalid_penalty
            else:  # Done action or other
                # No improvement rewards for done action
                reward = invalid_penalty
        
        return float(reward), float(grid_size_accuracy), float(content_accuracy)
    
    def _get_observation(self):
        """
        Generate observation dict with all paired images.
        
        Action masking system:
        - valid_actions: [4] array masking action types matching action_type indices [0:paint, 1:resize, 2:done, 3:set_color]
          - paint (index 0) is only available AFTER resize is performed (0 before resize, 1 after)
          - resize (index 1) can only be used once as the first action (1 on first step, 0 after)
          - done (index 2) is only available after 10 steps (0 before step 10, 1 after)
          - set_color (index 3) is only available AFTER step 0 (0 on step 0, 1 after)
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

        # Add grid dimensions for each training pair (use output grid dims)
        for i in range(5):
            if i < self.num_valid_pairs:
                out_h, out_w = self.train_outputs[i].shape
                obs[f'pair_{i+1}_height'] = np.int32(min(out_h, 30))
                obs[f'pair_{i+1}_width'] = np.int32(min(out_w, 30))
            else:
                # Zero out invalid pairs to match masking behavior for images
                obs[f'pair_{i+1}_height'] = np.int32(0)
                obs[f'pair_{i+1}_width'] = np.int32(0)

        # Add grid dimensions for the test target grid (ground truth)
        t_h, t_w = self.test_output.shape
        obs['test_grid_height'] = np.int32(min(t_h, 30))
        obs['test_grid_width'] = np.int32(min(t_w, 30))

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
        
        # Action availability mask based on CURRENT PHASE
        # valid_actions: [paint, resize, done, set_color] matching action_type indices [0, 1, 2, 3]
        has_any_valid_positions = np.int32(obs['valid_positions'].sum() > 0)
        
        if self.phase == 'SETUP':
            # In SETUP: only resize is valid
            obs['valid_actions'] = np.array([0, 1, 0, 0], dtype=np.int32)
        
        elif self.phase == 'COLOR_SELECT':
            # In COLOR_SELECT: set_color and done are both always valid
            # done = submit answer and end episode
            # set_color = continue working on another color
            obs['valid_actions'] = np.array([
                0,  # [0] paint (not in this phase)
                0,  # [1] resize (not in this phase)
                1,  # [2] done (always valid - submits answer)
                1,  # [3] set_color (always valid - pick another color)
            ], dtype=np.int32)
        
        elif self.phase == 'PAINT':
            # In PAINT: only paint action is valid
            # Auto-transitions to COLOR_SELECT when remaining_paint_count reaches 0
            # Done action is still allowed for backward compatibility
            obs['valid_actions'] = np.array([
                1 if has_any_valid_positions == 1 else 0,  # [0] paint (valid if positions available)
                0,  # [1] resize (not allowed in PAINT phase)
                1 if self.paints_in_current_phase > 0 else 0,  # [2] done (optional, for backward compatibility)
                0,  # [3] set_color (not allowed in PAINT phase)
            ], dtype=np.int32)
        
        else:
            # Fallback (should never happen)
            obs['valid_actions'] = np.array([0, 0, 1, 0], dtype=np.int32)
        
        # Add current phase to observation
        # Encode as: 0=SETUP, 1=COLOR_SELECT, 2=PAINT
        phase_map = {'SETUP': 0, 'COLOR_SELECT': 1, 'PAINT': 2}
        obs['current_phase'] = np.int32(phase_map.get(self.phase, 0))
        
        # Add current color to observation
        obs['current_color'] = np.int32(self.current_color)
        
        # Add remaining paint count to observation
        # This tells the agent how many more times it needs to paint with the current color
        # Clamp to 0 to ensure it's always non-negative (can temporarily go negative during transitions)
        obs['remaining_paint_count'] = np.int32(max(0, self.remaining_paint_count))
        
        # Color mask: disable black (color 0) and any colors already selected in this episode
        valid_colors = np.ones(10, dtype=np.int32)
        valid_colors[0] = 0  # never allow black
        for c in self.selected_colors:
            if 0 <= c < 10:
                valid_colors[c] = 0
        obs['valid_colors'] = valid_colors.astype(np.int32)
        
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