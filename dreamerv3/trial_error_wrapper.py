import numpy as np
import embodied


class TrialErrorWrapper(embodied.Env):
    """
    Wrapper that tries different actions until one produces a reward improvement.
    When an action doesn't improve reward, it reverts the state and tries another.
    """
    
    def __init__(self, env, max_attempts=10, reward_threshold=0.01):
        """
        Args:
            env: The wrapped environment (ARC)
            max_attempts: Maximum number of action attempts before accepting any result
            reward_threshold: Minimum reward improvement to accept an action
        """
        self._env = env
        self.max_attempts = max_attempts
        self.reward_threshold = reward_threshold
        
        # Track state for reverting
        self._saved_state = None
        self._last_reward = 0.0
        self._attempt_count = 0
        
    @property
    def obs_space(self):
        return self._env.obs_space
    
    @property
    def act_space(self):
        return self._env.act_space
    
    def step(self, action):
        """
        Try the action. If reward doesn't improve, revert and try a different action.
        This happens automatically until reward improves or max_attempts reached.
        """
        # Handle reset
        if action.get('reset', False):
            obs = self._env.step(action)
            self._last_reward = obs.get('reward', 0.0)
            self._save_state()
            return obs
        
        # Save current state before attempting action
        if self._saved_state is None:
            self._save_state()
        
        # Try actions until we get reward improvement
        best_obs = None
        best_reward = self._last_reward
        accepted = False
        
        for attempt in range(self.max_attempts):
            # Try the action
            obs = self._env.step(action)
            current_reward = obs.get('reward', 0.0)
            
            # Check if this action improved the reward
            reward_improvement = current_reward - self._last_reward
            
            if reward_improvement >= self.reward_threshold:
                # Success! Accept this action
                self._last_reward = current_reward
                self._save_state()
                accepted = True
                return obs
            
            # No improvement - revert state
            if attempt < self.max_attempts - 1:
                self._restore_state()
                # Modify action slightly for next attempt
                action = self._perturb_action(action)
            else:
                # Max attempts reached - accept whatever we have
                best_obs = obs
        
        # If we get here, we've exhausted attempts
        # Accept the last attempt even without reward improvement
        self._last_reward = best_obs.get('reward', self._last_reward)
        self._save_state()
        return best_obs
    
    def _save_state(self):
        """Save the current environment state."""
        # For ARC environment, we need to save:
        # - current_output grid
        # - has_copied, has_resized flags
        # - painted_positions set
        # - step_count
        
        self._saved_state = {
            'current_output': self._env.current_output.copy(),
            'has_copied': self._env.has_copied,
            'has_resized': self._env.has_resized,
            'painted_positions': self._env.painted_positions.copy(),
            'step_count': self._env.step_count,
            'action_history': [a.copy() for a in self._env.action_history],
        }
    
    def _restore_state(self):
        """Restore the saved environment state."""
        if self._saved_state is None:
            return
        
        self._env.current_output = self._saved_state['current_output'].copy()
        self._env.has_copied = self._saved_state['has_copied']
        self._env.has_resized = self._saved_state['has_resized']
        self._env.painted_positions = self._saved_state['painted_positions'].copy()
        self._env.step_count = self._saved_state['step_count']
        self._env.action_history = [a.copy() for a in self._saved_state['action_history']]
    
    def _perturb_action(self, action):
        """
        Create a slightly different version of the action to try next.
        This encourages exploration of alternative actions.
        """
        new_action = {k: v for k, v in action.items()}
        
        action_type = int(action['action_type'])
        
        if action_type == 0:  # Paint action
            # Try different position or color
            if np.random.random() < 0.5:
                # Try nearby position
                new_action['x'] = np.clip(action['x'] + np.random.randint(-2, 3), 0, 29)
                new_action['y'] = np.clip(action['y'] + np.random.randint(-2, 3), 0, 29)
            else:
                # Try different color
                new_action['color'] = np.random.randint(0, 10)
        
        elif action_type == 1:  # Copy action
            # Copy is a one-time action, try paint instead
            new_action['action_type'] = 0
            new_action['x'] = np.random.randint(0, 30)
            new_action['y'] = np.random.randint(0, 30)
            new_action['color'] = np.random.randint(0, 10)
        
        elif action_type == 2:  # Resize action
            # Try different dimensions
            new_action['width'] = np.clip(action['width'] + np.random.randint(-2, 3), 1, 30)
            new_action['height'] = np.clip(action['height'] + np.random.randint(-2, 3), 1, 30)
        
        # action_type == 3 (done) stays the same
        
        return new_action
    
    def close(self):
        return self._env.close()


class SimpleTrialErrorWrapper(embodied.Env):
    """
    Simpler version: Just tries the agent's action, and if it doesn't improve reward,
    it reverts and signals the agent to try again (by not incrementing episode).
    
    This version lets the agent learn which actions work rather than randomly perturbing.
    """
    
    def __init__(self, env, min_reward_improvement=0.001):
        """
        Args:
            env: The wrapped environment (ARC)
            min_reward_improvement: Minimum reward gain required to accept action
        """
        self._env = env
        self.min_reward_improvement = min_reward_improvement
        
        self._saved_state = None
        self._last_reward = 0.0
        
    @property
    def obs_space(self):
        return self._env.obs_space
    
    @property
    def act_space(self):
        return self._env.act_space
    
    def step(self, action):
        """
        Execute action. If reward doesn't improve enough, revert and let agent try again.
        """
        # Handle reset
        if action.get('reset', False):
            obs = self._env.step(action)
            self._last_reward = obs.get('reward', 0.0)
            self._save_state()
            return obs
        
        # Save state before action
        if self._saved_state is None:
            self._save_state()
        
        saved_state = self._save_state()
        
        # Execute action
        obs = self._env.step(action)
        current_reward = obs.get('reward', 0.0)
        reward_delta = current_reward - self._last_reward
        
        # Check if action was good enough
        if reward_delta >= self.min_reward_improvement or obs.get('is_last', False):
            # Accept this action
            self._last_reward = current_reward
            return obs
        else:
            # Reject this action - revert state
            self._restore_state(saved_state)
            
            # Return observation but modify flags to indicate this wasn't accepted
            # Don't mark as terminal so agent tries again
            obs['is_last'] = False
            obs['is_terminal'] = False
            obs['reward'] = np.float32(-0.1)  # Small penalty for failed attempt
            
            return obs
    
    def _save_state(self):
        """Save and return current state."""
        state = {
            'current_output': self._env.current_output.copy(),
            'has_copied': self._env.has_copied,
            'has_resized': self._env.has_resized,
            'painted_positions': self._env.painted_positions.copy(),
            'step_count': self._env.step_count,
            'action_history': [a.copy() for a in self._env.action_history],
        }
        self._saved_state = state
        return state
    
    def _restore_state(self, state):
        """Restore environment to saved state."""
        self._env.current_output = state['current_output'].copy()
        self._env.has_copied = state['has_copied']
        self._env.has_resized = state['has_resized']
        self._env.painted_positions = state['painted_positions'].copy()
        self._env.step_count = state['step_count']
        self._env.action_history = [a.copy() for a in state['action_history']]
    
    def close(self):
        return self._env.close()