import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


def compute_color_target(target_pair):
    """
    Extract which colors appear in the target output.
    
    Args:
        target_pair: (B, T, H, W*2, 3) RGB image where right half is the target output
    
    Returns:
        target_dist: (B, T, 10) - probability distribution over colors 0-9
    """
    # Extract right half (target output)
    B, T, H, W2, C = target_pair.shape
    W = W2 // 2
    target_output = target_pair[:, :, :, W:, :]  # (B, T, H, W, 3)
    
    # Convert RGB to color indices (0-9)
    # Define ARC color palette as array
    color_palette = jnp.array([
        [0, 0, 0],           # 0: Black
        [0, 116, 217],       # 1: Blue
        [255, 65, 54],       # 2: Red
        [46, 204, 64],       # 3: Green
        [255, 220, 0],       # 4: Yellow
        [170, 170, 170],     # 5: Gray
        [240, 18, 190],      # 6: Magenta
        [255, 133, 27],      # 7: Orange
        [127, 219, 255],     # 8: Light Blue
        [135, 12, 37],       # 9: Maroon
    ], dtype=jnp.float32)
    
    # Compute distance to each color for each pixel
    target_flat = target_output.reshape(B, T, -1, 3)  # (B, T, H*W, 3)
    # Compute L2 distance: (B, T, H*W, 3) vs (10, 3) -> (B, T, H*W, 10)
    distances = jnp.sum((target_flat[:, :, :, None, :] - color_palette[None, None, None, :, :]) ** 2, axis=-1)
    color_indices = jnp.argmin(distances, axis=-1)  # (B, T, H*W)
    
    # Create binary mask: which colors appear at least once
    color_mask = jnp.zeros((B, T, 10), dtype=jnp.float32)
    for color_idx in range(10):
        # Check if this color appears anywhere in the image
        appears = (color_indices == color_idx).any(axis=-1)  # (B, T)
        color_mask = color_mask.at[:, :, color_idx].set(appears.astype(jnp.float32))
    
    # Convert to uniform distribution over present colors
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    color_sum = color_mask.sum(axis=-1, keepdims=True) + eps
    target_dist = color_mask / color_sum
    
    return target_dist


def ordinal_loss(predicted_dist, target_class, num_classes):
    """
    Compute ordinal loss that penalizes predictions proportionally to distance from target.
    
    This loss encourages the model to learn that classes are ordered (e.g., predicting 4 
    when target is 5 is better than predicting 0 when target is 5).
    
    Loss = sum_i P(class_i) * |i - target|
    
    Args:
        predicted_dist: Distribution object with logits or probs
        target_class: (B, T) integer target class indices
        num_classes: int, number of classes (e.g., 5 for counts 0-4)
    
    Returns:
        loss: (B, T) scalar loss per sample
    """
    # Get predicted probabilities
    if hasattr(predicted_dist, 'probs_parameter'):
        # TensorFlow Probability style
        probs = predicted_dist.probs_parameter()
    elif hasattr(predicted_dist, 'probs'):
        probs = predicted_dist.probs
    elif hasattr(predicted_dist, 'dist'):
        # Nested distribution
        if hasattr(predicted_dist.dist, 'probs_parameter'):
            probs = predicted_dist.dist.probs_parameter()
        elif hasattr(predicted_dist.dist, 'logits'):
            probs = jax.nn.softmax(predicted_dist.dist.logits)
        else:
            probs = predicted_dist.dist.probs
    elif hasattr(predicted_dist, 'logits'):
        probs = jax.nn.softmax(predicted_dist.logits)
    else:
        raise ValueError(f"Cannot extract probabilities from distribution: {type(predicted_dist)}")
    
    # Create distance matrix: |i - target| for each class i
    # Shape: (B, T, num_classes)
    class_indices = jnp.arange(num_classes, dtype=jnp.float32)[None, None, :]  # (1, 1, num_classes)
    target_expanded = target_class[:, :, None].astype(jnp.float32)  # (B, T, 1)
    distances = jnp.abs(class_indices - target_expanded)  # (B, T, num_classes)
    
    # Compute weighted sum: sum_i P(i) * |i - target|
    ordinal_loss_value = (probs * distances).sum(axis=-1)  # (B, T)
    
    return ordinal_loss_value


def compute_color_count_target(target_pair, test_grid_height, test_grid_width):
    """
    Extract how many pixels of each color appear in the target output.
    
    Args:
        target_pair: (B, T, H, W*2, 3) RGB image where right half is the target output
        test_grid_height: (B, T) actual grid height (1-30)
        test_grid_width: (B, T) actual grid width (1-30)
    
    Returns:
        color_counts: (B, T, 10) - count of each color (0-9), normalized to [0, 4]
    """
    # Extract right half (target output)
    B, T, H, W2, C = target_pair.shape
    W = W2 // 2
    target_output = target_pair[:, :, :, W:, :]  # (B, T, H, W, 3)
    
    # Define ARC color palette as array
    color_palette = jnp.array([
        [0, 0, 0],           # 0: Black
        [0, 116, 217],       # 1: Blue
        [255, 65, 54],       # 2: Red
        [46, 204, 64],       # 3: Green
        [255, 220, 0],       # 4: Yellow
        [170, 170, 170],     # 5: Gray
        [240, 18, 190],      # 6: Magenta
        [255, 133, 27],      # 7: Orange
        [127, 219, 255],     # 8: Light Blue
        [135, 12, 37],       # 9: Maroon
    ], dtype=jnp.float32)
    
    # Extract the valid grid region (first test_grid_height x test_grid_width pixels)
    grid_colors = target_output[:, :, :30, :30, :]  # (B, T, 30, 30, 3)
    
    # Create valid region mask based on grid dimensions
    y_coords = jnp.arange(30)[None, None, :, None]  # (1, 1, 30, 1)
    x_coords = jnp.arange(30)[None, None, None, :]  # (1, 1, 1, 30)
    height = test_grid_height[:, :, None, None]  # (B, T, 1, 1)
    width = test_grid_width[:, :, None, None]     # (B, T, 1, 1)
    valid_mask = ((y_coords < height) & (x_coords < width)).astype(jnp.float32)  # (B, T, 30, 30)
    
    # Compute distance to each color for each pixel in the valid region
    # grid_colors: (B, T, 30, 30, 3), color_palette: (10, 3)
    # Compute L2 distance: (B, T, 30, 30, 3) vs (10, 3) -> (B, T, 30, 30, 10)
    distances = jnp.sum(
        (grid_colors[:, :, :, :, None, :] - color_palette[None, None, None, None, :, :]) ** 2, 
        axis=-1
    )  # (B, T, 30, 30, 10)
    color_indices = jnp.argmin(distances, axis=-1)  # (B, T, 30, 30) - best matching color for each pixel
    
    # Count occurrences of each color in the valid region
    color_counts = jnp.zeros((B, T, 10), dtype=jnp.float32)
    for color_idx in range(10):
        # Count pixels where this color appears AND the pixel is in valid region
        is_color = (color_indices == color_idx).astype(jnp.float32)  # (B, T, 30, 30)
        count = (is_color * valid_mask).sum(axis=(2, 3))  # (B, T)
        color_counts = color_counts.at[:, :, color_idx].set(count)
    
    # Clip counts to [0, 4] range for categorical prediction
    # (most ARC grids are smaller, and this gives reasonable discretization)
    color_counts = jnp.clip(color_counts, 0, 4)
    
    return color_counts


def compute_position_targets(target_pair, test_grid_height, test_grid_width):
    """
    Extract which positions need to be painted in the target.
    
    Args:
        target_pair: (B, T, H, W*2, 3) RGB image where right half is target output
        test_grid_height: (B, T) actual grid height (1-30)
        test_grid_width: (B, T) actual grid width (1-30)
    
    Returns:
        x_target: (B, T, 30) - probability distribution over x coordinates
        y_target: (B, T, 30) - probability distribution over y coordinates
    """
    B, T, H, W2, C = target_pair.shape
    W = W2 // 2
    
    # Extract target output (right half)
    target_output = target_pair[:, :, :, W:, :]  # (B, T, H, W, 3)
    
    # The grid is in the top-left corner of the padded space
    # Create mask of valid positions based on actual grid dimensions
    y_coords = jnp.arange(30)[None, None, :, None]  # (1, 1, 30, 1)
    x_coords = jnp.arange(30)[None, None, None, :]  # (1, 1, 1, 30)
    
    # Expand grid dimensions for broadcasting
    height = test_grid_height[:, :, None, None]  # (B, T, 1, 1)
    width = test_grid_width[:, :, None, None]     # (B, T, 1, 1)
    
    # Create valid region mask
    valid_mask = ((y_coords < height) & (x_coords < width)).astype(jnp.float32)  # (B, T, 30, 30)
    
    # Sample the output image at grid positions
    # We need to map grid coordinates [0, 29] to pixel coordinates
    # Since the image is padded, grid position i corresponds to pixel i
    pixel_y = jnp.arange(30)
    pixel_x = jnp.arange(30)
    
    # Extract colors at grid positions (first 30x30 pixels of target)
    grid_colors = target_output[:, :, :30, :30, :]  # (B, T, 30, 30, 3)
    
    # Check if position is non-black (has content to paint)
    # A position needs painting if it's not black [0,0,0]
    is_nonblack = (grid_colors.sum(axis=-1) > 10).astype(jnp.float32)  # (B, T, 30, 30)
    
    # Combine: position is relevant if it's both valid and has content
    painting_target = valid_mask * is_nonblack  # (B, T, 30, 30)
    
    # Marginalize to get per-coordinate distributions
    eps = 1e-8
    x_sum = painting_target.sum(axis=2) + eps  # (B, T, 30)
    y_sum = painting_target.sum(axis=3) + eps  # (B, T, 30)
    
    # Total positions to normalize
    total = painting_target.sum(axis=(2, 3), keepdims=True) + eps
    
    x_target = x_sum / total.squeeze(axis=2)  # (B, T, 30)
    y_target = y_sum / total.squeeze(axis=3)  # (B, T, 30)
    
    return x_target, y_target


def compute_position_targets_for_color(target_pair, test_grid_height, test_grid_width, color_idx):
    """
    Extract which positions need to be painted with a SPECIFIC color.
    
    Args:
        target_pair: (B, T, H, W*2, 3) RGB image where right half is target output
        test_grid_height: (B, T) actual grid height (1-30)
        test_grid_width: (B, T) actual grid width (1-30)
        color_idx: int (0-9) - which color to look for
    
    Returns:
        x_target: (B, T, 30) - probability distribution over x coordinates where this color appears
        y_target: (B, T, 30) - probability distribution over y coordinates where this color appears
    """
    B, T, H, W2, C = target_pair.shape
    W = W2 // 2
    
    # Extract target output (right half)
    target_output = target_pair[:, :, :, W:, :]  # (B, T, H, W, 3)
    
    # ARC color palette
    color_palette = jnp.array([
        [0, 0, 0],           # 0: Black
        [0, 116, 217],       # 1: Blue
        [255, 65, 54],       # 2: Red
        [46, 204, 64],       # 3: Green
        [255, 220, 0],       # 4: Yellow
        [170, 170, 170],     # 5: Gray
        [240, 18, 190],      # 6: Magenta
        [255, 133, 27],      # 7: Orange
        [127, 219, 255],     # 8: Light Blue
        [135, 12, 37],       # 9: Maroon
    ], dtype=jnp.float32)
    
    # Extract colors at grid positions (first 30x30 pixels of target)
    grid_colors = target_output[:, :, :30, :30, :]  # (B, T, 30, 30, 3)
    
    # Convert RGB to color indices at each position
    # Compute L2 distance to target color
    target_color = color_palette[color_idx]  # (3,)
    distances = jnp.sum((grid_colors - target_color[None, None, None, None, :]) ** 2, axis=-1)  # (B, T, 30, 30)
    
    # Also compute distance to all colors to get best match
    all_distances = jnp.sum(
        (grid_colors[:, :, :, :, None, :] - color_palette[None, None, None, None, :, :]) ** 2, 
        axis=-1
    )  # (B, T, 30, 30, 10)
    best_color = jnp.argmin(all_distances, axis=-1)  # (B, T, 30, 30)
    
    # Position has this color if best match is this color_idx
    has_color = (best_color == color_idx).astype(jnp.float32)  # (B, T, 30, 30)
    
    # Create valid region mask based on grid dimensions
    y_coords = jnp.arange(30)[None, None, :, None]  # (1, 1, 30, 1)
    x_coords = jnp.arange(30)[None, None, None, :]  # (1, 1, 1, 30)
    height = test_grid_height[:, :, None, None]  # (B, T, 1, 1)
    width = test_grid_width[:, :, None, None]     # (B, T, 1, 1)
    valid_mask = ((y_coords < height) & (x_coords < width)).astype(jnp.float32)  # (B, T, 30, 30)
    
    # Combine: position is relevant if it's valid AND has this color
    color_positions = valid_mask * has_color  # (B, T, 30, 30)
    
    # Marginalize to get per-coordinate distributions
    eps = 1e-8
    x_sum = color_positions.sum(axis=2) + eps  # (B, T, 30)
    y_sum = color_positions.sum(axis=3) + eps  # (B, T, 30)
    
    # Total positions with this color
    total = color_positions.sum(axis=(2, 3), keepdims=True) + eps
    
    x_target = x_sum / total.squeeze(axis=2)  # (B, T, 30)
    y_target = y_sum / total.squeeze(axis=3)  # (B, T, 30)
    
    return x_target, y_target


def compute_position_heatmap_for_color(target_pair, test_grid_height, test_grid_width, color_idx):
    """
    Extract joint position heatmap showing where a SPECIFIC color should be painted.
    This is the JOINT distribution p(x,y) rather than separate marginals p(x) and p(y).
    
    Args:
        target_pair: (B, T, H, W*2, 3) RGB image where right half is target output
        test_grid_height: (B, T) actual grid height (1-30)
        test_grid_width: (B, T) actual grid width (1-30)
        color_idx: int (0-9) - which color to look for
    
    Returns:
        position_heatmap: (B, T, 900) - probability distribution over 900 positions (30×30 flattened)
    """
    B, T, H, W2, C = target_pair.shape
    W = W2 // 2
    
    # Extract target output (right half)
    target_output = target_pair[:, :, :, W:, :]  # (B, T, H, W, 3)
    
    # ARC color palette
    color_palette = jnp.array([
        [0, 0, 0],           # 0: Black
        [0, 116, 217],       # 1: Blue
        [255, 65, 54],       # 2: Red
        [46, 204, 64],       # 3: Green
        [255, 220, 0],       # 4: Yellow
        [170, 170, 170],     # 5: Gray
        [240, 18, 190],      # 6: Magenta
        [255, 133, 27],      # 7: Orange
        [127, 219, 255],     # 8: Light Blue
        [135, 12, 37],       # 9: Maroon
    ], dtype=jnp.float32)
    
    # Extract colors at grid positions (first 30x30 pixels of target)
    grid_colors = target_output[:, :, :30, :30, :]  # (B, T, 30, 30, 3)
    
    # Convert RGB to color indices at each position
    all_distances = jnp.sum(
        (grid_colors[:, :, :, :, None, :] - color_palette[None, None, None, None, :, :]) ** 2, 
        axis=-1
    )  # (B, T, 30, 30, 10)
    best_color = jnp.argmin(all_distances, axis=-1)  # (B, T, 30, 30)
    
    # Position has this color if best match is this color_idx
    has_color = (best_color == color_idx).astype(jnp.float32)  # (B, T, 30, 30)
    
    # Create valid region mask based on grid dimensions
    y_coords = jnp.arange(30)[None, None, :, None]  # (1, 1, 30, 1)
    x_coords = jnp.arange(30)[None, None, None, :]  # (1, 1, 1, 30)
    height = test_grid_height[:, :, None, None]  # (B, T, 1, 1)
    width = test_grid_width[:, :, None, None]     # (B, T, 1, 1)
    valid_mask = ((y_coords < height) & (x_coords < width)).astype(jnp.float32)  # (B, T, 30, 30)
    
    # Combine: position is relevant if it's valid AND has this color
    color_positions = valid_mask * has_color  # (B, T, 30, 30)
    
    # Normalize to probability distribution
    eps = 1e-8
    total = color_positions.sum(axis=(2, 3), keepdims=True) + eps
    position_heatmap = color_positions / total  # (B, T, 30, 30)
    
    # Flatten to (B, T, 900)
    position_heatmap = position_heatmap.reshape((B, T, 900))
    
    return position_heatmap



class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward', 'valid_actions', 'valid_positions', 'valid_colors')
    
    # Encoder sees test_pair (current state) but NOT target_pair (ground truth)
    # The model must learn to predict the correct answer from context alone
    enc_space = {k: v for k, v in obs_space.items() 
                 if k not in exclude and k != 'target_pair'}
    
    # Decoder reconstructs BOTH test_pair (what it sees) and target_pair (what it should predict)
    # Loss on test_pair: standard reconstruction of observed state
    # Loss on target_pair: prediction of correct solution without having seen it as input
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    # Selection heads: discrete distributions conditioned on feat2tensor
    # to select spatial positions, sizes, and colors.
    # Position is now a joint 30×30 = 900 class distribution (heatmap)
    sel_spaces = {
        'position': elements.Space(np.int32, (), 0, 900),  # 30×30 flattened
        'width': elements.Space(np.int32, (), 0, 30),
        'height': elements.Space(np.int32, (), 0, 30),
        'color': elements.Space(np.int32, (), 0, 10),
        # Color count heads: one per color, predicting count in range [0, 4]
        'count_0': elements.Space(np.int32, (), 0, 10),  # Black count
        'count_1': elements.Space(np.int32, (), 0, 10),  # Blue count
        'count_2': elements.Space(np.int32, (), 0, 10),  # Red count
        'count_3': elements.Space(np.int32, (), 0, 10),  # Green count
        'count_4': elements.Space(np.int32, (), 0, 10),  # Yellow count
        'count_5': elements.Space(np.int32, (), 0, 10),  # Gray count
        'count_6': elements.Space(np.int32, (), 0, 10),  # Magenta count
        'count_7': elements.Space(np.int32, (), 0, 10),  # Orange count
        'count_8': elements.Space(np.int32, (), 0, 10),  # Light Blue count
        'count_9': elements.Space(np.int32, (), 0, 10),  # Maroon count
    }
    sel_outs = {k: d1 for k in sel_spaces.keys()}  # categorical for all
    self.sel = embodied.jax.MLPHead(sel_spaces, sel_outs, **config.policy, name='sel')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.sel, self.val]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    # Add scales for supervised selection heads if not already present
    if 'sel_width' not in scales:
      scales['sel_width'] = 1.0
    if 'sel_height' not in scales:
      scales['sel_height'] = 1.0
    if 'sel_color' not in scales:
      scales['sel_color'] = 1.0
    if 'sel_position' not in scales:
      scales['sel_position'] = 1.0
    # Add scales for color count heads
    for i in range(10):
      if f'sel_count_{i}' not in scales:
        scales[f'sel_count_{i}'] = 1.0
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol|sel)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    # Merge in selection heads derived from features
    sel = self.sel(self.feat2tensor(feat), bdims=1)
    # Only merge action-relevant heads (position, width, height, color)
    # Count heads are auxiliary predictions, not actions
    action_heads = {k: v for k, v in sel.items() if not k.startswith('count_')}
    policy.update(action_heads)

    # Apply action_type mask if provided
    if 'valid_actions' in obs and 'action_type' in policy:
      mask = f32(obs['valid_actions'])  # (B, 4)
      dist = policy['action_type']
      # Get logits from either Categorical or OneHot(Categorical)
      if hasattr(dist, 'dist'):
        logits = dist.dist.logits
      else:
        logits = dist.logits
      neg_inf = jnp.full_like(logits, -1e9)
      masked_logits = jnp.where(mask == 1, logits, neg_inf)
      if hasattr(dist, 'dist'):
        policy['action_type'].dist.logits = masked_logits
      else:
        policy['action_type'].logits = masked_logits
    
    # Apply color mask if provided (disable black color)
    if 'valid_colors' in obs and 'color' in policy:
      color_mask = f32(obs['valid_colors'])  # (B, 10)
      color_dist = policy['color']
      # Get logits from either Categorical or OneHot(Categorical)
      if hasattr(color_dist, 'dist'):
        color_logits = color_dist.dist.logits
      else:
        color_logits = color_dist.logits
      color_neg_inf = jnp.full_like(color_logits, -1e9)
      masked_color_logits = jnp.where(color_mask == 1, color_logits, color_neg_inf)
      if hasattr(color_dist, 'dist'):
        policy['color'].dist.logits = masked_color_logits
      else:
        policy['color'].logits = masked_color_logits

    # NEW: Apply spatial mask to position heatmap
    if 'valid_positions' in obs and 'position' in policy:
        spatial_mask = f32(obs['valid_positions'])  # (B, 30, 30)
        # Flatten to (B, 900) to match position distribution
        position_mask = spatial_mask.reshape((-1, 900))  # (B, 900)
        
        position_dist = policy['position']
        position_logits = position_dist.dist.logits if hasattr(position_dist, 'dist') else position_dist.logits
        position_neg_inf = jnp.full_like(position_logits, -1e9)
        masked_position_logits = jnp.where(position_mask == 1.0, position_logits, position_neg_inf)
        if hasattr(position_dist, 'dist'):
            policy['position'].dist.logits = masked_position_logits
        else:
            policy['position'].logits = masked_position_logits


    act = sample(policy)
    
    # Convert position index back to x,y coordinates for the environment
    # Remove 'position' from dict since environment only expects x,y
    if 'position' in act:
      position_idx = i32(act.pop('position'))  # (B,) values in [0, 899], remove from dict
      # Convert flat index to (y, x) coordinates: position = y * 30 + x
      act['y'] = position_idx // 30  # row
      act['x'] = position_idx % 30   # column
    
    # KEEP count predictions in actions - they're needed by the environment for constrained painting
    # The environment will use these to determine how many times to paint with each color
    # for i in range(10):
    #   act.pop(f'count_{i}', None)
    
    out = {}

    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    # Supervised selection losses for grid size (width/height)
    # Use world model features to predict target test grid dimensions.
    sel_pred = self.sel(self.feat2tensor(repfeat), bdims=2)
    if 'test_grid_width' in obs:
      target_w = jnp.clip((obs['test_grid_width']).astype(i32) - 1, 0, 29)
      losses['sel_width'] = sel_pred['width'].loss(target_w)
    if 'test_grid_height' in obs:
      target_h = jnp.clip((obs['test_grid_height']).astype(i32) - 1, 0, 29)
      losses['sel_height'] = sel_pred['height'].loss(target_h)
    
    # Supervised losses for color and position selection heads
    # These supervise the DISTRIBUTION SUPPORT rather than specific actions
    if 'target_pair' in obs:
      # Extract target distributions from ground truth
      color_target = compute_color_target(obs['target_pair'])  # (B, T, 10)
      color_counts_target = compute_color_count_target(
          obs['target_pair'], 
          obs['test_grid_height'], 
          obs['test_grid_width']
      )  # (B, T, 10) - counts for each color
      
      # Get predicted distributions (logits)
      color_logits = sel_pred['color'].dist.logits if hasattr(sel_pred['color'], 'dist') else sel_pred['color'].logits
      position_logits = sel_pred['position'].dist.logits if hasattr(sel_pred['position'], 'dist') else sel_pred['position'].logits
      
      # Convert to probabilities
      color_probs = jax.nn.softmax(color_logits)
      position_probs = jax.nn.softmax(position_logits)
      
      # Supervise color distribution (unconditional)
      eps = 1e-8
      losses['sel_color'] = -(color_target * jnp.log(color_probs + eps)).sum(axis=-1)
      
      # Supervise color count heads
      # For each color, predict its count as a categorical distribution over [0, 4]
      for color_idx in range(10):
        count_key = f'count_{color_idx}'
        # Target count for this color (clipped to [0, 4])
        target_count = color_counts_target[:, :, color_idx].astype(jnp.int32)  # (B, T)
        target_count = jnp.clip(target_count, 0, 4)  # Ensure in valid range
        
        # Predicted distribution for this color's count
        count_dist = sel_pred[count_key]
        
        # Ordinal loss: penalizes predictions proportionally to distance from target
        # This makes the model learn that 4 is closer to 5 than 0 is to 5
        losses[f'sel_{count_key}'] = ordinal_loss(count_dist, target_count, num_classes=10)
      
      # NEW: Color-conditional position supervision using joint heatmap
      # Compute position heatmap targets for each color (10 colors)
      color_conditional_positions = []
      for color_idx in range(10):
        pos_heatmap = compute_position_heatmap_for_color(
            obs['target_pair'], 
            obs['test_grid_height'], 
            obs['test_grid_width'],
            color_idx
        )  # (B, T, 900)
        color_conditional_positions.append(pos_heatmap)
      
      # Stack to get (B, T, 10, 900)
      color_conditional_positions = jnp.stack(color_conditional_positions, axis=2)  # (B, T, 10, 900)
      
      # Get the selected color from previous actions
      selected_color = obs['current_color']  # (B, T)
      
      # Index by selected color to get conditional targets (B, T, 900)
      batch_indices = jnp.arange(B)[:, None]
      time_indices = jnp.arange(T)[None, :]
      position_target_conditional = color_conditional_positions[batch_indices, time_indices, selected_color]
      
      # Supervise position with color-conditional heatmap targets
      # This teaches: "Given you picked color X, paint at positions where X appears in target"
      losses['sel_position'] = -(position_target_conditional * jnp.log(position_probs + eps)).sum(axis=-1)


    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    def policyfn(feat):
      inp = self.feat2tensor(feat)
      base = self.pol(inp, 1)
      selp = self.sel(inp, 1)
      # Only merge action-relevant heads, not count predictions
      action_selp = {k: v for k, v in selp.items() if not k.startswith('count_')}
      merged = {**base, **action_selp}
      return sample(merged)
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    # Use merged policy (base + selection heads) for imagination loss
    base_pol = self.pol(inp, 2)
    sel_pol = self.sel(inp, 2)
    # Only merge action-relevant heads, not count predictions
    action_sel_pol = {k: v for k, v in sel_pol.items() if not k.startswith('count_')}
    merged_pol = {**base_pol, **action_sel_pol}
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        merged_pol,
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      # Create border with the actual video time dimension
      VT = video.shape[1]  # Use video's actual time dimension
      border = jnp.full((VT, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[VT // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  # Gate contributions by action_type for selection heads.
  # action_type: 0=paint -> use x,y; 1=resize -> use width,height; 3=set_color -> use color
  atype = sg(act['action_type'])[:, :-1]
  paint_mask = (atype == 0)
  resize_mask = (atype == 1)
  color_mask = (atype == 3)

  masked_logpis = []
  masked_ents = []
  for k, dist in policy.items():
    lp = dist.logp(sg(act[k]))[:, :-1]
    ent = dist.entropy()[:, :-1]
    if k == 'position':
      m = paint_mask
    elif k in ('width', 'height'):
      m = resize_mask
    elif k == 'color':
      m = color_mask
    elif k.startswith('count_'):
      # Count heads are supervised only, not real actions - exclude from policy loss
      continue
    else:
      # Always include other heads (e.g., action_type, reset)
      m = jnp.ones_like(lp, dtype=bool)
    m = m.astype(lp.dtype)
    masked_logpis.append(lp * m)
    masked_ents.append(ent * m)
  logpi = sum(masked_logpis)
  ent_sum = sum(masked_ents)
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * ent_sum)
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k, dist in policy.items():
    ent = dist.entropy()[:, :-1]
    if k == 'position':
      m = paint_mask
    elif k in ('width', 'height'):
      m = resize_mask
    elif k == 'color':
      m = color_mask
    elif k.startswith('count_'):
      # Count heads are supervised only, skip entropy metrics
      continue
    else:
      m = jnp.ones_like(ent, dtype=bool)
    m = m.astype(ent.dtype)
    entm = (ent * m).mean()
    metrics[f'ent/{k}'] = entm
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (entm - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)