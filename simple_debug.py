#!/usr/bin/env python3
"""
Simple step-by-step memory test to isolate the issue.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def print_step(msg):
    print(f"\n{'='*60}")
    print(f"STEP: {msg}")
    print('='*60)
    sys.stdout.flush()

def get_mem():
    try:
        import psutil
        process = psutil.Process()
        mem_gb = process.memory_info().rss / (1024 ** 3)
        print(f"Memory: {mem_gb:.2f} GB")
        sys.stdout.flush()
        return mem_gb
    except:
        print("(psutil not available)")
        return 0

print_step("Starting")
get_mem()

print_step("Importing numpy")
import numpy as np
get_mem()

print_step("Importing elements")
import elements
get_mem()

print_step("Importing embodied")
import embodied
get_mem()

print_step("Loading config")
from dreamerv3 import configs as configs_module
args = embodied.Config(configs_module.defaults, name='defaults')
args = args.update(configs_module.configs['arc'])
print(f"Config loaded:")
print(f"  - replay.size: {args.replay.size}")
print(f"  - jax.prealloc: {args.jax.prealloc}")
print(f"  - jax.platform: {args.jax.platform}")
get_mem()

print_step("Creating ARC environment")
from embodied.envs import arc
env = arc.ARC(
    task='arc_training',
    puzzle_dir='./arc-data/',
    version='V2',
    split='training',
    length=100,
    size=64
)
print(f"Loaded {len(env.puzzles)} puzzles")
get_mem()

print_step("Testing environment reset")
obs = env.step({'reset': True, 'action_type': 0, 'x': 0, 'y': 0, 'color': 0})
obs_size_kb = sum(v.nbytes for k, v in obs.items() if hasattr(v, 'nbytes')) / 1024
print(f"Observation size: {obs_size_kb:.2f} KB")
get_mem()

print_step("BEFORE importing JAX")
get_mem()

print_step("Importing JAX (THIS MAY HANG OR CRASH)")
sys.stdout.flush()
import jax
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
get_mem()

print_step("Importing dreamerv3 agent")
from dreamerv3 import agent as dreamerv3_agent
get_mem()

print_step("Creating agent (THIS IS LIKELY WHERE IT CRASHES)")
sys.stdout.flush()

agent = dreamerv3_agent.Agent(
    env.obs_space,
    env.act_space,
    step=embodied.Counter(),
    config=args.agent,
)
print("Agent created!")
get_mem()

print("\n" + "="*60)
print("SUCCESS! All steps completed.")
print("="*60)

