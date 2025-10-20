#!/usr/bin/env python3
"""
Debug script to diagnose memory issues during ARC training initialization.
This script will track memory usage at each step to identify where OOM happens.
"""

import sys
import os
import time
import psutil
import traceback

# Add dreamerv3 to path
sys.path.insert(0, os.path.dirname(__file__))

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert to GB

def print_memory(step_name):
    """Print memory usage for a step"""
    mem_gb = get_memory_usage()
    print(f"[MEMORY] {step_name}: {mem_gb:.2f} GB")
    return mem_gb

def estimate_observation_size():
    """Estimate the size of a single ARC observation"""
    import numpy as np
    
    # Each observation has 6 images (5 training pairs + 1 test pair)
    # Each image is (64, 128, 3) uint8
    single_image_bytes = 64 * 128 * 3  # bytes
    total_images = 6
    obs_size_bytes = single_image_bytes * total_images
    
    print(f"\n[ESTIMATE] Single observation size:")
    print(f"  - Single image: {single_image_bytes / 1024:.2f} KB")
    print(f"  - Total (6 images): {obs_size_bytes / 1024:.2f} KB")
    print(f"  - Total: {obs_size_bytes / (1024**2):.4f} MB")
    
    return obs_size_bytes

def estimate_replay_size(replay_capacity, obs_size_bytes):
    """Estimate replay buffer memory usage"""
    # Replay stores multiple fields per step, but observation is the largest
    # Add ~50% overhead for other fields (actions, rewards, etc.)
    replay_bytes = replay_capacity * obs_size_bytes * 1.5
    replay_gb = replay_bytes / (1024 ** 3)
    
    print(f"\n[ESTIMATE] Replay buffer size:")
    print(f"  - Capacity: {int(replay_capacity):,} steps")
    print(f"  - Estimated RAM: {replay_gb:.2f} GB")
    
    return replay_gb

def check_system_resources():
    """Check available system resources"""
    import platform
    
    # Get total RAM
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    used_gb = mem.used / (1024 ** 3)
    percent_used = mem.percent
    
    print(f"\n[SYSTEM] Resources:")
    print(f"  - OS: {platform.system()} {platform.release()}")
    print(f"  - Total RAM: {total_gb:.2f} GB")
    print(f"  - Used RAM: {used_gb:.2f} GB ({percent_used:.1f}%)")
    print(f"  - Available RAM: {available_gb:.2f} GB")
    print(f"  - CPU cores: {psutil.cpu_count()}")
    
    return available_gb

def main():
    print("="*80)
    print("ARC Training Memory Diagnostic Tool")
    print("="*80)
    
    # Check system resources first
    available_gb = check_system_resources()
    
    # Estimate observation and replay sizes
    obs_size = estimate_observation_size()
    
    print_memory("Initial")
    
    try:
        # Import heavy modules
        print("\n[STEP 1] Importing modules...")
        print_memory("Before imports")
        
        import numpy as np
        import elements
        import embodied
        from dreamerv3 import agent as agent_module
        from dreamerv3 import configs as configs_module
        
        print_memory("After imports")
        
        # Load config
        print("\n[STEP 2] Loading config...")
        args = embodied.Config(configs_module.defaults, name='defaults')
        args = args.update(configs_module.configs['arc'])
        print(f"  - Replay size: {args.replay.size:,}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Batch length: {args.batch_length}")
        print(f"  - Model deter: {args.agent.dyn.rssm.deter}")
        print(f"  - Model hidden: {args.agent.dyn.rssm.hidden}")
        print(f"  - Model blocks: {args.agent.dyn.rssm.blocks}")
        
        estimate_replay_size(args.replay.size, obs_size)
        print_memory("After config")
        
        # Initialize environment
        print("\n[STEP 3] Creating environment...")
        from embodied.envs import arc
        env = arc.ARC(
            task='arc_training',
            puzzle_dir='./arc-data/',
            version='V2',
            split='training',
            length=100,
            size=64
        )
        print(f"  - Loaded {len(env.puzzles)} puzzles")
        print(f"  - Observation space keys: {list(env.obs_space.keys())}")
        print_memory("After environment")
        
        # Test environment step
        print("\n[STEP 4] Testing environment step...")
        obs = env.step({'reset': True, 'action_type': 0, 'x': 0, 'y': 0, 'color': 0})
        
        # Calculate actual observation size
        actual_obs_bytes = sum(v.nbytes for k, v in obs.items() if hasattr(v, 'nbytes'))
        print(f"  - Actual observation size: {actual_obs_bytes / 1024:.2f} KB")
        print_memory("After env step")
        
        # Initialize replay buffer
        print("\n[STEP 5] Creating replay buffer...")
        print(f"  - Capacity: {args.replay.size:,}")
        
        replay = embodied.replay.Replay(
            length=args.batch_length,
            capacity=args.replay.size,
            directory=None,  # Don't save to disk for testing
            chunksize=args.replay.chunksize,
        )
        print_memory("After replay creation")
        
        # Fill replay with some dummy data
        print("\n[STEP 6] Adding steps to replay...")
        for i in range(min(100, int(args.batch_size * args.batch_length))):
            replay.add(obs, worker=0)
            if i % 20 == 0:
                print(f"  - Added {i} steps")
                print_memory(f"After {i} steps")
        
        print_memory("After replay fill")
        
        # Initialize agent (this is often the memory hog)
        print("\n[STEP 7] Creating agent...")
        print("  This step may take a while and use significant memory...")
        
        import jax
        print(f"  - JAX platform: {args.jax.platform}")
        print(f"  - JAX devices: {jax.devices()}")
        
        mem_before_agent = get_memory_usage()
        
        from dreamerv3 import agent as dreamerv3_agent
        
        agent = dreamerv3_agent.Agent(
            env.obs_space,
            env.act_space,
            step=embodied.Counter(),
            config=args.agent,
        )
        
        mem_after_agent = get_memory_usage()
        agent_mem = mem_after_agent - mem_before_agent
        print(f"  - Agent memory usage: {agent_mem:.2f} GB")
        print_memory("After agent creation")
        
        # Try checkpoint save
        print("\n[STEP 8] Testing checkpoint save...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = elements.Checkpoint(elements.Path(tmpdir))
            cp.step = embodied.Counter()
            cp.agent = agent
            cp.replay = replay
            
            print("  - Saving checkpoint...")
            mem_before_save = get_memory_usage()
            cp.save()
            mem_after_save = get_memory_usage()
            save_mem = mem_after_save - mem_before_save
            print(f"  - Checkpoint save overhead: {save_mem:.2f} GB")
            print_memory("After checkpoint save")
        
        print("\n" + "="*80)
        print("SUCCESS! All steps completed without OOM.")
        print("="*80)
        
        final_mem = get_memory_usage()
        print(f"\nFinal memory usage: {final_mem:.2f} GB")
        print(f"Available RAM: {available_gb:.2f} GB")
        
        if final_mem > available_gb * 0.8:
            print("\n⚠️  WARNING: Using >80% of available RAM!")
            print("   Consider reducing model size or replay buffer further.")
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR occurred!")
        print("="*80)
        print(f"\nException: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        final_mem = get_memory_usage()
        print(f"\nMemory at error: {final_mem:.2f} GB")
        print(f"Available RAM: {available_gb:.2f} GB")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

