#!/usr/bin/env python3
"""
ARC Training Web Interface
Run with: python app.py
Then open browser to http://localhost:5000
"""

import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Global state
training_process = None
training_thread = None
log_monitor_thread = None
is_training = False
metrics_data = {
    'steps': deque(maxlen=1000),
    'accuracies': deque(maxlen=1000),  # Track end-of-episode accuracy (0-100%)
    'rewards': deque(maxlen=1000),  # Track raw episode rewards
    'lengths': deque(maxlen=1000),
    'train_loss': deque(maxlen=1000),
    'timestamps': deque(maxlen=1000),
}
console_logs = deque(maxlen=500)  # Store last 500 lines of console output
current_logdir = None
training_start_time = None
latest_episode_grids = None  # Store the latest episode's grid visualization


def calculate_moving_average(data, window=10):
    """Calculate moving average for a sequence of values."""
    if len(data) < window:
        window = len(data)
    if window == 0:
        return []
    
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        window_data = list(data)[start_idx:i+1]
        result.append(sum(window_data) / len(window_data))
    return result


def monitor_console_output():
    """Monitor and capture console output from the training process."""
    global training_process, console_logs, is_training
    
    if training_process is None:
        return
    
    try:
        for line in iter(training_process.stdout.readline, ''):
            if not is_training:
                break
            
            if line:
                # Add timestamp to log line
                timestamp = datetime.now().strftime('%H:%M:%S')
                log_entry = f"[{timestamp}] {line.rstrip()}"
                console_logs.append(log_entry)
                print(log_entry)  # Also print to server console
    except Exception as e:
        print(f"Error monitoring console output: {e}")


def monitor_training():
    """Monitor training logs and extract metrics."""
    global is_training, current_logdir, metrics_data, latest_episode_grids
    
    # Wait for logdir to be created
    max_wait = 30  # seconds
    waited = 0
    while waited < max_wait and is_training:
        if current_logdir and Path(current_logdir).exists():
            break
        time.sleep(0.5)
        waited += 0.5
    
    if not current_logdir or not Path(current_logdir).exists():
        print(f"Warning: Could not find logdir: {current_logdir}")
        return
    
    metrics_file = Path(current_logdir) / 'metrics.jsonl'
    scores_file = Path(current_logdir) / 'scores.jsonl'
    
    # Episode grids are written to a fixed location by the environment
    grids_file = Path('latest_episode.json')
    last_grids_mtime = 0
    
    # Wait for metrics file to be created
    waited = 0
    while waited < max_wait and is_training:
        if metrics_file.exists():
            break
        time.sleep(0.5)
        waited += 0.5
    
    print(f"Monitoring metrics from: {metrics_file}")
    print(f"Monitoring episode grids from: {grids_file}")
    
    # Monitor the files
    last_position = 0
    last_scores_position = 0
    
    while is_training:
        try:
            # Read new metrics
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                    
                    for line in new_lines:
                        if line.strip():
                            try:
                                metric = json.loads(line)
                                
                                # DEBUG: Print available keys for first metric to help with debugging
                                if last_position == 0 and 'step' in metric:
                                    print(f"DEBUG: Available metric keys: {list(metric.keys())}")
                                
                                step = metric.get('step', 0)
                                
                                # Extract end-of-episode accuracy from final reward (0-1) -> percentage (0-100)
                                # Try multiple possible metric names for rewards
                                # 'episode/score' is the primary DreamerV3 metric for episode rewards
                                reward_keys = ['episode/score', 'episode/final_reward', 'episode/reward', 'reward', 'episode_reward']
                                reward_value = None
                                
                                for key in reward_keys:
                                    if key in metric:
                                        reward_value = float(metric[key])
                                        break
                                
                                if reward_value is not None:
                                    metrics_data['steps'].append(step)
                                    metrics_data['rewards'].append(reward_value)
                                    accuracy_percent = reward_value * 100.0
                                    # Clamp to [0, 100] for display stability
                                    accuracy_percent = max(0.0, min(100.0, accuracy_percent))
                                    metrics_data['accuracies'].append(accuracy_percent)
                                    metrics_data['timestamps'].append(time.time())
                                    print(f"DEBUG: Step {step}, Reward: {reward_value:.3f}, Accuracy: {accuracy_percent:.1f}%")
                                else:
                                    # Check if this looks like an episode metric but we couldn't find reward
                                    if any(key.startswith('episode') for key in metric.keys()):
                                        print(f"DEBUG: Found episode metric at step {step} but no reward key. Keys: {[k for k in metric.keys() if 'episode' in k.lower() or 'reward' in k.lower()]}")
                                
                                # Try multiple possible metric names for episode length
                                length_keys = ['episode/length', 'episode_length', 'length']
                                for key in length_keys:
                                    if key in metric:
                                        metrics_data['lengths'].append(metric[key])
                                        break
                                
                                # Extract training loss (may vary by model)
                                for key in metric:
                                    if 'loss' in key.lower() or 'train/' in key:
                                        if 'train_loss' not in metrics_data:
                                            metrics_data['train_loss'] = deque(maxlen=1000)
                                        metrics_data['train_loss'].append(metric[key])
                                        break
                                    
                            except json.JSONDecodeError:
                                continue
            
            # Read scores if available (this contains episode/score metric)
            if scores_file.exists():
                with open(scores_file, 'r') as f:
                    f.seek(last_scores_position)
                    new_lines = f.readlines()
                    last_scores_position = f.tell()
                    
                    for line in new_lines:
                        if line.strip():
                            try:
                                score_data = json.loads(line)
                                
                                # Process score data - usually just contains step and episode/score
                                if 'step' in score_data and 'episode/score' in score_data:
                                    step = score_data['step']
                                    reward_value = float(score_data['episode/score'])
                                    
                                    # Check if we already have data for this step from metrics.jsonl
                                    # If not, add it from scores.jsonl
                                    if not metrics_data['steps'] or step not in metrics_data['steps']:
                                        metrics_data['steps'].append(step)
                                        metrics_data['rewards'].append(reward_value)
                                        accuracy_percent = reward_value * 100.0
                                        accuracy_percent = max(0.0, min(100.0, accuracy_percent))
                                        metrics_data['accuracies'].append(accuracy_percent)
                                        metrics_data['timestamps'].append(time.time())
                                        print(f"DEBUG: (from scores.jsonl) Step {step}, Reward: {reward_value:.3f}, Accuracy: {accuracy_percent:.1f}%")
                                    
                            except json.JSONDecodeError:
                                continue
            
            # Check for updated episode grids (based on file modification time)
            if grids_file.exists():
                try:
                    current_mtime = grids_file.stat().st_mtime
                    if current_mtime > last_grids_mtime:
                        last_grids_mtime = current_mtime
                        with open(grids_file, 'r') as f:
                            grid_data = json.load(f)
                            latest_episode_grids = grid_data
                except Exception as e:
                    print(f"Error reading episode grids: {e}")
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Error monitoring metrics: {e}")
            time.sleep(1)


def start_training_process(config='arc', custom_args=''):
    """Start the training process."""
    global training_process, is_training, current_logdir, training_start_time, training_thread, log_monitor_thread, console_logs
    
    if is_training:
        return {'status': 'error', 'message': 'Training already running'}
    
    # Clear previous console logs
    console_logs.clear()
    
    # Create timestamp for logdir
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    base_logdir = Path.home() / 'logdir'
    current_logdir = str(base_logdir / timestamp)

    # Best-effort prune old run directories according to default config keep_last_runs
    try:
        # Default keep from configs.yaml run.keep_last_runs
        default_keep = 10
        keep_last_runs = default_keep
        # Allow override via custom_args e.g. --run.keep_last_runs=20
        for token in (custom_args or '').split():
            if token.startswith('--run.keep_last_runs='):
                try:
                    keep_last_runs = int(token.split('=', 1)[1])
                except ValueError:
                    pass
        if keep_last_runs >= 0 and base_logdir.exists():
            runs = [p for p in base_logdir.iterdir() if p.is_dir()]
            runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for old in runs[keep_last_runs:]:
                try:
                    # Only delete directories that look like timestamps YYYYMMDD-HHMMSS
                    if '-' in old.name and len(old.name) >= 15:
                        # Remove large ckpt subdirs first for speed
                        ckpt = old / 'ckpt'
                        if ckpt.exists():
                            import shutil
                            shutil.rmtree(ckpt, ignore_errors=True)
                        # Then remove the run dir
                        import shutil
                        shutil.rmtree(old, ignore_errors=True)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Build command
    cmd = [
        'python', 'dreamerv3/main.py',
        '--configs', config,
        '--logdir', current_logdir,
    ]
    
    # Add custom arguments if provided
    if custom_args:
        cmd.extend(custom_args.split())
    
    print(f"Starting training with command: {' '.join(cmd)}")
    print(f"Logdir: {current_logdir}")
    
    try:
        # Start the training process
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        is_training = True
        training_start_time = time.time()
        
        # Start console output monitoring thread
        log_monitor_thread = threading.Thread(target=monitor_console_output, daemon=True)
        log_monitor_thread.start()
        
        # Start metrics monitoring thread
        training_thread = threading.Thread(target=monitor_training, daemon=True)
        training_thread.start()
        
        return {
            'status': 'success',
            'message': 'Training started',
            'logdir': current_logdir
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def stop_training_process():
    """Stop the training process."""
    global training_process, is_training
    
    if not is_training or training_process is None:
        return {'status': 'error', 'message': 'No training running'}
    
    try:
        training_process.terminate()
        training_process.wait(timeout=10)
        is_training = False
        
        return {'status': 'success', 'message': 'Training stopped'}
    except Exception as e:
        # Force kill if terminate doesn't work
        try:
            training_process.kill()
            is_training = False
            return {'status': 'success', 'message': 'Training force stopped'}
        except:
            return {'status': 'error', 'message': str(e)}


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def api_start():
    """API endpoint to start training."""
    data = request.json or {}
    config = data.get('config', 'arc')
    custom_args = data.get('custom_args', '')
    
    result = start_training_process(config, custom_args)
    return jsonify(result)


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """API endpoint to stop training."""
    result = stop_training_process()
    return jsonify(result)


@app.route('/api/status', methods=['GET'])
def api_status():
    """API endpoint to get training status."""
    status_data = {
        'is_training': is_training,
        'logdir': current_logdir,
        'num_datapoints': len(metrics_data['steps']),
        'num_log_lines': len(console_logs),
    }
    
    if training_start_time:
        status_data['elapsed_time'] = time.time() - training_start_time
    
    return jsonify(status_data)


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API endpoint to get training metrics."""
    window = request.args.get('window', 10, type=int)
    
    # Convert deques to lists
    steps = list(metrics_data['steps'])
    accuracies = list(metrics_data['accuracies'])
    rewards = list(metrics_data.get('rewards', []))
    lengths = list(metrics_data['lengths'])
    train_loss = list(metrics_data.get('train_loss', []))
    
    # Calculate moving averages
    accuracies_ma = calculate_moving_average(accuracies, window)
    rewards_ma = calculate_moving_average(rewards, window)
    lengths_ma = calculate_moving_average(lengths, window)
    train_loss_ma = calculate_moving_average(train_loss, window) if train_loss else []
    
    # Get latest values
    latest = {
        'step': steps[-1] if steps else 0,
        'accuracy': accuracies[-1] if accuracies else 0,
        'accuracy_ma': accuracies_ma[-1] if accuracies_ma else 0,
        'reward': rewards[-1] if rewards else 0,
        'reward_ma': rewards_ma[-1] if rewards_ma else 0,
        'length': lengths[-1] if lengths else 0,
        'length_ma': lengths_ma[-1] if lengths_ma else 0,
        'train_loss': train_loss[-1] if train_loss else 0,
        'train_loss_ma': train_loss_ma[-1] if train_loss_ma else 0,
    }
    
    return jsonify({
        'steps': steps,
        'accuracies': accuracies,
        'accuracies_ma': accuracies_ma,
        'rewards': rewards,
        'rewards_ma': rewards_ma,
        'lengths': lengths,
        'lengths_ma': lengths_ma,
        'train_loss': train_loss,
        'train_loss_ma': train_loss_ma,
        'latest': latest,
    })


@app.route('/api/logs', methods=['GET'])
def api_logs():
    """API endpoint to get console logs."""
    # Get optional parameters
    lines = request.args.get('lines', None, type=int)
    
    logs = list(console_logs)
    
    # Return only last N lines if specified
    if lines is not None and lines > 0:
        logs = logs[-lines:]
    
    return jsonify({
        'logs': logs,
        'total_lines': len(console_logs)
    })


@app.route('/api/grids', methods=['GET'])
def api_grids():
    """API endpoint to get the latest episode grid visualization."""
    global latest_episode_grids
    
    if latest_episode_grids is None:
        return jsonify({
            'status': 'no_data',
            'message': 'No episode data available yet'
        })
    
    return jsonify({
        'status': 'success',
        'data': latest_episode_grids
    })


@app.route('/api/clear', methods=['POST'])
def api_clear():
    """API endpoint to clear metrics data."""
    global metrics_data, console_logs, latest_episode_grids
    
    metrics_data = {
        'steps': deque(maxlen=1000),
        'accuracies': deque(maxlen=1000),
        'rewards': deque(maxlen=1000),
        'lengths': deque(maxlen=1000),
        'train_loss': deque(maxlen=1000),
        'timestamps': deque(maxlen=1000),
    }
    console_logs.clear()
    latest_episode_grids = None
    
    return jsonify({'status': 'success', 'message': 'Metrics and logs cleared'})


if __name__ == '__main__':
    print("=" * 60)
    print("ARC Training Web Interface")
    print("=" * 60)
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    # Ensure templates and static directories exist
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)