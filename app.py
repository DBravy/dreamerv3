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
is_training = False
metrics_data = {
    'steps': deque(maxlen=1000),
    'scores': deque(maxlen=1000),
    'lengths': deque(maxlen=1000),
    'train_loss': deque(maxlen=1000),
    'fps': deque(maxlen=1000),
    'timestamps': deque(maxlen=1000),
}
current_logdir = None
training_start_time = None


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


def monitor_training():
    """Monitor training logs and extract metrics."""
    global is_training, current_logdir, metrics_data
    
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
    
    # Wait for metrics file to be created
    waited = 0
    while waited < max_wait and is_training:
        if metrics_file.exists():
            break
        time.sleep(0.5)
        waited += 0.5
    
    print(f"Monitoring metrics from: {metrics_file}")
    
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
                                step = metric.get('step', 0)
                                
                                # Extract various metrics
                                if 'episode/score' in metric:
                                    metrics_data['steps'].append(step)
                                    metrics_data['scores'].append(metric['episode/score'])
                                    metrics_data['timestamps'].append(time.time())
                                
                                if 'episode/length' in metric:
                                    metrics_data['lengths'].append(metric['episode/length'])
                                
                                # Extract training loss (may vary by model)
                                for key in metric:
                                    if 'loss' in key.lower() or 'train/' in key:
                                        if 'train_loss' not in metrics_data:
                                            metrics_data['train_loss'] = deque(maxlen=1000)
                                        metrics_data['train_loss'].append(metric[key])
                                        break
                                
                                if 'fps/policy' in metric:
                                    metrics_data['fps'].append(metric['fps/policy'])
                                    
                            except json.JSONDecodeError:
                                continue
            
            # Read scores if available
            if scores_file.exists():
                with open(scores_file, 'r') as f:
                    f.seek(last_scores_position)
                    new_lines = f.readlines()
                    last_scores_position = f.tell()
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Error monitoring metrics: {e}")
            time.sleep(1)


def start_training_process(config='arc', custom_args=''):
    """Start the training process."""
    global training_process, is_training, current_logdir, training_start_time, training_thread
    
    if is_training:
        return {'status': 'error', 'message': 'Training already running'}
    
    # Create timestamp for logdir
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_logdir = str(Path.home() / 'logdir' / timestamp)
    
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
        
        # Start monitoring thread
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
    scores = list(metrics_data['scores'])
    lengths = list(metrics_data['lengths'])
    train_loss = list(metrics_data.get('train_loss', []))
    fps = list(metrics_data['fps'])
    
    # Calculate moving averages
    scores_ma = calculate_moving_average(scores, window)
    lengths_ma = calculate_moving_average(lengths, window)
    train_loss_ma = calculate_moving_average(train_loss, window) if train_loss else []
    fps_ma = calculate_moving_average(fps, window)
    
    # Get latest values
    latest = {
        'step': steps[-1] if steps else 0,
        'score': scores[-1] if scores else 0,
        'score_ma': scores_ma[-1] if scores_ma else 0,
        'length': lengths[-1] if lengths else 0,
        'length_ma': lengths_ma[-1] if lengths_ma else 0,
        'train_loss': train_loss[-1] if train_loss else 0,
        'train_loss_ma': train_loss_ma[-1] if train_loss_ma else 0,
        'fps': fps[-1] if fps else 0,
        'fps_ma': fps_ma[-1] if fps_ma else 0,
    }
    
    return jsonify({
        'steps': steps,
        'scores': scores,
        'scores_ma': scores_ma,
        'lengths': lengths,
        'lengths_ma': lengths_ma,
        'train_loss': train_loss,
        'train_loss_ma': train_loss_ma,
        'fps': fps,
        'fps_ma': fps_ma,
        'latest': latest,
    })


@app.route('/api/clear', methods=['POST'])
def api_clear():
    """API endpoint to clear metrics data."""
    global metrics_data
    
    metrics_data = {
        'steps': deque(maxlen=1000),
        'scores': deque(maxlen=1000),
        'lengths': deque(maxlen=1000),
        'train_loss': deque(maxlen=1000),
        'fps': deque(maxlen=1000),
        'timestamps': deque(maxlen=1000),
    }
    
    return jsonify({'status': 'success', 'message': 'Metrics cleared'})


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

