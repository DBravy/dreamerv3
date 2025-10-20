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

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Global state
training_process = None
training_thread = None
log_monitor_thread = None
is_training = False
metrics_data = {
    'steps': deque(maxlen=1000),
    'accuracies': deque(maxlen=1000),  # Track end-of-episode accuracy (0-100%)
    'lengths': deque(maxlen=1000),
    'train_loss': deque(maxlen=1000),
    'timestamps': deque(maxlen=1000),
}
console_logs = deque(maxlen=500)  # Store last 500 lines of console output
grid_visualizations = deque(maxlen=50)  # Store last 50 grid visualizations
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


def save_grid_visualization(grid_data, step, accuracy):
    """Save a grid visualization as an image."""
    try:
        import numpy as np
        from PIL import Image
        
        # ARC color palette (same as in arc.py)
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
        
        test_input = grid_data['test_input']
        agent_output = grid_data['agent_output']
        
        # Scale factor for visualization
        cell_size = 20
        
        # Create images for input and output
        def grid_to_image(grid):
            h, w = grid.shape
            img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
            
            for i in range(h):
                for j in range(w):
                    color = COLOR_MAP.get(int(grid[i, j]), [0, 0, 0])
                    img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = color
            
            return img
        
        input_img = grid_to_image(test_input)
        output_img = grid_to_image(agent_output)
        
        # Add grid lines
        def add_grid_lines(img, grid_shape):
            h, w = grid_shape
            img_h, img_w = img.shape[:2]
            
            # Horizontal lines
            for i in range(1, h):
                img[i*cell_size-1:i*cell_size+1, :] = [80, 80, 80]
            
            # Vertical lines
            for j in range(1, w):
                img[:, j*cell_size-1:j*cell_size+1] = [80, 80, 80]
            
            return img
        
        input_img = add_grid_lines(input_img, test_input.shape)
        output_img = add_grid_lines(output_img, agent_output.shape)
        
        # Combine side by side with padding
        padding = 40
        max_height = max(input_img.shape[0], output_img.shape[0])
        combined_width = input_img.shape[1] + output_img.shape[1] + padding
        
        combined = np.ones((max_height + 60, combined_width, 3), dtype=np.uint8) * 240
        
        # Paste input
        y_offset_input = (max_height - input_img.shape[0]) // 2 + 30
        combined[y_offset_input:y_offset_input+input_img.shape[0], :input_img.shape[1]] = input_img
        
        # Paste output
        y_offset_output = (max_height - output_img.shape[0]) // 2 + 30
        x_offset_output = input_img.shape[1] + padding
        combined[y_offset_output:y_offset_output+output_img.shape[0], x_offset_output:x_offset_output+output_img.shape[1]] = output_img
        
        # Convert to PIL Image
        img = Image.fromarray(combined)
        
        # Save to static directory
        static_dir = Path('static/grids')
        static_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'grid_step_{step}_{timestamp}.png'
        filepath = static_dir / filename
        
        img.save(filepath)
        
        return {
            'step': step,
            'accuracy': accuracy,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'input_shape': test_input.shape,
            'output_shape': agent_output.shape
        }
        
    except Exception as e:
        print(f"Error saving grid visualization: {e}")
        return None


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
    global is_training, current_logdir, metrics_data, grid_visualizations
    
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
    grids_dir = Path(current_logdir) / 'grids'
    
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
    processed_grids = set()
    
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
                                
                                # Extract end-of-episode accuracy from final reward (0-1) -> percentage (0-100)
                                if 'episode/final_reward' in metric:
                                    metrics_data['steps'].append(step)
                                    accuracy_percent = float(metric['episode/final_reward']) * 100.0
                                    # Clamp to [0, 100] for display stability
                                    accuracy_percent = max(0.0, min(100.0, accuracy_percent))
                                    metrics_data['accuracies'].append(accuracy_percent)
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
                                    
                            except json.JSONDecodeError:
                                continue
            
            # Check for new grid visualizations
            if grids_dir.exists():
                for grid_file in grids_dir.glob('grid_*.json'):
                    if grid_file.name not in processed_grids:
                        try:
                            with open(grid_file, 'r') as f:
                                grid_data = json.load(f)
                            
                            # Convert lists back to numpy arrays
                            import numpy as np
                            grid_data['test_input'] = np.array(grid_data['test_input'], dtype=np.uint8)
                            grid_data['agent_output'] = np.array(grid_data['agent_output'], dtype=np.uint8)
                            
                            # Save visualization
                            viz_info = save_grid_visualization(
                                grid_data,
                                grid_data.get('step', 0),
                                grid_data.get('accuracy', 0)
                            )
                            
                            if viz_info:
                                grid_visualizations.append(viz_info)
                                print(f"Saved grid visualization for step {viz_info['step']}")
                            
                            processed_grids.add(grid_file.name)
                            
                        except Exception as e:
                            print(f"Error processing grid file {grid_file}: {e}")
            
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
    global training_process, is_training, current_logdir, training_start_time, training_thread, log_monitor_thread, console_logs
    
    if is_training:
        return {'status': 'error', 'message': 'Training already running'}
    
    # Clear previous console logs
    console_logs.clear()
    
    # Create timestamp for logdir (matches format used in main.py)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir_base = Path(__file__).parent.parent / 'logdir'
    current_logdir = str(logdir_base / f'arc-{timestamp}')
    
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
    lengths = list(metrics_data['lengths'])
    train_loss = list(metrics_data.get('train_loss', []))
    
    # Calculate moving averages
    accuracies_ma = calculate_moving_average(accuracies, window)
    lengths_ma = calculate_moving_average(lengths, window)
    train_loss_ma = calculate_moving_average(train_loss, window) if train_loss else []
    
    # Get latest values
    latest = {
        'step': steps[-1] if steps else 0,
        'accuracy': accuracies[-1] if accuracies else 0,
        'accuracy_ma': accuracies_ma[-1] if accuracies_ma else 0,
        'length': lengths[-1] if lengths else 0,
        'length_ma': lengths_ma[-1] if lengths_ma else 0,
        'train_loss': train_loss[-1] if train_loss else 0,
        'train_loss_ma': train_loss_ma[-1] if train_loss_ma else 0,
    }
    
    return jsonify({
        'steps': steps,
        'accuracies': accuracies,
        'accuracies_ma': accuracies_ma,
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


@app.route('/api/clear', methods=['POST'])
def api_clear():
    """API endpoint to clear metrics data."""
    global metrics_data, console_logs, grid_visualizations
    
    metrics_data = {
        'steps': deque(maxlen=1000),
        'accuracies': deque(maxlen=1000),
        'lengths': deque(maxlen=1000),
        'train_loss': deque(maxlen=1000),
        'timestamps': deque(maxlen=1000),
    }
    console_logs.clear()
    grid_visualizations.clear()
    
    return jsonify({'status': 'success', 'message': 'Metrics and logs cleared'})


@app.route('/api/grids', methods=['GET'])
def api_grids():
    """API endpoint to get grid visualizations."""
    # Get optional parameters
    limit = request.args.get('limit', 10, type=int)
    
    grids = list(grid_visualizations)
    
    # Return only last N grids if specified
    if limit > 0:
        grids = grids[-limit:]
    
    return jsonify({
        'grids': grids,
        'total_grids': len(grid_visualizations)
    })


@app.route('/static/grids/<filename>')
def serve_grid_image(filename):
    """Serve grid visualization images."""
    try:
        filepath = Path('static/grids') / filename
        if filepath.exists():
            return send_file(filepath, mimetype='image/png')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

