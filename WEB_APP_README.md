# ARC Training Web Interface

A web-based dashboard for managing and monitoring DreamerV3 ARC training sessions.

## Quick Start

1. **Install Flask** (if not already installed):
   ```bash
   pip install flask
   ```

2. **Start the Web App**:
   ```bash
   cd dreamerv3
   python app.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5003`

## Features

### Training Control
- **Start Training**: Select a configuration and optionally add custom arguments
- **Stop Training**: Gracefully stop the current training session
- **Clear Data**: Reset all accumulated metrics
- **⭐ Configurable Log Frequency**: Control how often training writes metrics (default: 30s)
- **⭐ Configurable Update Speed**: Adjust web app refresh rate with presets or custom intervals

### Available Configurations
- `arc` - Default ARC V2 Training
- `arc_v2_train` - ARC V2 Training Set
- `arc_v2_eval` - ARC V2 Evaluation Set
- `arc_v1_train` - ARC V1 Training Set
- `arc_v1_eval` - ARC V1 Evaluation Set

### Real-Time Monitoring

The dashboard automatically updates (default: every 0.5 seconds, configurable) with:

1. **Score (Accuracy) Chart**
   - Raw score values per episode
   - Moving average for trend analysis
   - Represents how accurately the agent solves puzzles (0.0 - 1.0)

2. **Episode Length Chart**
   - Number of steps per episode
   - Moving average for trend analysis

3. **FPS Chart**
   - Policy FPS (frames per second)
   - Moving average for performance monitoring

### Metrics Display

The interface shows both **raw values** and **moving averages** for all metrics:
- Current step number
- Latest score and moving average
- Episode length and moving average
- FPS and moving average

### Moving Average Window

You can adjust the moving average window size (default: 10) to control smoothing:
- Smaller values (e.g., 5): More responsive to recent changes
- Larger values (e.g., 50): Smoother trends, less noise

### ⚙️ Update Settings (NEW!)

Control how quickly metrics update with flexible configuration options:

**Training Log Frequency** (`Log Every` field):
- Controls how often the training process writes metrics to disk
- Default: 30 seconds (faster than previous 120s default)
- Lower values = more responsive metrics but slightly higher overhead
- Range: 1-600 seconds

**Web Update Interval** (Update Settings panel):
- Controls how often the web app checks for new metrics
- Presets:
  - **Fast (0.2s)**: Very responsive, best for demos and active monitoring
  - **Default (0.5s)**: Balanced, 4x faster than old default
  - **Moderate (1.0s)**: Good for slower systems
  - **Slow (2.0s)**: Matches old behavior, lower overhead
  - **Custom**: Set any value between 0.1s and 60s

💡 **Tip**: For the most responsive experience during active monitoring, use:
- Log Every: 10-30 seconds
- Web Update Interval: 0.2s (Fast) or 0.5s (Default)

📖 **For detailed information**, see [WEB_APP_UPDATE_GUIDE.md](../WEB_APP_UPDATE_GUIDE.md)

## How It Works

### Backend (`app.py`)
- Flask web server running on port 5003
- Spawns training subprocess when started
- Monitors `metrics.jsonl` file in the logdir
- Calculates moving averages in real-time
- Provides REST API endpoints for status and metrics
- **NEW**: Configurable metrics update interval (default: 0.5s)

### Frontend
- Modern, responsive UI built with HTML/CSS/JavaScript
- Chart.js for interactive visualizations
- Auto-refreshing at configurable intervals (default: 0.5 seconds)
- Smooth animations and transitions
- **NEW**: Update settings panel with preset and custom options

## API Endpoints

If you want to integrate with the backend programmatically:

- `GET /api/status` - Get training status (now includes `metrics_update_interval`)
- `GET /api/metrics?window=10` - Get metrics with specified moving average window
- `GET /api/logs?lines=100` - Get console logs (optional: last N lines)
- `POST /api/start` - Start training (JSON body: `{config, custom_args}`)
- `POST /api/stop` - Stop training
- `POST /api/clear` - Clear metrics data
- **NEW** `GET /api/config/update_interval` - Get current metrics update interval
- **NEW** `POST /api/config/update_interval` - Set metrics update interval (JSON body: `{interval}`)

## Training Logs

Training logs are saved to `~/logdir/<timestamp>/`:
- `metrics.jsonl` - All training metrics
- `scores.jsonl` - Episode scores
- `ckpt/` - Model checkpoints
- `config.yaml` - Training configuration

## Tips

1. **For Quick Tests**: Use custom args like `--run.steps 1e5` to limit training steps
2. **For Fast Feedback**: Set "Log Every" to 10-20 seconds and use "Fast" update interval
3. **Memory Management**: Clear data between runs if needed
4. **Multiple Sessions**: The app tracks the most recent training session
5. **Patience**: Training may take a few seconds to start and begin logging metrics
6. **Performance Tuning**: If browser feels sluggish, increase the web update interval to 1-2 seconds

## Troubleshooting

**No metrics showing up?**
- With default settings (Log Every: 30s), expect first metrics within 30-60 seconds
- Check that the logdir path is displayed in the UI
- Ensure the ARC data is correctly placed in `arc-data/` directory
- Try lowering "Log Every" to 10 seconds for faster initial feedback

**Training won't start?**
- Make sure no other training process is running
- Check that all dependencies are installed
- Verify the `dreamerv3/main.py` file exists

**Web app won't start?**
- Ensure Flask is installed: `pip install flask`
- Check that port 5003 is not already in use
- Try running with: `python -u app.py` for unbuffered output

**Metrics updating too slowly?**
- Decrease "Log Every" (training side) for more frequent metric writes
- Decrease "Web Update Interval" (web side) for faster UI refresh
- See [WEB_APP_UPDATE_GUIDE.md](../WEB_APP_UPDATE_GUIDE.md) for optimal settings

**Browser feeling sluggish or high CPU usage?**
- Increase the "Web Update Interval" to 1.0s or 2.0s
- This reduces the frequency of chart updates and API calls

## Architecture

```
dreamerv3/
├── app.py                    # Flask backend
├── templates/
│   └── index.html           # Main web interface
└── static/
    ├── style.css            # Styling
    └── app.js               # Frontend JavaScript
```

The app is completely self-contained in the `dreamerv3` folder and requires no additional configuration.

## Credits

Built for DreamerV3 ARC training monitoring.
Uses Chart.js for visualizations and Flask for the backend.

