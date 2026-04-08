from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify
import os
import subprocess
import sys
import json
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'runs/detect'
app.config['STATS_FILE'] = 'stats.json'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def load_stats():
    if os.path.exists(app.config['STATS_FILE']):
        with open(app.config['STATS_FILE'], 'r') as f:
            stats = json.load(f)
            # Ensure new keys exist
            if 'recent_missions' not in stats: stats['recent_missions'] = []
            if 'class_counts' not in stats: stats['class_counts'] = {}
            return stats
    return {'total_inferences': 0, 'total_time': 0, 'files_processed': [], 'recent_missions': [], 'class_counts': {}}

def save_stats(stats):
    with open(app.config['STATS_FILE'], 'w') as f:
        json.dump(stats, f)

@app.route('/')
def index():
    stats = load_stats()
    return render_template('index.html', recent=stats['recent_missions'])

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Run inference
        output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{file.filename}")
        start_time = time.time()
        try:
            result = subprocess.run([
                sys.executable, 'run_inference_video.py',
                '--model', 'runs/detect/train/weights/best.pt',
                '--source', filepath,
                '--output', output_path,
                '--conf', '0.4'
            ], capture_output=True, text=True, cwd=os.getcwd())
            end_time = time.time()
            
            if result.returncode == 0:
                # Parse metrics from stdout
                stdout = result.stdout
                metrics = {}
                if "DETECTION_METRICS_START" in stdout:
                    try:
                        metrics_str = stdout.split("DETECTION_METRICS_START")[1].split("DETECTION_METRICS_END")[0]
                        metrics = json.loads(metrics_str)
                    except:
                        pass

                # Update stats
                stats = load_stats()
                stats['total_inferences'] += 1
                stats['total_time'] += (end_time - start_time)
                stats['files_processed'].append({'filename': file.filename, 'time': end_time - start_time})
                
                # Update class counts
                summary = metrics.get('summary', {})
                for cls, count in summary.items():
                    stats['class_counts'][cls] = stats['class_counts'].get(cls, 0) + count
                
                # Update recent missions (keep last 5)
                stats['recent_missions'].insert(0, {
                    'filename': file.filename,
                    'result_file': f"result_{file.filename}",
                    'summary': summary,
                    'time': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                stats['recent_missions'] = stats['recent_missions'][:5]
                
                save_stats(stats)
                return render_template('result.html', filename=f"result_{file.filename}", summary=summary)
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error running inference: {str(e)}"

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/stats')
def stats():
    stats = load_stats()
    avg_time = stats['total_time'] / stats['total_inferences'] if stats['total_inferences'] > 0 else 0
    return render_template('stats.html', stats=stats, avg_time=avg_time)

if __name__ == '__main__':
    app.run(debug=True)