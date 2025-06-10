from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import requests
import json
import os
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Load environment variables
load_dotenv()


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'dev-key-123'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the LogEntry model
class LogEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=True)
    level = db.Column(db.String(20))
    message = db.Column(db.Text)
    source_file = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables (run this once)
with app.app_context():
    db.create_all()
# Configure Gemini
import google.generativeai as genai
import os
import json
os.environ['GEMINI_API_KEY'] = ''
print(f"API Key loaded: {os.getenv('GEMINI_API_KEY')}")
os.environ['GEMINI_API_URL']='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
def parse_logs(file_path):
    with open(file_path, 'r') as f:
        log_text = f.read()  # Read the entire file

    prompt = """You are a log file analyzer. Analyze the following log entries and return a JSON array where each entry has:
    - timestamp (in ISO format if available)
    - level (one of: ERROR, WARNING, INFO, DEBUG, or other appropriate level)
    - message (the actual log message)

    For each log entry, extract the timestamp (if available), log level, and the message. If a field cannot be determined, use null.

    Log entries:
    {log_text}"""

    payload = {
        "contents": [{"parts": [{"text": prompt.format(log_text=log_text[:10000])}]}],  # Limit to first 10k chars
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }

    try:
        response = requests.post(
            f"{os.getenv('GEMINI_API_URL')}?key={os.getenv('GEMINI_API_KEY')}",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()
        raw_output = data['candidates'][0]['content']['parts'][0]['text']

        # Extract JSON from the response
        json_start = raw_output.find('[')
        json_end = raw_output.rfind(']') + 1
        if json_start >= 0 and json_end > 0:
            print(json.loads(raw_output[json_start:json_end]))
            return json.loads(raw_output[json_start:json_end])
        else:
            print("Failed to parse JSON from response")
            return []

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return []
@app.context_processor
def inject_now():
    return {'now': datetime.now()}
@app.route('/logs')
def view_logs():
    page = request.args.get('page', 1, type=int)
    source_file = request.args.get('source_file')
    
    # Get all unique source files for the sidebar
    analyzed_files = db.session.query(
        LogEntry.source_file,
        db.func.count(LogEntry.id).label('entry_count'),
        db.func.max(LogEntry.created_at).label('last_analyzed')
    ).group_by(LogEntry.source_file).order_by(db.desc('last_analyzed')).all()
    
    # Get logs, filtered by source_file if specified
    query = LogEntry.query
    if source_file:
        query = query.filter_by(source_file=source_file)
    
    logs = query.order_by(LogEntry.created_at.desc()).paginate(page=page, per_page=50)
    
    # Get log level statistics for the current filter
    level_stats = db.session.query(
        LogEntry.level, 
        db.func.count(LogEntry.level)
    )
    if source_file:
        level_stats = level_stats.filter_by(source_file=source_file)
    level_stats = dict(level_stats.group_by(LogEntry.level).all())
    
    return render_template(
        'logs.html', 
        logs=logs,
        analyzed_files=analyzed_files,
        current_file=source_file,
        level_stats=level_stats
    )
@app.route('/api/logs/time-series')
def get_time_series_data():
    import pandas as pd
    from sqlalchemy import func, extract
    from datetime import datetime, timedelta
    print("=== TIME SERIES API CALLED ===")
    print(f"Request args: {dict(request.args)}")

    # Get time window from query params (default to 'minute')
    time_window = request.args.get('window', 'minute')
    
    # Get the actual time range of your data
    time_range_query = db.session.query(
        func.min(LogEntry.timestamp).label('earliest'),
        func.max(LogEntry.timestamp).label('latest')
    ).first()

    if time_range_query.latest is None:
        return jsonify({'error': 'No log entries found'}), 404

    end_time = time_range_query.latest
    earliest_time = time_range_query.earliest
    print(f"Earliest timestamp: {time_range_query.earliest}")
    print(f"Latest timestamp: {time_range_query.latest}")
    # Calculate start time, but don't go earlier than the earliest log entry
    if time_window == 'minute':
        calculated_start = end_time - timedelta(hours=24)
    elif time_window == 'hour':
        calculated_start = end_time - timedelta(days=7)
    else:  # day
        calculated_start = end_time - timedelta(days=30)

    start_time = max(calculated_start, earliest_time)
    
    # Base query with time filtering
    query = db.session.query(
        func.strftime('%Y-%m-%d %H:%M:00', LogEntry.timestamp).label('time_window'),
        LogEntry.level,
        func.count().label('count')
    ).filter(
        LogEntry.timestamp.isnot(None),
        LogEntry.timestamp.between(start_time, end_time)
    )
    
    # Apply source file filter if specified
    if 'source_file' in request.args:
        query = query.filter(LogEntry.source_file == request.args['source_file'])
    
    # Group by time window and level
    query = query.group_by('time_window', LogEntry.level).order_by('time_window')
    
    # Execute query and convert to DataFrame
    sql_statement = str(query.statement.compile(compile_kwargs={"literal_binds": True}))
    df = pd.read_sql(sql_statement, db.engine)
    
    if df.empty:
        return jsonify({'error': 'No data available'}), 404
    
    # Create a complete time index with all expected time points
    if not df.empty:
        time_range = pd.date_range(
            start=start_time,
            end=end_time,
            freq=time_window[0].upper()  # 'M' for minute, 'H' for hour, 'D' for day
        )
        
        # Pivot the data for Chart.js
        pivot_df = df.pivot(index='time_window', columns='level', values='count').reindex(time_range).fillna(0)
    else:
        pivot_df = pd.DataFrame()
    
    # Convert to Chart.js format
    datasets = []
    colors = {
        'ERROR': 'rgb(220, 53, 69)',
        'WARNING': 'rgb(255, 193, 7)',
        'INFO': 'rgb(13, 110, 253)',
        'DEBUG': 'rgb(108, 117, 125)'
    }
    
    for level in pivot_df.columns:
        datasets.append({
            'label': level,
            'data': [{'x': idx.isoformat(), 'y': int(val)} for idx, val in pivot_df[level].items()],
            'borderColor': colors.get(level, 'rgb(0, 0, 0)'),
            'backgroundColor': colors.get(level, 'rgb(0, 0, 0)'),
            'borderWidth': 2,
            'fill': False
        })
    
    print(f"Time range: {start_time} to {end_time}")
    print(f"Datasets: {len(datasets)} levels found")
    
    return jsonify({
        'datasets': datasets,
        'timeRange': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'earliest': earliest_time.isoformat()
        }
    })
# ────────────────────────────────────────────────────────────────
#  Analytics view   (bar-chart + per-minute line chart, Plotly)
# ────────────────────────────────────────────────────────────────
@app.route('/analytics')
def view_analytics():
    from sqlalchemy import func
    import pandas as pd
    import plotly.express as px

    # ── Which file? ──────────────────────────────────────────────
    source_file = request.args.get('source_file')

    # Sidebar listing
    analyzed_files = db.session.query(
        LogEntry.source_file
    ).group_by(LogEntry.source_file
    ).order_by(func.max(LogEntry.created_at).desc()).all()

    if not source_file and analyzed_files:
        source_file = analyzed_files[0].source_file

    # ── BAR CHART: counts per level ─────────────────────────────
    level_rows = db.session.query(
        LogEntry.level, func.count().label('cnt')
    ).filter(
        LogEntry.source_file == source_file
    ).group_by(LogEntry.level).all()

    level_order = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
    level_dict  = {lvl: 0 for lvl in level_order}
    level_dict.update({lvl: cnt for lvl, cnt in level_rows})

    bar_fig = px.bar(
        x=list(level_dict.keys()),
        y=list(level_dict.values()),
        labels={'x': 'Log Level', 'y': 'Occurrences'},
        color=list(level_dict.keys()),
        color_discrete_map={
            'ERROR':'rgb(220,53,69)', 'WARNING':'rgb(255,193,7)',
            'INFO':'rgb(13,110,253)', 'DEBUG':'rgb(108,117,125)'
        },
        title=f'Log-Level Distribution for {source_file}'
    )
    bar_fig.update_layout(showlegend=False, plot_bgcolor='white')

    # ── LINE CHART: entries per minute ──────────────────────────
    per_min_rows = db.session.query(
        func.strftime('%Y-%m-%d %H:%M:00', LogEntry.timestamp).label('minute'),
        func.count().label('cnt')
    ).filter(
        LogEntry.source_file == source_file
    ).group_by('minute').order_by('minute').all()

    if per_min_rows:
        df_ts = pd.DataFrame(per_min_rows, columns=['minute', 'cnt'])
        df_ts['minute'] = pd.to_datetime(df_ts['minute'])

        # Fill missing minutes with 0
        full_range = pd.date_range(df_ts['minute'].min(), df_ts['minute'].max(), freq='T')
        df_ts = df_ts.set_index('minute').reindex(full_range).fillna(0).rename_axis('minute').reset_index()

        line_fig = px.line(
            df_ts, x='minute', y='cnt',
            labels={'minute': 'Time', 'cnt': 'Entries / minute'},
            title='Entries per Minute'
        )
        line_fig.update_traces(mode='lines+markers')
        line_fig.update_layout(plot_bgcolor='white')
    else:
        # Empty figure if no rows
        import plotly.graph_objects as go
        line_fig = go.Figure()
        line_fig.update_layout(title='Entries per Minute', xaxis_title='Time', yaxis_title='Entries')

    # ── Embed HTML ───────────────────────────────────────────────
    plot_bar  = bar_fig .to_html(full_html=False, include_plotlyjs='cdn')
    plot_line = line_fig.to_html(full_html=False, include_plotlyjs=False)

    # ── Render template ─────────────────────────────────────────
    return render_template(
        'analytics.html',
        analyzed_files=analyzed_files,
        current_file=source_file,
        plot_bar=plot_bar,
        plot_line=plot_line
    )



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Parse logs
            logs = parse_logs(filepath)
            
            if logs:
                # Store logs in database
                for log in logs:
                    entry = LogEntry(
                        timestamp=datetime.fromisoformat(log['timestamp']) if log.get('timestamp') else None,
                        level=log.get('level', 'INFO'),
                        message=log.get('message', ''),
                        source_file=filename
                    )
                    db.session.add(entry)
                db.session.commit()
                
                # Get log statistics
                level_counts = db.session.query(
                    LogEntry.level, 
                    db.func.count(LogEntry.level)
                ).filter_by(source_file=filename).group_by(LogEntry.level).all()
                
                level_counts = dict(level_counts)
                
                # Get recent logs
                recent_logs = LogEntry.query.filter_by(source_file=filename).order_by(
                    db.desc(LogEntry.created_at)
                ).limit(50).all()
                
                return render_template('index.html', 
                                     logs=recent_logs,
                                     level_counts=level_counts,
                                     filename=filename)
    
    # For GET requests or if no file was processed
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)