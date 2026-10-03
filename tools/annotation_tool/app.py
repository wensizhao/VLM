#!/usr/bin/env python3
"""
人工标注工具Web应用
用于VLM-VAD项目的视频异常prompt标注
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import json
import sqlite3
from datetime import datetime
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'vlm-vad-annotation-secret-key-2024'

# 配置
UPLOAD_FOLDER = 'data/videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
DATABASE = 'annotations.db'

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """初始化数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 创建视频表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            duration INTEGER,
            fps REAL,
            width INTEGER,
            height INTEGER,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    # 创建标注表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            annotator_id TEXT NOT NULL,
            annotation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            anomaly_segments TEXT NOT NULL,
            overall_quality_score REAL,
            review_status TEXT DEFAULT 'pending',
            notes TEXT,
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
    ''')
    
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'annotator',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def allowed_file(filename):
    """检查文件扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 简单的登录验证（实际项目中应该使用更安全的方式）
        if username == 'admin' and password == 'admin':
            return redirect(url_for('dashboard'))
        else:
            flash('用户名或密码错误')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """仪表板"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 统计信息
    cursor.execute('SELECT COUNT(*) FROM videos')
    total_videos = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM annotations')
    total_annotations = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM videos WHERE status = "pending"')
    pending_videos = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM annotations WHERE review_status = "pending"')
    pending_reviews = cursor.fetchone()[0]
    
    conn.close()
    
    return render_template('dashboard.html', 
                         total_videos=total_videos,
                         total_annotations=total_annotations,
                         pending_videos=pending_videos,
                         pending_reviews=pending_reviews)

@app.route('/videos')
def videos_list():
    """视频列表"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    offset = (page - 1) * per_page
    
    cursor.execute('''
        SELECT * FROM videos 
        ORDER BY upload_time DESC 
        LIMIT ? OFFSET ?
    ''', (per_page, offset))
    
    videos = cursor.fetchall()
    
    cursor.execute('SELECT COUNT(*) FROM videos')
    total = cursor.fetchone()[0]
    
    total_pages = (total + per_page - 1) // per_page
    
    conn.close()
    
    return render_template('videos.html', 
                         videos=videos, 
                         page=page, 
                         total_pages=total_pages)

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    """上传视频"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('没有选择文件')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('没有选择文件')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            file.save(filepath)
            
            # TODO: 获取视频信息（时长、分辨率等）
            # 这里简化处理，实际应该使用OpenCV或FFmpeg获取
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO videos (video_id, filename, filepath, status)
                VALUES (?, ?, ?, ?)
            ''', (video_id, filename, filepath, 'pending'))
            
            conn.commit()
            conn.close()
            
            flash('视频上传成功')
            return redirect(url_for('videos_list'))
        else:
            flash('不支持的文件格式')
    
    return render_template('upload.html')

@app.route('/annotate/<video_id>')
def annotate_video(video_id):
    """标注视频"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 获取视频信息
    cursor.execute('SELECT * FROM videos WHERE video_id = ?', (video_id,))
    video = cursor.fetchone()
    
    if video is None:
        conn.close()
        flash('视频不存在')
        return redirect(url_for('videos_list'))
    
    # 检查是否已有标注
    cursor.execute('SELECT * FROM annotations WHERE video_id = ?', (video_id,))
    existing_annotation = cursor.fetchone()
    
    conn.close()
    
    return render_template('annotate.html', 
                         video=video, 
                         existing_annotation=existing_annotation)

@app.route('/api/videos/<video_id>/info')
def get_video_info(video_id):
    """获取视频信息API"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM videos WHERE video_id = ?', (video_id,))
    video = cursor.fetchone()
    
    conn.close()
    
    if video:
        return jsonify(dict(video))
    else:
        return jsonify({'error': '视频不存在'}), 404

@app.route('/api/annotations', methods=['POST'])
def create_annotation():
    """创建标注"""
    data = request.get_json()
    
    video_id = data.get('video_id')
    annotator_id = data.get('annotator_id', 'default_user')
    anomaly_segments = data.get('anomaly_segments', [])
    notes = data.get('notes', '')
    
    if not video_id or not anomaly_segments:
        return jsonify({'error': '缺少必要参数'}), 400
    
    # 计算质量分数（简化处理）
    quality_score = 0.8  # TODO: 实现实际的质量评估
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO annotations (video_id, annotator_id, anomaly_segments, overall_quality_score, notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (video_id, annotator_id, json.dumps(anomaly_segments), quality_score, notes))
    
    conn.commit()
    
    # 更新视频状态
    cursor.execute('UPDATE videos SET status = "annotated" WHERE video_id = ?', (video_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': '标注创建成功'})

@app.route('/api/annotations/<video_id>')
def get_annotations(video_id):
    """获取视频标注"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM annotations WHERE video_id = ?', (video_id,))
    annotations = cursor.fetchall()
    
    conn.close()
    
    result = []
    for ann in annotations:
        result.append(dict(ann))
    
    return jsonify(result)

@app.route('/api/stats/progress')
def get_progress_stats():
    """获取标注进度统计"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 总体进度
    cursor.execute('SELECT COUNT(*) FROM videos')
    total_videos = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM videos WHERE status = "annotated"')
    annotated_videos = cursor.fetchone()[0]
    
    # 按状态统计
    cursor.execute('''
        SELECT status, COUNT(*) 
        FROM videos 
        GROUP BY status
    ''')
    status_stats = cursor.fetchall()
    
    # 标注质量统计
    cursor.execute('SELECT AVG(overall_quality_score) FROM annotations')
    avg_quality = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_videos': total_videos,
        'annotated_videos': annotated_videos,
        'annotation_rate': annotated_videos / total_videos if total_videos > 0 else 0,
        'status_stats': [dict(row) for row in status_stats],
        'average_quality': avg_quality if avg_quality else 0
    })

@app.route('/api/prompts')
def get_prompts():
    """获取prompt配置"""
    prompts = {
        'label': {
            'template': 'a {behavior} behavior',
            'positive_labels': [
                'fighting', 'shooting', 'stabbing', 'vandalism', 'theft',
                'assault', 'burglary', 'arson', 'shoplifting', 'drug_use'
            ],
            'negative_labels': [
                'normal', 'walking', 'standing', 'sitting', 'talking',
                'running', 'exercising', 'shopping', 'working', 'studying'
            ]
        },
        'scene': {
            'template': 'a {scene} with abnormal activity',
            'positive_scenes': [
                'street', 'mall', 'bank', 'school', 'office',
                'restaurant', 'park', 'subway', 'airport', 'hospital'
            ],
            'negative_scenes': [
                'empty street', 'quiet mall', 'normal bank', 'calm school',
                'regular office', 'peaceful restaurant', 'empty park'
            ]
        },
        'contrast': {
            'template': 'normal vs abnormal {activity}',
            'positive_activities': [
                'behavior', 'movement', 'action', 'activity', 'conduct',
                'performance', 'gesture', 'motion', 'interaction', 'event'
            ],
            'negative_activities': [
                'normal behavior', 'regular movement', 'usual action',
                'standard activity', 'typical conduct'
            ]
        }
    }
    
    return jsonify(prompts)

if __name__ == '__main__':
    # 初始化数据库
    init_database()
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)