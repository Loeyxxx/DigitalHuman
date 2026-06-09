#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字人播报系统 - Flask后端服务
"""

import os
import json
import uuid
import shutil
import threading
import base64
import time
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import requests

from digital_human.tts_engine import TTSEngine
from digital_human.video_generator import VideoGenerator
from digital_human.utils import get_upload_path, allowed_file, cleanup_old_files
from digital_human.voice_resolver import resolve_voice

# 配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'mp3', 'wav', 'm4a'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get('SECRET_KEY', 'digital-human-secret-key-2024')
CORS(app)

# 任务存储（生产环境应使用Redis等）
tasks = {}
tasks_lock = threading.Lock()

# 初始化引擎
tts_engine = TTSEngine()
video_generator = VideoGenerator()

def _normalize_openai_base_url(url: str) -> str:
    url = (url or '').strip().rstrip('/')
    # Allow users to set OPENAI_BASE_URL as either:
    # - https://api.openai.com
    # - https://api.openai.com/v1
    if url.endswith('/v1'):
        url = url[:-3]
    return url


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '').strip()
OPENAI_BASE_URL = _normalize_openai_base_url(os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com'))
OPENAI_IMAGE_MODEL = os.environ.get('OPENAI_IMAGE_MODEL', 'gpt-image-2').strip()
OPENAI_TEXT_MODEL = (os.environ.get('OPENAI_TEXT_MODEL') or os.environ.get('OPENAI_MODEL') or 'gpt-4.1-mini').strip()
OPENAI_IMAGE_TIMEOUT_SECS = int(os.environ.get('OPENAI_IMAGE_TIMEOUT_SECS', '180') or 180)
OPENAI_IMAGE_MAX_RETRIES = int(os.environ.get('OPENAI_IMAGE_MAX_RETRIES', '1') or 1)


@app.route('/api/avatar/generate', methods=['POST'])
def generate_avatar():
    """根据文本描述生成数字人形象图片（OpenAI image model）"""
    try:
        app.logger.info("POST /api/avatar/generate")
        if not OPENAI_API_KEY:
            return jsonify({'code': -1, 'message': '未配置 OPENAI_API_KEY，无法生成图片'}), 400

        data = request.get_json() or {}
        prompt = (data.get('prompt') or '').strip()
        if not prompt:
            return jsonify({'code': -1, 'message': 'prompt 不能为空'}), 400

        size = (data.get('size') or '1024x1024').strip()
        if size not in {'1024x1024', '1024x1536', '1536x1024'}:
            return jsonify({'code': -1, 'message': 'size 仅支持 1024x1024 / 1024x1536 / 1536x1024'}), 400

        payload = {
            'model': OPENAI_IMAGE_MODEL,
            'prompt': prompt,
            'size': size,
        }

        app.logger.info("OpenAI images: base_url=%s model=%s size=%s", OPENAI_BASE_URL, OPENAI_IMAGE_MODEL, size)
        last_err = None
        for attempt in range(OPENAI_IMAGE_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f'{OPENAI_BASE_URL}/v1/images/generations',
                    headers={
                        'Authorization': f'Bearer {OPENAI_API_KEY}',
                        'Content-Type': 'application/json',
                    },
                    json=payload,
                    timeout=(10, OPENAI_IMAGE_TIMEOUT_SECS),
                )
            except requests.Timeout as e:
                last_err = e
                if attempt >= OPENAI_IMAGE_MAX_RETRIES:
                    raise
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp.status_code != 200:
                # Azure/OpenAI rate limit messages often include: "Please retry after N seconds"
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text

                retry_after = None
                try:
                    msg = err.get('error', {}).get('message', '') if isinstance(err, dict) else str(err)
                    m = re.search(r"retry after\\s+(\\d+)\\s+seconds", msg, flags=re.IGNORECASE)
                    if m:
                        retry_after = int(m.group(1))
                except Exception:
                    retry_after = None

                if retry_after is not None:
                    # If we still have retry budget, wait and retry. Otherwise surface to client.
                    if attempt < OPENAI_IMAGE_MAX_RETRIES:
                        app.logger.warning("OpenAI images rate-limited, retry after %ss (attempt %s)", retry_after, attempt + 1)
                        time.sleep(max(1, retry_after))
                        continue
                    return jsonify({
                        'code': -1,
                        'message': f'图片生成被限流，请 {retry_after} 秒后重试',
                        'data': {'retry_after_seconds': retry_after},
                    }), 429

                return jsonify({'code': -1, 'message': f'图片生成失败: {err}'}), 500

            # Success
            result = resp.json()
            b64 = None
            if isinstance(result, dict):
                data_list = result.get('data') or []
                if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
                    b64 = data_list[0].get('b64_json')

            if not b64:
                return jsonify({'code': -1, 'message': '图片生成失败：未返回 b64_json'}), 500

            img_bytes = base64.b64decode(b64)
            filename = f"{uuid.uuid4().hex}_avatar.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(img_bytes)

            return jsonify({
                'code': 0,
                'data': {
                    'filename': filename,
                    'original_name': 'generated_avatar.png',
                    'url': f'/static/uploads/{filename}',
                    'type': 'avatar'
                },
                'message': '生成成功'
            })

        if last_err is not None:
            raise last_err

        return jsonify({'code': -1, 'message': '图片生成失败：未知错误'}), 500

    except requests.Timeout:
        return jsonify({
            'code': -1,
            'message': f'图片生成超时（>{OPENAI_IMAGE_TIMEOUT_SECS}s），可提高 OPENAI_IMAGE_TIMEOUT_SECS 后重试'
        }), 504
    except requests.RequestException as e:
        return jsonify({'code': -1, 'message': f'图片生成网络错误: {str(e)}'}), 502
    except Exception as e:
        return jsonify({'code': -1, 'message': f'图片生成异常: {str(e)}'}), 500



def _course_draft_to_narration(draft: dict) -> str:
    title = (draft.get('title') or '').strip()
    disclaimer = (draft.get('disclaimer') or '').strip()
    sections = draft.get('sections') or []
    out = []
    if title:
        out.append(title)
    if disclaimer:
        out.append(disclaimer)
    for s in sections:
        if not isinstance(s, dict):
            continue
        heading = (s.get('heading') or '').strip()
        content = (s.get('content') or '').strip()
        if heading:
            out.append(heading)
        if content:
            out.append(content)
    return "\n\n".join([x for x in out if x])


@app.route('/api/course/draft', methods=['POST'])
def generate_course_draft():
    """生成证券投教课程脚本（结构化 JSON），供审核后再生成视频。"""
    try:
        if not OPENAI_API_KEY:
            return jsonify({'code': -1, 'message': '未配置 OPENAI_API_KEY，无法生成课程脚本'}), 400

        data = request.get_json() or {}
        topic = (data.get('topic') or '').strip()
        if not topic:
            return jsonify({'code': -1, 'message': 'topic 不能为空'}), 400

        audience = (data.get('audience') or 'beginner').strip().lower()
        if audience not in {'beginner', 'intermediate', 'advanced'}:
            return jsonify({'code': -1, 'message': 'audience 仅支持 beginner/intermediate/advanced'}), 400

        duration_min = int(data.get('duration_min') or 5)
        duration_min = max(1, min(duration_min, 30))

        style = (data.get('style') or 'professional_friendly').strip()
        if len(style) > 80:
            style = style[:80]

        key_points = (data.get('key_points') or '').strip()
        if len(key_points) > 600:
            key_points = key_points[:600]

        # Lazy import so non-LLM deployments don't require openai installed.
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=f"{OPENAI_BASE_URL}/v1")

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "disclaimer": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"},
                            "takeaways": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["heading", "content", "takeaways"],
                    },
                    "minItems": 3,
                    "maxItems": 10,
                },
                "quiz": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "question": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
                            "answer_index": {"type": "integer"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["question", "options", "answer_index", "explanation"],
                    },
                    "minItems": 0,
                    "maxItems": 5,
                },
            },
            "required": ["title", "disclaimer", "sections", "quiz"],
        }

        user_payload = {
            "topic": topic,
            "audience": audience,
            "duration_min": duration_min,
            "style": style,
            "key_points": key_points,
            "hard_rules": [
                "仅做证券投资者教育，不构成投资建议",
                "不得出现具体个股/代码的买卖指令或收益承诺",
                "出现风险提示与不确定性，避免绝对化措辞",
                "用通俗中文解释概念，给出日常类比或简单例子",
            ],
        }

        resp = client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是证券投资者教育课程编写助手。"
                        "请根据输入参数生成一份适合口播的课程脚本草稿，结构化输出。"
                        "必须遵守 hard_rules；输出应清晰、分段、易于朗读。"
                        "只输出符合 JSON schema 的对象，不要输出多余文字。"
                    ),
                },
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            # Responses API uses `text.format` for structured outputs / JSON mode.
            text={"format": {"type": "json_schema", "name": "course_draft", "schema": schema, "strict": True}},
        )

        raw = getattr(resp, "output_text", None) or ""
        if not raw:
            try:
                raw = resp.output[0].content[0].text  # type: ignore[attr-defined]
            except Exception:
                raw = ""

        # Fallback: if structured outputs isn't supported by this model / SDK, retry with plain JSON mode.
        if not raw:
            resp = client.responses.create(
                model=OPENAI_TEXT_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "你是证券投资者教育课程编写助手。"
                            "请根据输入参数生成一份适合口播的课程脚本草稿，结构化输出。"
                            "必须遵守 hard_rules；输出应清晰、分段、易于朗读。"
                            "只输出 JSON 对象，不要输出多余文字。"
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                text={"format": {"type": "json_object"}},
            )
            raw = getattr(resp, "output_text", None) or ""
            if not raw:
                try:
                    raw = resp.output[0].content[0].text  # type: ignore[attr-defined]
                except Exception:
                    raw = ""

        draft = json.loads(raw)
        narration_text = _course_draft_to_narration(draft)

        return jsonify({
            'code': 0,
            'data': {
                'draft': draft,
                'narration_text': narration_text,
                'model': OPENAI_TEXT_MODEL,
            },
            'message': 'success'
        })

    except Exception as e:
        return jsonify({'code': -1, 'message': f'课程脚本生成失败: {str(e)}'}), 500


@app.route('/')
def index():
    """渲染前端页面"""
    return render_template('index.html')


@app.route('/api/voices', methods=['GET'])
def get_voices():
    """获取可用的音色列表"""
    try:
        voices = tts_engine.get_available_voices()
        return jsonify({
            'code': 0,
            'data': voices,
            'message': 'success'
        })
    except Exception as e:
        return jsonify({
            'code': -1,
            'data': None,
            'message': str(e)
        }), 500

@app.route('/api/voice/resolve', methods=['POST'])
def resolve_voice_api():
    """根据自然语言描述自动匹配音色（从内置音色库中选择）"""
    try:
        data = request.get_json() or {}
        prompt = (data.get('prompt') or data.get('voicePrompt') or '').strip()
        text = (data.get('text') or '').strip()

        if not prompt:
            return jsonify({'code': -1, 'message': '缺少音色描述 prompt'}), 400

        voices = tts_engine.get_available_voices()
        resolution = resolve_voice(prompt=prompt, text=text, voices=voices)

        return jsonify({
            'code': 0,
            'data': {
                'voice_id': resolution.voice_id,
                'speed': resolution.speed,
                'pitch_hz': resolution.pitch_hz,
                'volume_pct': resolution.volume_pct,
                'confidence': resolution.confidence,
                'rationale': resolution.rationale,
            },
            'message': 'success'
        })
    except Exception as e:
        return jsonify({'code': -1, 'message': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件接口"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'code': -1,
                'message': '没有文件被上传'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'code': -1,
                'message': '文件名为空'
            }), 400
        
        file_type = request.form.get('type', 'general')
        
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            # 生成唯一文件名
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}_{file_type}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)
            
            return jsonify({
                'code': 0,
                'data': {
                    'filename': unique_name,
                    'original_name': file.filename,
                    'url': f'/static/uploads/{unique_name}',
                    'type': file_type
                },
                'message': '上传成功'
            })
        else:
            return jsonify({
                'code': -1,
                'message': f'不支持的文件类型，允许: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'code': -1,
            'message': f'上传失败: {str(e)}'
        }), 500


@app.route('/api/generate', methods=['POST'])
def generate_digital_human():
    """生成数字人视频"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'code': -1,
                'message': '请求数据为空'
            }), 400
        
        # 验证必填参数
        avatar_file = data.get('avatar')
        text = data.get('text', '').strip()
        
        if not avatar_file:
            return jsonify({
                'code': -1,
                'message': '请上传数字人形象照片'
            }), 400
        
        if not text:
            return jsonify({
                'code': -1,
                'message': '请输入播报文本'
            }), 400
        
        if len(text) > 5000:
            return jsonify({
                'code': -1,
                'message': '播报文本不能超过5000字'
            }), 400
        
        # 可选参数
        voice_id = data.get('voice', 'zh-CN-XiaoxiaoNeural')
        voice_prompt = (data.get('voicePrompt') or data.get('voice_prompt') or '').strip()
        bg_file = data.get('background')
        voice_ref_file = data.get('voiceReference')
        speed = float(data.get('speed', 1.0))
        prompt_text = data.get('promptText', '').strip()
        
        pitch_hz = int(data.get('pitch_hz', 0) or 0)
        volume_pct = int(data.get('volume_pct', 0) or 0)

        # 允许前端用 voice=auto（或空值）触发“描述选音色”
        if (not voice_id) or str(voice_id).strip().lower() == 'auto':
            if not voice_prompt:
                return jsonify({
                    'code': -1,
                    'message': '选择“自动匹配音色”时必须填写音色描述'
                }), 400
            resolution = resolve_voice(
                prompt=voice_prompt,
                text=text,
                voices=tts_engine.get_available_voices(),
                default_voice_id='zh-CN-XiaoxiaoNeural',
            )
            voice_id = resolution.voice_id
            # 若前端保持默认语速（1.0x），允许 resolver 根据描述给出更合适的语速
            if abs(speed - 1.0) < 1e-6 and abs(resolution.speed - 1.0) > 1e-6:
                speed = resolution.speed
            # 可选：若前端未显式传 pitch/volume，则采用 resolver 建议值
            if int(data.get('pitch_hz', 0) or 0) == 0:
                pitch_hz = resolution.pitch_hz
            if int(data.get('volume_pct', 0) or 0) == 0:
                volume_pct = resolution.volume_pct
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 记录任务
        with tasks_lock:
            tasks[task_id] = {
                'id': task_id,
                'status': 'pending',
                'progress': 0,
                'message': '任务创建成功，等待处理',
                'created_at': datetime.now().isoformat(),
                'result': None,
                'avatar': avatar_file,
                'background': bg_file,
                'text': text,
                'voice': voice_id,
                'voice_prompt': voice_prompt if voice_prompt else None,
            }
        
        # 异步处理任务
        thread = threading.Thread(
            target=process_generation_task,
            args=(task_id, avatar_file, text, voice_id, bg_file, voice_ref_file, speed, prompt_text, pitch_hz, volume_pct)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'code': 0,
            'data': {
                'task_id': task_id,
                'status': 'pending'
            },
            'message': '任务已创建'
        })
        
    except Exception as e:
        return jsonify({
            'code': -1,
            'message': f'创建任务失败: {str(e)}'
        }), 500


def process_generation_task(task_id, avatar_file, text, voice_id, bg_file, voice_ref_file, speed, prompt_text, pitch_hz, volume_pct):
    """处理生成任务的后台线程"""
    try:
        avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], avatar_file)
        bg_path = os.path.join(app.config['UPLOAD_FOLDER'], bg_file) if bg_file else None
        voice_ref_path = os.path.join(app.config['UPLOAD_FOLDER'], voice_ref_file) if voice_ref_file else None
        
        # 检查文件是否存在
        if not os.path.exists(avatar_path):
            update_task(task_id, 'failed', 0, '数字人形象照片不存在')
            return
        
        if bg_path and not os.path.exists(bg_path):
            update_task(task_id, 'failed', 0, '背景图片不存在')
            return
        
        # 步骤1: 生成语音
        update_task(task_id, 'processing', 10, '正在合成语音...')
        
        audio_filename = f"{task_id}_audio.mp3"
        audio_path = os.path.join(app.config['OUTPUT_FOLDER'], audio_filename)
        
        try:
            tts_engine.synthesize(
                text=text,
                voice=voice_id,
                output_path=audio_path,
                speed=speed,
                reference_audio=voice_ref_path,
                prompt_text=prompt_text,
                pitch_hz=pitch_hz,
                volume_pct=volume_pct,
            )
        except Exception as e:
            update_task(task_id, 'failed', 10, f'语音合成失败: {str(e)}')
            return
        
        if not os.path.exists(audio_path):
            update_task(task_id, 'failed', 10, '语音合成失败：未生成音频文件')
            return
        
        # 步骤2: 生成数字人视频
        update_task(task_id, 'processing', 40, '正在生成数字人视频...')
        
        video_filename = f"{task_id}_output.mp4"
        video_path = os.path.join(app.config['OUTPUT_FOLDER'], video_filename)
        
        try:
            video_generator.generate(
                avatar_path=avatar_path,
                audio_path=audio_path,
                output_path=video_path,
                background_path=bg_path
            )
        except Exception as e:
            update_task(task_id, 'failed', 40, f'视频生成失败: {str(e)}')
            return
        
        if not os.path.exists(video_path):
            update_task(task_id, 'failed', 40, '视频生成失败：未生成视频文件')
            return
        
        # 任务完成
        update_task(task_id, 'completed', 100, '生成完成', {
            'video_url': f'/static/outputs/{video_filename}',
            'audio_url': f'/static/outputs/{audio_filename}',
            'video_filename': video_filename
        })
        
        # 清理旧文件（保留最近24小时）
        cleanup_old_files(app.config['UPLOAD_FOLDER'], hours=24)
        cleanup_old_files(app.config['OUTPUT_FOLDER'], hours=24)
        
    except Exception as e:
        update_task(task_id, 'failed', 0, f'任务处理异常: {str(e)}')


def update_task(task_id, status, progress, message, result=None):
    """更新任务状态"""
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update({
                'status': status,
                'progress': progress,
                'message': message,
                'updated_at': datetime.now().isoformat()
            })
            if result is not None:
                tasks[task_id]['result'] = result


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    with tasks_lock:
        task = tasks.get(task_id)
    
    if not task:
        return jsonify({
            'code': -1,
            'message': '任务不存在'
        }), 404
    
    return jsonify({
        'code': 0,
        'data': task,
        'message': 'success'
    })


@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """获取任务列表"""
    with tasks_lock:
        task_list = sorted(
            tasks.values(),
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:20]
    
    return jsonify({
        'code': 0,
        'data': task_list,
        'message': 'success'
    })


@app.route('/static/outputs/<path:filename>')
def serve_output(filename):
    """提供输出文件访问"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'code': -1,
        'message': '文件太大，单个文件不能超过50MB'
    }), 413


if __name__ == '__main__':
    print("=" * 60)
    print("数字人播报系统启动中...")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
