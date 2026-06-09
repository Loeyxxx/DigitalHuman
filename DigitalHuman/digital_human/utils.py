#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import os
import time
from pathlib import Path


def allowed_file(filename, allowed_extensions):
    """检查文件类型是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_upload_path(base_path, filename):
    """获取上传文件的完整路径"""
    return os.path.join(base_path, filename)


def cleanup_old_files(directory, hours=24):
    """清理指定目录下超过指定时间的旧文件"""
    try:
        current_time = time.time()
        max_age = hours * 3600
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age:
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
    except Exception:
        pass


def ensure_dir(directory):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)
