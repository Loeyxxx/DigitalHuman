#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字人视频生成器
结合数字人照片、音频和可选背景图生成播报视频
"""

import os
import re
import subprocess
import wave

import cv2
import numpy as np
from PIL import Image
from imageio_ffmpeg import get_ffmpeg_exe


class VideoGenerator:
    """数字人视频生成器"""

    def __init__(self, fps=25):
        self.fps = fps
        self.ffmpeg = get_ffmpeg_exe()

    def generate(self, avatar_path, audio_path, output_path, background_path=None):
        """
        生成数字人播报视频
        GPU 环境使用 Wav2Lip 实现照片级嘴型同步，
        CPU 环境使用改进的音量驱动动画保证可用性。

        Args:
            avatar_path: 数字人形象照片路径
            audio_path: 音频文件路径
            output_path: 输出视频路径
            background_path: 背景图片路径（可选）

        Returns:
            output_path: 输出视频路径
        """
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            has_gpu = False

        if has_gpu:
            return self._generate_wav2lip(avatar_path, audio_path, output_path, background_path)
        else:
            return self._generate_fast(avatar_path, audio_path, output_path, background_path)

    def _generate_wav2lip(self, avatar_path, audio_path, output_path, background_path=None):
        """GPU 环境：使用 Wav2Lip 生成照片级嘴型同步视频"""
        import tempfile
        import shutil

        avatar_img = self._load_image(avatar_path)
        bg_img = self._load_image(background_path) if background_path else None

        if bg_img is not None:
            out_h, out_w = bg_img.shape[:2]
        else:
            out_h, out_w = avatar_img.shape[:2]

        out_h = out_h if out_h % 2 == 0 else out_h - 1
        out_w = out_w if out_w % 2 == 0 else out_w - 1

        # 计算头像在画布上的目标尺寸和位置
        avatar_processed = self._process_avatar(avatar_img, out_w, out_h)
        target_h, target_w = avatar_processed.shape[:2]

        # 先把原始头像缩放到目标尺寸，再做抠图
        # 这样 Wav2Lip 处理的尺寸 = 最终显示尺寸，彻底避免二次缩放损失
        # H.264/YUV420p 要求宽高为偶数，强制对齐避免 ffmpeg 自动裁剪导致尺寸错位
        target_h = target_h if target_h % 2 == 0 else target_h - 1
        target_w = target_w if target_w % 2 == 0 else target_w - 1

        if avatar_img.shape[:2] != (target_h, target_w):
            avatar_resized = cv2.resize(
                avatar_img, (target_w, target_h),
                interpolation=cv2.INTER_LANCZOS4
            )
        else:
            avatar_resized = avatar_img.copy()

        avatar_resized = self._remove_background(avatar_resized)

        # 分离 Alpha（Wav2Lip 只需要 RGB）
        has_alpha = False
        avatar_alpha = None
        if len(avatar_resized.shape) == 3 and avatar_resized.shape[2] == 4:
            has_alpha = True
            avatar_alpha = avatar_resized[:, :, 3:4]
            avatar_rgb = avatar_resized[:, :, :3]
        else:
            avatar_rgb = avatar_resized

        temp_dir = tempfile.gettempdir()
        temp_avatar_path = os.path.join(temp_dir, f'avatar_processed_{os.getpid()}.png')
        cv2.imwrite(temp_avatar_path, avatar_rgb)

        from .wav2lip_engine import Wav2LipEngine
        engine = Wav2LipEngine()

        temp_avatar_video = os.path.join(temp_dir, f'wav2lip_avatar_{os.getpid()}.mp4')

        try:
            # 静态图片场景：关闭检测框平滑，并只做一次人脸检测（引擎内部复用），可显著减少嘴部错位/分离问题
            engine.generate(temp_avatar_path, audio_path, temp_avatar_video, fps=self.fps, smooth=False)

            # 无背景：创建纯色画布，把 Wav2Lip 输出居中叠加
            if bg_img is None:
                cap = cv2.VideoCapture(temp_avatar_video)
                temp_final_video = output_path.replace('.mp4', '_temp.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(temp_final_video, fourcc, self.fps, (out_w, out_h))

                x = (out_w - target_w) // 2
                y = (out_h - target_h) // 2

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    bg = np.full((out_h, out_w, 3), (30, 30, 40), dtype=np.uint8)
                    if has_alpha:
                        frame = np.dstack([frame, avatar_alpha])
                    final_frame = self._overlay_image(bg, frame, x, y)
                    video_writer.write(final_frame)

                cap.release()
                video_writer.release()
                self._merge_audio_video(temp_final_video, audio_path, output_path)
                if os.path.exists(temp_final_video):
                    os.remove(temp_final_video)
                return output_path

            # 有背景：逐帧读取 Wav2Lip 输出，直接叠加到背景上（无需缩放）
            cap = cv2.VideoCapture(temp_avatar_video)
            if not cap.isOpened():
                raise RuntimeError("无法读取 Wav2Lip 生成的视频")

            temp_final_video = output_path.replace('.mp4', '_temp.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_final_video, fourcc, self.fps, (out_w, out_h))

            if not video_writer.isOpened():
                cap.release()
                raise RuntimeError("无法创建视频写入器")

            try:
                x = (out_w - target_w) // 2
                y = out_h - target_h - int(out_h * 0.05)

                while True:
                    ret, avatar_frame = cap.read()
                    if not ret:
                        break

                    # Wav2Lip 输出已经是目标尺寸，不需要再 resize
                    if has_alpha:
                        avatar_frame = np.dstack([avatar_frame, avatar_alpha])

                    bg_resized = cv2.resize(bg_img, (out_w, out_h))
                    final_frame = bg_resized.copy()
                    final_frame = self._overlay_image(final_frame, avatar_frame, x, y)
                    video_writer.write(final_frame)

            finally:
                cap.release()
                video_writer.release()

            self._merge_audio_video(temp_final_video, audio_path, output_path)
            if os.path.exists(temp_final_video):
                os.remove(temp_final_video)

        finally:
            if os.path.exists(temp_avatar_video):
                os.remove(temp_avatar_video)
            if os.path.exists(temp_avatar_path):
                os.remove(temp_avatar_path)

        return output_path

    def _generate_fast(self, avatar_path, audio_path, output_path, background_path=None):
        """CPU 环境：使用改进的音量驱动动画快速生成视频"""
        audio_duration = self._get_audio_duration(audio_path)
        total_frames = int(audio_duration * self.fps)

        avatar_img = self._load_image(avatar_path)
        bg_img = self._load_image(background_path) if background_path else None

        volume_envelope = self._get_volume_envelope(audio_path, total_frames)

        temp_video = output_path.replace('.mp4', '_temp.mp4')

        if bg_img is not None:
            out_h, out_w = bg_img.shape[:2]
        else:
            out_h, out_w = avatar_img.shape[:2]

        out_h = out_h if out_h % 2 == 0 else out_h - 1
        out_w = out_w if out_w % 2 == 0 else out_w - 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video, fourcc, self.fps, (out_w, out_h))

        if not video_writer.isOpened():
            raise RuntimeError("无法创建视频写入器")

        try:
            avatar_processed = self._process_avatar(avatar_img, out_w, out_h)

            for i in range(total_frames):
                volume = volume_envelope[i] if i < len(volume_envelope) else 0

                if bg_img is not None:
                    frame = self._create_frame_with_bg(
                        bg_img, avatar_processed, volume, out_w, out_h
                    )
                else:
                    frame = self._create_frame_no_bg(
                        avatar_processed, volume, out_w, out_h
                    )

                video_writer.write(frame)

        finally:
            video_writer.release()

        self._merge_audio_video(temp_video, audio_path, output_path)

        if os.path.exists(temp_video):
            os.remove(temp_video)

        return output_path

    def _remove_background(self, img):
        """
        自动去除图片背景，返回带 Alpha 通道的图像。
        需要安装 rembg 和 onnxruntime：
            pip install rembg onnxruntime
        未安装时直接返回原图。
        """
        if img is None:
            return None
        # 如果已经有 Alpha 通道，无需处理
        if len(img.shape) == 3 and img.shape[2] == 4:
            return img
        try:
            from rembg import remove
            from PIL import Image
            # OpenCV BGR -> PIL RGB
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = remove(pil_img)
            # PIL RGBA -> OpenCV BGRA
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGBA2BGRA)
        except ImportError:
            # 未安装 rembg，保持原样
            return img

    def _load_image(self, path):
        """加载图片（使用PIL避免中文路径问题）"""
        if path is None:
            return None
        # PIL 支持 UTF-8 路径，OpenCV 在 Windows 上不支持
        pil_img = Image.open(path)
        # 转换为 RGB/RGBA
        if pil_img.mode == 'RGBA':
            img = np.array(pil_img)
            # PIL 是 RGBA，OpenCV 需要 BGRA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        elif pil_img.mode == 'RGB':
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pil_img.mode == 'P':  # 调色板模式
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pil_img.mode == 'L':  # 灰度
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is None:
            raise ValueError(f"无法加载图片: {path}")
        return img

    def _get_audio_duration(self, audio_path):
        """使用ffmpeg获取音频时长"""
        cmd = [
            self.ffmpeg, '-i', audio_path,
            '-hide_banner', '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 从stderr中解析时长
        match = re.search(r'Duration:\s+(\d+):(\d+):(\d+\.\d+)', result.stderr)
        if match:
            hours = float(match.group(1))
            minutes = float(match.group(2))
            seconds = float(match.group(3))
            return hours * 3600 + minutes * 60 + seconds

        raise RuntimeError(f"无法获取音频时长: {result.stderr[:200]}")

    def _get_volume_envelope(self, audio_path, num_frames):
        """获取音频音量包络"""
        # 使用ffmpeg将音频转为wav
        temp_wav = audio_path + '_temp.wav'
        try:
            cmd = [
                self.ffmpeg, '-y', '-i', audio_path,
                '-ac', '1', '-ar', '16000',
                '-acodec', 'pcm_s16le',
                temp_wav
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                # 如果转换失败，返回零音量
                return np.zeros(num_frames)

            # 读取wav文件
            with wave.open(temp_wav, 'rb') as wf:
                n_frames = wf.getnframes()
                sr = wf.getframerate()
                raw_data = wf.readframes(n_frames)
                audio = np.frombuffer(raw_data, dtype=np.int16)

            # 计算每一帧对应的音量
            samples_per_frame = int(sr / self.fps)
            rms = []

            for i in range(num_frames):
                start = i * samples_per_frame
                end = min(start + samples_per_frame, len(audio))
                if end > start:
                    frame_audio = audio[start:end].astype(np.float32)
                    rms_val = np.sqrt(np.mean(frame_audio ** 2))
                    rms.append(rms_val)
                else:
                    rms.append(0)

            rms = np.array(rms)
            if len(rms) > 0 and np.max(rms) > 0:
                rms = rms / np.max(rms)

            return rms

        except Exception:
            return np.zeros(num_frames)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def _process_avatar(self, avatar_img, target_w, target_h):
        """预处理数字人头像"""
        h, w = avatar_img.shape[:2]

        # 计算缩放比例，保持头像在画面中央，占画面高度的70%
        scale = (target_h * 0.7) / h
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放头像
        avatar_resized = cv2.resize(avatar_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return avatar_resized

    def _create_frame_with_bg(self, bg_img, avatar_img, volume, target_w, target_h):
        """创建带背景的一帧"""
        # 调整背景尺寸
        bg_resized = cv2.resize(bg_img, (target_w, target_h))

        # 创建副本
        frame = bg_resized.copy()

        # 计算头像位置（居中偏下）
        avatar_h, avatar_w = avatar_img.shape[:2]
        x = (target_w - avatar_w) // 2
        y = target_h - avatar_h - int(target_h * 0.05)

        # 应用嘴型动画效果
        avatar_animated = self._apply_lip_animation(avatar_img, volume)

        # 将头像叠加到背景上
        frame = self._overlay_image(frame, avatar_animated, x, y)

        return frame

    def _create_frame_no_bg(self, avatar_img, volume, target_w, target_h):
        """创建无背景的一帧（使用纯色背景）"""
        # 创建深色背景
        frame = np.full((target_h, target_w, 3), (30, 30, 40), dtype=np.uint8)

        # 计算头像位置
        avatar_h, avatar_w = avatar_img.shape[:2]
        x = (target_w - avatar_w) // 2
        y = (target_h - avatar_h) // 2

        # 应用嘴型动画效果
        avatar_animated = self._apply_lip_animation(avatar_img, volume)

        # 将头像叠加到背景上
        frame = self._overlay_image(frame, avatar_animated, x, y)

        return frame

    def _apply_lip_animation(self, avatar_img, volume):
        """
        应用嘴型动画效果 - 改进版
        使用椭圆嘴洞、牙齿模拟、边缘羽化，让效果更自然
        """
        h, w = avatar_img.shape[:2]
        result = avatar_img.copy()

        # 嘴部中心定位
        mouth_y = int(h * 0.71)
        mouth_x = w // 2
        mouth_w = int(w * 0.26)
        mouth_h = int(h * 0.13)

        # 非线性映射张嘴幅度
        open_ratio = volume ** 0.6
        max_open = int(mouth_h * 0.5)
        open_height = int(max_open * open_ratio)

        if open_height < 2:
            return result

        # 嘴部区域边界
        y1 = max(0, mouth_y - mouth_h // 2)
        y2 = min(h, mouth_y + mouth_h // 2)
        x1 = max(0, mouth_x - mouth_w // 2)
        x2 = min(w, mouth_x + mouth_w // 2)

        if y2 <= y1 or x2 <= x1:
            return result

        region_h = y2 - y1
        region_w = x2 - x1
        channels = avatar_img.shape[2]

        # 分离上下唇
        upper_h = int(region_h * 0.42)
        lower_h = region_h - upper_h
        lip_gap = open_height
        new_region_h = region_h + lip_gap

        # 创建新区域：复制上下唇，中间先填充嘴唇过渡色（避免黑色条带）
        new_region = np.zeros((new_region_h, region_w, channels), dtype=avatar_img.dtype)
        new_region[0:upper_h, :] = result[y1:y1+upper_h, x1:x2]
        new_region[upper_h + lip_gap:, :] = result[y1+upper_h:y2, x1:x2]

        # 先填充中间区域为上下唇过渡色
        upper_lip_color = result[y1 + upper_h - 1, mouth_x, :3].astype(np.float32)
        lower_lip_color = result[y1 + upper_h, mouth_x, :3].astype(np.float32)
        skin_base = (upper_lip_color + lower_lip_color) / 2

        for yy in range(upper_h, upper_h + lip_gap):
            t = (yy - upper_h) / max(lip_gap, 1)
            trans_color = (upper_lip_color * (1 - t) + lower_lip_color * t).astype(np.uint8)
            new_region[yy, :, :3] = np.tile(trans_color, (region_w, 1))

        # ====== 创建椭圆嘴洞（代替矩形）======
        cavity_w = int(region_w * (0.55 + volume * 0.15))
        cavity_h = lip_gap
        cavity_cx = region_w // 2
        cavity_cy = upper_h + lip_gap // 2

        # 创建椭圆遮罩（羽化边缘）
        mask = np.zeros((new_region_h, region_w), dtype=np.float32)
        if cavity_h > 3 and cavity_w > 3:
            cv2.ellipse(mask, (cavity_cx, cavity_cy), (cavity_w // 2, cavity_h // 2),
                        0, 0, 360, 1.0, -1)
            blur_k = max(3, min(cavity_h // 2, 15))
            if blur_k % 2 == 0:
                blur_k += 1
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        # 嘴洞颜色：牙齿 + 口腔内部
        for yy in range(upper_h, upper_h + lip_gap):
            for xx in range(region_w):
                m = mask[yy, xx]
                if m < 0.01:
                    continue

                rel_y = (yy - upper_h) / max(lip_gap, 1)
                is_teeth = rel_y < 0.30

                if is_teeth:
                    # 牙齿：带弧度的白色（上唇内侧边缘更亮）
                    tooth_base = np.array([235, 220, 210], dtype=np.float32)
                    # 牙齿下方与口腔交界处略带阴影
                    tooth_shadow = 1.0 - (rel_y / 0.30) * 0.15
                    target_color = tooth_base * tooth_shadow
                else:
                    # 口腔：暗红色，带深度渐变
                    cavity_color = skin_base * 0.30
                    cavity_color[2] = min(255, cavity_color[2] * 1.5)  # 增强红色
                    cavity_color[1] = cavity_color[1] * 0.5
                    # 越深处越暗，两侧也略暗（模拟口腔曲面）
                    depth = (rel_y - 0.30) / 0.70
                    side_falloff = 1.0 - abs(xx - cavity_cx) / (region_w / 2 + 0.001) * 0.2
                    target_color = cavity_color * (1 - depth * 0.25) * side_falloff

                # 羽化混合
                original = new_region[yy, xx, :3].astype(np.float32)
                blended = (target_color * m + original * (1 - m)).astype(np.uint8)
                new_region[yy, xx, :3] = blended

        # ====== 嘴角微上扬 =====
        corner_lift = int(volume * 2.5)
        if corner_lift > 0 and upper_h > 5:
            for side in [0, 1]:  # 0=左, 1=右
                x_start = 0 if side == 0 else region_w * 2 // 3
                x_end = region_w // 3 if side == 0 else region_w
                patch = new_region[0:upper_h, x_start:x_end].copy()
                if patch.shape[1] > 0:
                    patch = cv2.resize(patch, (patch.shape[1], patch.shape[0] + corner_lift))
                    h_patch = min(patch.shape[0], upper_h)
                    new_region[0:h_patch, x_start:x_end] = patch[0:h_patch, :]

        # 放回原图
        new_y1 = max(0, mouth_y - new_region_h // 2)
        new_y2 = min(h, new_y1 + new_region_h)
        actual_h = new_y2 - new_y1
        if actual_h > 0:
            result[new_y1:new_y2, x1:x2] = new_region[0:actual_h, :]

        return result

    def _overlay_image(self, background, foreground, x, y):
        """将前景图片叠加到背景上"""
        h, w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # 计算实际叠加区域
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + w)
        y2 = min(bg_h, y + h)

        # 计算前景对应区域
        fx1 = max(0, -x)
        fy1 = max(0, -y)
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1 or fx2 <= fx1 or fy2 <= fy1:
            return background

        # 如果有Alpha通道，使用Alpha混合
        has_alpha = len(foreground.shape) == 3 and foreground.shape[2] == 4
        if has_alpha:
            # 提取RGB和Alpha
            fg_rgb = foreground[fy1:fy2, fx1:fx2, :3]
            fg_alpha = foreground[fy1:fy2, fx1:fx2, 3:4].astype(float) / 255.0
            bg_region = background[y1:y2, x1:x2].astype(float)

            # Alpha混合
            blended = (fg_rgb.astype(float) * fg_alpha + bg_region * (1 - fg_alpha)).astype(np.uint8)
            background[y1:y2, x1:x2] = blended
        else:
            # 直接覆盖
            background[y1:y2, x1:x2] = foreground[fy1:fy2, fx1:fx2]

        return background

    def _merge_audio_video(self, video_path, audio_path, output_path):
        """使用ffmpeg合并音频和视频"""
        cmd = [
            self.ffmpeg, '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'error',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg合并失败: {result.stderr}")
