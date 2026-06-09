#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wav2Lip 推理引擎封装
基于 Wav2Lip 实现照片级嘴型同步
"""

import os
import sys
import cv2
import numpy as np
import torch
import subprocess

# 将 wav2lip 包加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from digital_human.wav2lip import audio
from digital_human.wav2lip.models import Wav2Lip
from digital_human.wav2lip.face_detection import FaceAlignment, LandmarksType
from imageio_ffmpeg import get_ffmpeg_exe


class Wav2LipEngine:
    """Wav2Lip 嘴型同步引擎"""

    def __init__(self, checkpoint_path=None, face_detect_batch=8, wav2lip_batch=64):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detect_batch = face_detect_batch
        self.wav2lip_batch = wav2lip_batch
        self.ffmpeg = get_ffmpeg_exe()

        # 模型路径
        if checkpoint_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(base_dir, 'wav2lip', 'models', 'wav2lip_gan.pth')
        self.checkpoint_path = checkpoint_path

        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 Wav2Lip 模型

        支持两种格式：
        - PyTorch checkpoint（通常是 .pth/.pt，且包含 state_dict）
        - TorchScript archive（通常是 torch.jit.save 导出的 .pt）

        注意：TorchScript archive 与 checkpoint 不是同一种文件。
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Wav2Lip 模型未找到: {self.checkpoint_path}\n"
                f"请从以下地址下载并放入该路径:\n"
                f"https://huggingface.co/spaces/Rudrabha/Wav2Lip/resolve/main/wav2lip_gan.pth"
            )

        # 1) 优先尝试按 TorchScript archive 加载（避免 PyTorch>=2.6 的 weights_only 默认值冲突）
        # TorchScript archive 需要用 torch.jit.load，而不是 torch.load(state_dict)。
        try:
            scripted = torch.jit.load(self.checkpoint_path, map_location=self.device)
            scripted.eval()
            self.model = scripted
            return
        except Exception:
            pass

        # 2) 回退到传统 checkpoint（dict + state_dict）
        self.model = Wav2Lip()

        # PyTorch 2.6 将 torch.load 的 weights_only 默认从 False 改为 True；
        # 这里显式设置为 True，优先读取 tensor 权重，避免不必要的反序列化风险。
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            # 兼容旧版本 torch（无 weights_only 参数）
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                "Wav2Lip 模型文件格式不兼容：需要 TorchScript archive，或包含 'state_dict' 的 checkpoint 字典。"
            )

        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        self.model.load_state_dict(new_s)
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        face_path,
        audio_path,
        output_path,
        fps=25,
        resize_factor=1,
        pads=(0, 20, 0, 0),
        smooth=True,
    ):
        """
        生成嘴型同步视频

        Args:
            face_path: 头像图片路径（jpg/png）
            audio_path: 音频文件路径（mp3/wav）
            output_path: 输出视频路径
            fps: 帧率
            resize_factor: 分辨率缩放因子（1为原始大小，2为一半大小）
            pads: 人脸检测 padding (上, 下, 左, 右)
            smooth: 是否对检测框做时间平滑（静态图片建议关闭以避免抖动/错位）
        """
        # CPU 环境自动优化
        if self.device == 'cpu' and resize_factor == 1:
            print('[Wav2Lip] CPU 环境 detected，自动启用快速模式 (resize_factor=2, fps=15)')
            resize_factor = 2
            fps = 15
        # 读取静态图片（使用 Pillow 避免 Windows 中文路径问题）
        from PIL import Image
        pil_img = Image.open(face_path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        full_frames = [cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)]

        # 如果需要缩放
        if resize_factor > 1:
            full_frames = [
                cv2.resize(f, (f.shape[1] // resize_factor, f.shape[0] // resize_factor))
                for f in full_frames
            ]

        # 音频转 wav（Wav2Lip 需要 wav 格式）
        temp_wav = audio_path.replace('.mp3', '_temp.wav').replace('.m4a', '_temp.wav')
        if not audio_path.endswith('.wav'):
            cmd = [
                self.ffmpeg, '-y', '-i', audio_path,
                '-strict', '-2', '-ar', '16000', '-ac', '1',
                temp_wav
            ]
            subprocess.call(cmd, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio_input = temp_wav
        else:
            audio_input = audio_path

        # 加载音频并计算 mel spectrogram
        wav = audio.load_wav(audio_input, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel spectrogram 包含 NaN')

        # 将 mel 分块
        mel_step_size = 16
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1

        # 确保帧数与 mel 块数匹配（静态图片复制多帧）
        full_frames = full_frames * len(mel_chunks)
        full_frames = full_frames[:len(mel_chunks)]

        # 人脸检测
        # 静态图片：只检测一次，再复用检测框，避免逐帧检测导致的轻微抖动/错位（常见为“嘴和脸分离/双嘴”）。
        face_det_first = self._face_detect([full_frames[0]], pads, smooth=smooth)[0]
        face_det_results = [face_det_first for _ in range(len(full_frames))]

        # 批量推理
        batch_size = self.wav2lip_batch
        gen = self._datagen(full_frames.copy(), mel_chunks, face_det_results)

        frame_h, frame_w = full_frames[0].shape[:-1]
        temp_video = output_path.replace('.mp4', '_temp.avi')
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        # 合并音频
        cmd = [
            self.ffmpeg, '-y', '-i', audio_input, '-i', temp_video,
            '-strict', '-2', '-q:v', '1', '-c:v', 'libx264', '-c:a', 'aac',
            '-pix_fmt', 'yuv420p', output_path
        ]
        subprocess.call(cmd, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 清理临时文件
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if temp_wav != audio_path and os.path.exists(temp_wav):
            os.remove(temp_wav)

        return output_path

    def _face_detect(self, images, pads, smooth=True):
        """检测人脸位置"""
        detector = FaceAlignment(
            LandmarksType._2D, flip_input=False,
            device=self.device, face_detector='sfd'
        )

        batch_size = self.face_detect_batch
        predictions = []

        for i in range(0, len(images), batch_size):
            batch = np.array(images[i:i + batch_size])
            predictions.extend(detector.get_detections_for_batch(batch))

        pady1, pady2, padx1, padx2 = pads
        results = []
        for rect, image in zip(predictions, images):
            if rect is None:
                raise ValueError('未检测到人脸！请确保图片中包含清晰的人脸。')

            x1, y1, x2, y2 = rect
            y1 = max(0, y1 - pady1)
            y2 = min(image.shape[0], y2 + pady2)
            x1 = max(0, x1 - padx1)
            x2 = min(image.shape[1], x2 + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        # 平滑检测框（视频场景有帮助；静态图片通常建议关闭）
        if smooth:
            boxes = self._get_smoothened_boxes(boxes, T=5)
        results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results

    @staticmethod
    def _get_smoothened_boxes(boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def _datagen(self, frames, mels, face_det_results):
        """生成训练批次数据"""
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        img_size = 96

        for i, m in enumerate(mels):
            frame_to_save = frames[i].copy()
            face, coords = face_det_results[i].copy()

            face = cv2.resize(face, (img_size, img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, img_size // 2:, :] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:, :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
