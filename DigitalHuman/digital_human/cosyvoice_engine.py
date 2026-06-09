#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice 语音克隆引擎（HTTP）
通过调用独立运行的 CosyVoice FastAPI 服务实现 zero-shot 声音克隆。
"""

import io
import os
import wave
import tempfile
import subprocess

import requests


class CosyVoiceEngine:
    """CosyVoice zero-shot 引擎"""

    def __init__(self, api_base: str = None, target_sample_rate: int = 22050, timeout: int = 180):
        self.api_base = (api_base or os.getenv("COSYVOICE_API", "http://127.0.0.1:50000")).rstrip("/")
        self.target_sample_rate = target_sample_rate
        self.timeout = timeout
        self.endpoint_zero_shot = f"{self.api_base}/inference_zero_shot"

    def synthesize_zero_shot(self, text: str, prompt_wav_path: str, output_path: str, prompt_text: str = "", speed: float = 1.0):
        """
        调用 CosyVoice zero-shot 合成。

        Args:
            text: 目标播报文本
            prompt_wav_path: 参考音频路径
            output_path: 输出音频路径（建议 mp3）
            prompt_text: 参考音频对应文本（可空）
            speed: 保留参数，当前通过后处理 atempo 近似实现
        """
        if not text or not text.strip():
            raise ValueError("text 不能为空")
        if not prompt_wav_path or not os.path.exists(prompt_wav_path):
            raise ValueError("参考音频不存在")
        if not output_path:
            raise ValueError("output_path 不能为空")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 1) 调用 CosyVoice，返回 PCM int16 字节流
        with open(prompt_wav_path, "rb") as f:
            files = {
                "prompt_wav": (os.path.basename(prompt_wav_path), f, "application/octet-stream")
            }
            data = {
                "tts_text": text,
                "prompt_text": (prompt_text.strip() if isinstance(prompt_text, str) and prompt_text.strip() else "这是一段用于声音克隆的参考音频。")
            }
            resp = requests.post(
                self.endpoint_zero_shot,
                data=data,
                files=files,
                timeout=self.timeout,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"CosyVoice 请求失败: HTTP {resp.status_code} - {resp.text[:300]}")

        pcm_bytes = resp.content
        if not pcm_bytes:
            raise RuntimeError("CosyVoice 返回空音频")

        # 2) 先把裸 PCM 封装成临时 wav（1ch, s16le, 22050）
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "cosyvoice_output.wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(self.target_sample_rate)
                wf.writeframes(pcm_bytes)

            # 3) 可选语速处理（ffmpeg atempo 0.5~2.0）
            speed = float(speed or 1.0)
            speed = max(0.5, min(2.0, speed))

            src_for_convert = wav_path
            if abs(speed - 1.0) > 1e-6:
                sped_wav_path = os.path.join(tmpdir, "cosyvoice_output_sped.wav")
                self._run_ffmpeg([
                    "ffmpeg", "-y", "-i", wav_path,
                    "-filter:a", f"atempo={speed}",
                    sped_wav_path,
                ])
                src_for_convert = sped_wav_path

            # 4) 转成 mp3，与现有 DigitalHuman 下游保持一致
            self._run_ffmpeg([
                "ffmpeg", "-y", "-i", src_for_convert,
                "-codec:a", "libmp3lame", "-q:a", "2",
                output_path,
            ])

        return output_path

    @staticmethod
    def _run_ffmpeg(cmd):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"ffmpeg 执行失败: {stderr[:500]}")