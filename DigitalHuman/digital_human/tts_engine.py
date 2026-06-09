#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS语音合成引擎
默认使用 edge-tts；当传入参考音频时切换到 CosyVoice zero-shot 声音克隆。
"""

import os
import asyncio
import edge_tts

from .cosyvoice_engine import CosyVoiceEngine


class TTSEngine:
    """TTS语音合成引擎"""

    # 预设的中文音色列表
    DEFAULT_VOICES = [
        {"id": "zh-CN-XiaoxiaoNeural", "name": "晓晓", "gender": "女", "style": "活泼温暖", "description": "年轻女性，适合新闻播报、客服等场景"},
        {"id": "zh-CN-XiaoyiNeural", "name": "晓伊", "gender": "女", "style": "温柔甜美", "description": "年轻女性，适合故事讲述、儿童内容"},
        {"id": "zh-CN-YunjianNeural", "name": "云健", "gender": "男", "style": "沉稳大气", "description": "成年男性，适合新闻播报、纪录片解说"},
        {"id": "zh-CN-YunxiNeural", "name": "云希", "gender": "男", "style": "年轻阳光", "description": "年轻男性，适合动漫、游戏角色"},
        {"id": "zh-CN-YunxiaNeural", "name": "云夏", "gender": "男", "style": "少年清亮", "description": "少年男性，适合青少年内容"},
        {"id": "zh-CN-YunyangNeural", "name": "云扬", "gender": "男", "style": "专业正式", "description": "成年男性，适合专业播报、商务场景"},
        {"id": "zh-CN-liaoning-XiaobeiNeural", "name": "晓北", "gender": "女", "style": "东北方言", "description": "东北口音女性，带有地域特色"},
        {"id": "zh-CN-shaanxi-XiaoniNeural", "name": "晓妮", "gender": "女", "style": "陕西方言", "description": "陕西口音女性，带有地域特色"},
        {"id": "zh-HK-HiuMaanNeural", "name": "晓曼", "gender": "女", "style": "粤语", "description": "粤语女性，适合香港地区内容"},
        {"id": "zh-TW-HsiaoChenNeural", "name": "晓晨", "gender": "女", "style": "台湾腔", "description": "台湾口音女性，适合台湾地区内容"},
    ]

    def __init__(self):
        self.voices = self.DEFAULT_VOICES
        self.clone_engine = CosyVoiceEngine()

    def get_available_voices(self):
        """获取可用的音色列表"""
        return self.voices

    def synthesize(self, text, voice="zh-CN-XiaoxiaoNeural", output_path=None, speed=1.0, reference_audio=None,prompt_text=""):
        """
        合成语音

        Args:
            text: 要合成的文本
            voice: 音色ID（edge-tts 路径生效）
            output_path: 输出文件路径
            speed: 语速，0.5-2.0
            reference_audio: 参考音频路径（传入时走 CosyVoice 克隆）

        Returns:
            output_path: 输出文件路径
        """
        if not output_path:
            raise ValueError("必须指定输出路径")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 有参考音频 => CosyVoice zero-shot
        if reference_audio and os.path.exists(reference_audio):
            return self.clone_engine.synthesize_zero_shot(
                text=text,
                prompt_wav_path=reference_audio,
                output_path=output_path,
                prompt_text=prompt_text,
                speed=speed,
            )

        # 无参考音频 => edge-tts
        rate_value = int((speed - 1) * 100)
        rate_param = f"{'+' if rate_value >= 0 else ''}{rate_value}%"
        asyncio.run(self._do_synthesize(text, voice, output_path, rate_param))
        return output_path

    async def _do_synthesize(self, text, voice, output_path, rate_param):
        """执行异步语音合成"""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate_param,
            volume="+0%",
            pitch="+0Hz"
        )
        await communicate.save(output_path)