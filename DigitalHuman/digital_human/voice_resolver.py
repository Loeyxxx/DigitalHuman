#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice resolver: choose the best voice from a fixed voice library by natural-language description.

- Default mode: heuristic keyword matching (no network, no extra deps).
- Optional mode: LLM (if configured) returns a structured JSON choice.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class VoiceResolution:
    voice_id: str
    speed: float = 1.0
    pitch_hz: int = 0
    volume_pct: int = 0
    confidence: float = 0.5
    rationale: str = ""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _voice_by_id(voices: list[dict[str, Any]], voice_id: str) -> dict[str, Any] | None:
    for v in voices:
        if v.get("id") == voice_id:
            return v
    return None


def resolve_voice(
    *,
    prompt: str,
    voices: list[dict[str, Any]],
    default_voice_id: str = "zh-CN-XiaoxiaoNeural",
    text: str | None = None,
) -> VoiceResolution:
    """
    Resolve a voice by natural-language prompt.

    If LLM mode is enabled, uses LLM; otherwise uses heuristic resolver.
    """
    mode = os.environ.get("VOICE_RESOLVER_MODE", "heuristic").strip().lower()
    if mode == "llm":
        try:
            return _resolve_voice_llm(prompt=prompt, text=text, voices=voices, default_voice_id=default_voice_id)
        except Exception:
            # Hard fallback to heuristic for robustness.
            pass

    return _resolve_voice_heuristic(prompt=prompt, text=text, voices=voices, default_voice_id=default_voice_id)


def _resolve_voice_heuristic(
    *,
    prompt: str,
    voices: list[dict[str, Any]],
    default_voice_id: str,
    text: str | None,
) -> VoiceResolution:
    p = _norm(prompt)
    t = _norm(text or "")

    # Dialect first (most deterministic given current library).
    dialect_candidates: list[str] = []
    if any(k in p for k in ["粤语", "广东", "港式", "香港", "粤"]):
        dialect_candidates.append("zh-HK-HiuMaanNeural")
    if any(k in p for k in ["台湾", "台式", "台灣", "台语", "台語"]):
        dialect_candidates.append("zh-TW-HsiaoChenNeural")
    if any(k in p for k in ["东北", "东百", "辽宁"]):
        dialect_candidates.append("zh-CN-liaoning-XiaobeiNeural")
    if any(k in p for k in ["陕西", "西安", "陕"]):
        dialect_candidates.append("zh-CN-shaanxi-XiaoniNeural")

    # Gender/age/style cues.
    wants_male = any(k in p for k in ["男声", "男性", "男"])
    wants_female = any(k in p for k in ["女声", "女性", "女"])
    wants_young = any(k in p for k in ["年轻", "少年", "男孩", "女孩", "青春", "大学生"])
    wants_mature = any(k in p for k in ["成熟", "沉稳", "稳重", "大气", "大叔", "中年", "商务", "正式"])
    wants_news = any(k in p for k in ["新闻", "播报", "主持", "旁白", "解说", "纪录片"])
    wants_story = any(k in p for k in ["故事", "童话", "儿童", "绘本", "哄睡"])
    wants_warm = any(k in p for k in ["温柔", "亲切", "暖", "治愈", "甜美"])
    wants_lively = any(k in p for k in ["活泼", "元气", "俏皮", "可爱"])

    # Speed from explicit request.
    speed = 1.0
    speed_match = re.search(r"(\d+(?:\.\d+)?)\s*[x×]", p)
    if speed_match:
        speed = float(speed_match.group(1))
    elif any(k in p for k in ["慢一点", "慢点", "放慢", "慢速"]):
        speed = 0.9
    elif any(k in p for k in ["快一点", "快点", "加快", "快速"]):
        speed = 1.1
    speed = float(_clamp(speed, 0.5, 2.0))

    # Pick from available voices. Current library is small, so we bias to known “archetypes”.
    preferred: list[str] = []

    # If dialect matched, prioritize it.
    preferred.extend(dialect_candidates)

    if wants_news or wants_mature:
        preferred.extend(["zh-CN-YunyangNeural", "zh-CN-YunjianNeural", "zh-CN-XiaoxiaoNeural"])
    if wants_story or wants_warm:
        preferred.extend(["zh-CN-XiaoyiNeural", "zh-CN-XiaoxiaoNeural"])
    if wants_lively or wants_young:
        preferred.extend(["zh-CN-YunxiNeural", "zh-CN-YunxiaNeural", "zh-CN-XiaoxiaoNeural"])

    if wants_male and not wants_female:
        preferred.extend(["zh-CN-YunyangNeural", "zh-CN-YunjianNeural", "zh-CN-YunxiNeural", "zh-CN-YunxiaNeural"])
    if wants_female and not wants_male:
        preferred.extend(["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural"])

    # If prompt indicates language/region but no dialect keyword, still use generic Mandarin voices.
    if not preferred:
        if any(k in p for k in ["正式", "专业", "商务"]):
            preferred.extend(["zh-CN-YunyangNeural", "zh-CN-YunjianNeural"])
        else:
            preferred.extend([default_voice_id])

    # Choose first that exists in current voices list; else fallback to default.
    chosen = None
    for vid in preferred:
        if _voice_by_id(voices, vid) is not None:
            chosen = vid
            break
    if chosen is None:
        chosen = default_voice_id

    # Simple confidence heuristic.
    confidence = 0.55
    if dialect_candidates and chosen in dialect_candidates:
        confidence = 0.85
    elif ("新闻" in p or "播报" in p) and chosen in ["zh-CN-YunyangNeural", "zh-CN-YunjianNeural", "zh-CN-XiaoxiaoNeural"]:
        confidence = 0.75

    rationale_parts: list[str] = []
    if dialect_candidates:
        rationale_parts.append("命中方言/口音关键词")
    if wants_news:
        rationale_parts.append("偏新闻/播报风格")
    if wants_story:
        rationale_parts.append("偏故事/儿童风格")
    if wants_male ^ wants_female:
        rationale_parts.append("命中性别偏好")

    rationale = "；".join(rationale_parts) if rationale_parts else "默认匹配"
    if t and any(k in t for k in ["公告", "通知", "播报", "新闻"]):
        # Slightly bias toward clearer diction for “announcement” texts.
        rationale = (rationale + "；文本偏正式") if rationale else "文本偏正式"

    return VoiceResolution(
        voice_id=chosen,
        speed=speed,
        pitch_hz=0,
        volume_pct=0,
        confidence=float(_clamp(confidence, 0.0, 1.0)),
        rationale=rationale,
    )


def _resolve_voice_llm(
    *,
    prompt: str,
    voices: list[dict[str, Any]],
    default_voice_id: str,
    text: str | None,
) -> VoiceResolution:
    """
    Optional LLM resolver.

    Requirements at runtime:
    - `openai` python package installed
    - `OPENAI_API_KEY` env var set
    - Optional: `OPENAI_MODEL` env var, default 'gpt-4.1-mini'
    """
    # Lazy import so heuristic mode stays dependency-free.
    from openai import OpenAI  # type: ignore

    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()

    client = OpenAI()

    voice_candidates = [
        {
            "id": v.get("id"),
            "name": v.get("name"),
            "gender": v.get("gender"),
            "style": v.get("style"),
            "description": v.get("description"),
        }
        for v in voices
    ]

    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "voice_id": {"type": "string"},
            "speed": {"type": "number"},
            "pitch_hz": {"type": "integer"},
            "volume_pct": {"type": "integer"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["voice_id", "speed", "pitch_hz", "volume_pct", "confidence", "rationale"],
    }

    user_payload = {
        "prompt": prompt,
        "text": text or "",
        "voices": voice_candidates,
        "constraints": {
            "speed_range": [0.5, 2.0],
            "pitch_hz_range": [-50, 50],
            "volume_pct_range": [-50, 50],
            "must_choose_from_voice_ids": [v["id"] for v in voice_candidates if v.get("id")],
            "default_voice_id": default_voice_id,
        },
    }

    # Note: This is a best-effort integration. If your environment uses a different LLM provider,
    # keep the same JSON contract and swap out the call.
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "你是一个“音色选择器”。任务：根据用户的音色描述与播报文本，"
                    "从给定 voices 列表中选择最合适的 voice_id，并给出建议的 speed/pitch_hz/volume_pct。"
                    "必须只输出符合 JSON schema 的对象，不要输出多余文字。"
                ),
            },
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        # Responses API uses `text.format` for structured outputs / JSON mode.
        # Some OpenAI SDK versions do not accept `response_format` here.
        text={"format": {"type": "json_schema", "name": "voice_resolution", "schema": schema, "strict": True}},
    )

    # OpenAI SDK returns output text in resp.output_text; keep parsing robust.
    raw = getattr(resp, "output_text", None) or ""
    if not raw:
        # Fallback for SDKs that don't expose output_text.
        try:
            raw = resp.output[0].content[0].text  # type: ignore[attr-defined]
        except Exception:
            raw = ""

    # If structured outputs isn't supported by the selected model, retry with plain JSON mode.
    if not raw:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是一个“音色选择器”。任务：根据用户的音色描述与播报文本，"
                        "从给定 voices 列表中选择最合适的 voice_id，并给出建议的 speed/pitch_hz/volume_pct。"
                        "必须只输出 JSON 对象，不要输出多余文字。"
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

    data = json.loads(raw)

    voice_id = data.get("voice_id") or default_voice_id
    if _voice_by_id(voices, voice_id) is None:
        voice_id = default_voice_id

    return VoiceResolution(
        voice_id=voice_id,
        speed=float(_clamp(float(data.get("speed", 1.0)), 0.5, 2.0)),
        pitch_hz=int(_clamp(int(data.get("pitch_hz", 0)), -50, 50)),
        volume_pct=int(_clamp(int(data.get("volume_pct", 0)), -50, 50)),
        confidence=float(_clamp(float(data.get("confidence", 0.5)), 0.0, 1.0)),
        rationale=str(data.get("rationale", ""))[:500],
    )
