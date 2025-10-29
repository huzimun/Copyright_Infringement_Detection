#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Courtroom simulator that coordinates LVLM-driven agents:
 - Third-party Expert (uses abstraction-filter-compare chain)
 - Plaintiff Lawyer (argues for infringement)
 - Defendant Lawyer (argues against infringement)
 - Judge (evaluates statements and issues verdict)

This file re-uses the LVLM adapter in My-CopyJudge/judge.py (LLMAdapterAgent,
PromptTemplates, FinalDecision, Judgment).

The simulator runs iterative rounds of statements until the judge issues a
decision or the max rounds is reached.
"""

import os
import json
import base64
import random
import argparse
import shutil
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
# Qwen2.5-VL本地加载
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 全局Qwen模型和processor（只加载一次）
GLOBAL_QWEN_MODEL = None
GLOBAL_QWEN_PROCESSOR = None

# 数据结构
# ======================

@dataclass
class AbstractionOutput:
    z: str
    z_cr: str

@dataclass
class FilteringOutput:
    zc: str
    zccr: str

@dataclass
class Judgment:
    score: float
    confidence: float
    rationale: str

@dataclass
class FinalDecision:
    score_final: float
    confidence_final: float
    rationale_final: str
    is_infringement: bool

class ExpertAgent:
    """Third-party expert which performs abstraction->filter->compare chain.

    Knowledge store K_expert = {E_e, O}
      - E_e: list of reflections produced by the expert based on the full `global_history` of the current case
      - O: list of AF-C outputs from different cases (abstraction, filtration, judgment)
    """

    def __init__(self, cfg: dict):
        self.agent = LLMAdapterAgent(cfg)
        self.cfg = cfg
        # E_e: per-case reflections (strings)
        # O: list of dicts {case_id, abstraction, filtration, judgment}
        self.K_expert = {'E_e': [], 'O': []}

    def analyze(self, image_x: str, image_xcr: str, human_refs: Optional[List[tuple]] = None, case_id: Optional[str] = None):
        """Run abstraction, filtration and comparison. Return dict with outputs and optionally store O."""
        log("Expert: running abstraction")
        abstraction = self.agent.abstract(image_x, image_xcr, PromptTemplates.ABSTRACTION)
        log(f"Expert abstraction: {abstraction[:200]}")
        log("Expert: running filtration")
        filtration = self.agent.filter(abstraction, PromptTemplates.FILTRATION)
        log(f"Expert filtration: {filtration[:200]}")
        log("Expert: running comparison")
        judgment = self.agent.compare(image_x, image_xcr, filtration, PromptTemplates.COMPARISON, human_refs)
        log(f"Expert comparison: score={judgment.score}, confidence={judgment.confidence}")
        out = {
            'abstraction': abstraction,
            'filtration': filtration,
            'judgment': judgment,
        }

        # store AF-C output in O (all expert outputs are kept, optionally could filter by confidence)
        try:
            entry = {
                'case_id': case_id or os.path.basename(image_x),
                'abstraction': abstraction,
                'filtration': filtration,
                'judgment': {'score': judgment.score, 'confidence': judgment.confidence, 'rationale': judgment.rationale}
            }
            self.K_expert['O'].append(entry)
        except Exception as e:
            log(f"Failed to store AF-C output in K_expert['O'] for case {case_id or os.path.basename(image_x)}: {e}")
            raise

        return out
    confidence_final: float
    rationale_final: str
    is_infringement: bool

    def reflect_and_summary(self, case_id: Optional[str] = None, global_history: Optional[List[dict]] = None, final_decision: Optional[FinalDecision] = None) -> str:
        """Expert reflects using only the full case `global_history` and the `final_decision`.

        Stores a short reflection into K_expert['E_e'] and returns the summary string.
        """
        if not self.cfg.get('enable_reflection_summary', True):
            return ""
        prompt = (
            "You are the Expert agent. Given the full dialog/history of the case and the final decision, reflect briefly (2-4 sentences) on which image features and comparisons influenced your judgment and any cautionary notes for future cases.\n\n"
        )
        try:
            if global_history and isinstance(global_history, list):
                tail = global_history[-12:]
                prompt += "Recent dialog/events (last few):\n" + json.dumps(tail, ensure_ascii=False, indent=0) + "\n\n"
        except Exception as e:
            log(f"Failed to include global_history tail in expert reflect prompt for case {case_id}: {e}")
            raise
        try:
            if final_decision:
                prompt += f"Final Decision: is_infringement={getattr(final_decision, 'is_infringement', '')}, score={getattr(final_decision, 'score_final', '')}, confidence={getattr(final_decision, 'confidence_final', '')}, rationale={getattr(final_decision, 'rationale_final', '')}\n"
        except Exception as e:
            log(f"Failed to include final_decision in expert reflect prompt for case {case_id}: {e}")
            raise
        try:
            summary = self.agent._chat(prompt).strip()
        except Exception as e:
            log(f"Expert reflect_and_summary failed: {e}")
            summary = ""
        try:
            next_id = len(self.K_expert['E_e']) + 1
            self.K_expert['E_e'].append({'case_id': case_id or f'case_{next_id}', 'summary': summary})
        except Exception as e:
            log(f"Failed to store expert reflection in K_expert['E_e'] for case {case_id}: {e}")
            raise
        return summary

# ======================
# In-context Learning Utils
# ======================

def load_in_context_examples(jsonl_path: str, base_dir: str = None, max_examples: int = 4):
    """
    Load in-context examples from a jsonl file.
    Each line: {gen_image, real_image, image_name, level}
    level: 0->0.0, 1->0.2, ..., 5->1.0
    Returns: List of tuples (gen_image_path, real_image_path, score)
    """
    level2score = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
    examples = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                gen_img = item['gen_image']
                real_img = item['real_image']
                level = item['level']
                score = level2score.get(level, 0.0)
                if base_dir:
                    gen_img = os.path.join(base_dir, gen_img)
                    real_img = os.path.join(base_dir, real_img)
                examples.append((gen_img, real_img, score))
                if len(examples) >= max_examples:
                    break
    except Exception as e:
        log(f"Failed to load in-context examples: {e}")
    return examples

# ======================
# 配置加载
# ======================
def load_config(path: str = "config.json") -> Dict[str, Any]:
    log(f"Loading config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ======================
# Prompt Templates (论文附录 A.1)
# ======================
class PromptTemplates:
    ABSTRACTION = (
        "Please help decompose the given two images into their abstract concepts. "
        "Identify key components such as composition, themes, color palette, textures, "
        "and any unique visual elements. Describe each component in detail. "
        "Ensure the output follows the template format, do not output any redundant content: "
        "'Image1: XXX, Image2: XXX'."
    )

    FILTRATION = (
        "Based on the image decomposition, please identify which elements are likely "
        "unprotectable under copyright law. These could include generic concepts, "
        "common patterns, functional aspects, or elements derived from public domain sources. "
        "Then, specify the remaining unique elements that are original and creative. "
        "Ensure the output follows the template format, do not output any redundant content: "
        "'Image1 Unique Elements: XXX, Image2 Unique Elements: XXX'."
    )

    COMPARISON = (
        "Please assign an overall similarity score (0–1) between the two images based on the unique elements. "
        "The score should be higher if there are similar elements. Provide a confidence score on a scale of 0 to 1, "
        "and a concise rationale for whether this case constitutes likely copyright infringement. "
        "Output strictly as: 'Score: [0-1], Confidence: [0-1], Reason: [text]'."
    )

    FEEDBACK = (
        "The following is feedback from other agents:\n{peers}\n"
        "Your previous judgment: {self_judgment}\n"
        "You may adjust your score based on this information or maintain your judgment. "
        "Output strictly as: 'Score: [0-1], Confidence: [0-1], Reason: [text]'."
    )

    META = (
        "You are the meta-judge. The original instruction is the comparison and scoring prompt. "
        "Here are the consensus-round scores, confidences, and rationales from multiple judges:\n{summary}\n"
        "Provide the final Score, Confidence, and Reason as: "
        "'Score: [0-1], Confidence: [0-1], Reason: [text]'."
    )

# ======================
# 抽象 LVLM 接口
# ======================

class BaseLVLM(ABC):
    @abstractmethod
    def abstract(self, image_x: Any, image_xcr: Any, prompt: str) -> AbstractionOutput: ...
    @abstractmethod
    def filter(self, abs_out: AbstractionOutput, prompt: str) -> FilteringOutput: ...
    @abstractmethod
    def compare(self, image_x: Any, image_xcr: Any,
                filter_zc_zccr: str, prompt: str,
                human_refs: Optional[List[Tuple[Any, Any, float]]] = None) -> Judgment: ...

class LLMAdapterAgent(BaseLVLM):
    """
    通用 LVLM 适配器，支持 GPT-4o (OpenAI API) 与 Qwen3 (本地 HTTP)。
    根据 cfg["agent_type"] 自动选择后端。
    """

    def __init__(self, cfg: Dict[str, Any], mbti: str = None):
        log(f"Initializing LLMAdapterAgent with agent_type={cfg['agent_type']}, mbti={mbti}")
        self.cfg = cfg
        self.agent_type = cfg["agent_type"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self.device = cfg["device"]
        self.mbti = mbti

        global GLOBAL_QWEN_MODEL, GLOBAL_QWEN_PROCESSOR

        if "gpt-4o" in self.agent_type:
            api_key = os.getenv(cfg["api_key_env"])
            if not api_key:
                raise RuntimeError(f"未找到环境变量 {cfg['api_key_env']}，请设置 OpenAI API key。")
            self.client = OpenAI(api_key=api_key,
                                 base_url="https://www.dmxapi.cn/v1/")
            self.model = self.agent_type
        elif self.agent_type == "qwen2.5-vl":
            # 全局只加载一次Qwen模型和processor
            if GLOBAL_QWEN_MODEL is None or GLOBAL_QWEN_PROCESSOR is None:
                log("Loading Qwen2.5-VL model and processor globally...")
                GLOBAL_QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    cfg["qwen_local_path"], 
                    dtype=torch.bfloat16,
                    device_map=cfg["device"],
                )
                GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(
                    cfg["qwen_local_path"], min_pixels=256*28*28, max_pixels=1280*28*28)
            self.model = GLOBAL_QWEN_MODEL
            self.processor = GLOBAL_QWEN_PROCESSOR
        elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY (or QWEN_API_KEY) environment variable is not set.\n"
                                   "Set it before running this script, e.g. export OPENAI_API_KEY=sk-...")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=300,  # 设置合理的超时时间
            )
            self.model = self.agent_type
        elif self.agent_type == "qwen3-vl-thinking":
            # 全局只加载一次Qwen模型和processor
            if GLOBAL_QWEN_MODEL is None or GLOBAL_QWEN_PROCESSOR is None:
                log("Loading Qwen2.5-VL model and processor globally...")
                GLOBAL_QWEN_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
                        cfg["qwen_local_path"],
                        dtype=torch.bfloat16,
                        device_map=cfg["device"],
                    )
                GLOBAL_QWEN_PROCESSOR = AutoProcessor.from_pretrained(cfg["qwen_local_path"],
                        dtype=torch.bfloat16,
                        device_map=cfg["device"],
                    )
            self.model = GLOBAL_QWEN_MODEL
            self.processor = GLOBAL_QWEN_PROCESSOR
        else:
            raise ValueError(f"Unsupported agent_type: {self.agent_type}")

    def get_mbti_prefix(self):
        if self.mbti:
            return f"[MBTI: {self.mbti}] You are an expert copyright judge with MBTI personality type: {self.mbti}. Please answer in the style of a {self.mbti}.\n"
        return ""

    # ---- 通用聊天接口 ----
    def _chat(self, prompt: str) -> str:
        log(f"Calling _chat with prompt: {prompt[:60]}...")
        mbti_prefix = self.get_mbti_prefix()
        prompt_with_mbti = mbti_prefix + prompt
        if "gpt-4o" in self.agent_type:
            messages = []
            messages.append({"role": "user", "content": prompt_with_mbti})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        elif self.agent_type == "qwen2.5-vl" or self.agent_type == "qwen3-vl-thinking":
            messages = []
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_with_mbti},
                ],
            })
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device if torch.cuda.is_available() else "cpu")
            max_new_tokens = 512
            if self.agent_type == "qwen3-vl-thinking":
                max_new_tokens = 2048
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_with_mbti},
                    ],
                }
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 81920
                }
            )
            return completion.choices[0].message.content.strip()

    def _debate_chat(self, image_x, image_xcr, filter_zc_zccr, prompt, human_refs):
        log(f"Calling _debate_chat with image_x={image_x}, image_xcr={image_xcr}, prompt={prompt[:40]}...")
        # human_refs: List[Tuple[Any, Any, float]]
        in_context_examples = human_refs if human_refs and len(human_refs) > 0 else []
        messages = []
        mbti_prefix = self.get_mbti_prefix()
        # Add in-context examples as user messages
        for gen_img, real_img, score in in_context_examples:
            if "gpt-4o" in self.agent_type:
                with open(gen_img, "rb") as f:
                    img_base64_x = base64.b64encode(f.read()).decode("utf-8")
                with open(real_img, "rb") as f:
                    img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                    {"type": "text", "text": f"Similarity Score: {score}"}
                ]
                messages.append({"role": "user", "content": content})
            elif self.agent_type in ("qwen2.5-vl", "qwen3-vl-thinking"):
                img1 = Image.open(gen_img).convert("RGB")
                img2 = Image.open(real_img).convert("RGB")
                content = [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": f"Similarity Score: {score}"}
                ]
                messages.append({"role": "user", "content": content})
            elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
                with open(gen_img, "rb") as f:
                    img_base64_x = base64.b64encode(f.read()).decode("utf-8")
                with open(real_img, "rb") as f:
                    img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                    {"type": "text", "text": f"Similarity Score: {score}"}
                ]
                messages.append({"role": "user", "content": content})
        # Add the target image pair as the last message
        if "gpt-4o" in self.agent_type:
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                {"type": "text", "text": mbti_prefix + f"{filter_zc_zccr}\n{prompt}"}
            ]
            messages.append({"role": "user", "content": content})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        elif self.agent_type in ("qwen2.5-vl", "qwen3-vl-thinking"):
            img1 = Image.open(image_x).convert("RGB")
            img2 = Image.open(image_xcr).convert("RGB")
            content = [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": mbti_prefix + f"{filter_zc_zccr}\n{prompt}"}
            ]
            messages.append({"role": "user", "content": content})
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device if torch.cuda.is_available() else "cpu")
            max_new_tokens = 512
            if self.agent_type == "qwen3-vl-thinking":
                max_new_tokens = 2048
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                {"type": "text", "text": mbti_prefix + f"{filter_zc_zccr}\n{prompt}"}
            ]
            messages.append({"role": "user", "content": content})
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 81920
                }
            )
            return completion.choices[0].message.content.strip()
        else:
            raise NotImplementedError

    # 专用聊天接口：摘要
    def _abstraction_chat(self, image_x, image_xcr, prompt):
        log(f"Calling _abstraction_chat with image_x={image_x}, image_xcr={image_xcr}, prompt={prompt[:40]}...")
        if "gpt-4o" in self.agent_type:
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        elif self.agent_type in ("qwen2.5-vl", "qwen3-vl-thinking"):
            # 一次传入两张图像，prompt不变
            img1 = Image.open(image_x).convert("RGB")
            img2 = Image.open(image_xcr).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": prompt},
                ]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device if torch.cuda.is_available() else "cpu")
            max_new_tokens = 512
            if self.agent_type == "qwen3-vl-thinking":
                max_new_tokens = 2048
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                stream=False,
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 81920
                }
            )
            return completion.choices[0].message.content.strip()
        else:
            raise NotImplementedError

    # 专用聊天接口：带两张图像的通用接口（律师等可调用）
    def _image_chat(self, image_x, image_xcr, prompt):
        """Generic image+text chat utility for agents that need to see both images and a prompt.

        Returns: string response from the model.
        """
        log(f"Calling _image_chat with image_x={image_x}, image_xcr={image_xcr}, prompt={prompt[:40]}...")
        if "gpt-4o" in self.agent_type:
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                {"type": "text", "text": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        elif self.agent_type in ("qwen2.5-vl", "qwen3-vl-thinking"):
            img1 = Image.open(image_x).convert("RGB")
            img2 = Image.open(image_xcr).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": prompt},
                ]}
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device if torch.cuda.is_available() else "cpu")
            max_new_tokens = 512
            if self.agent_type == "qwen3-vl-thinking":
                max_new_tokens = 2048
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                {"type": "text", "text": prompt},
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                stream=False,
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 81920
                }
            )
            return completion.choices[0].message.content.strip()
        else:
            raise NotImplementedError

    # 专用聊天接口：过滤
    def _filtering_chat(self, abstract_z_zcr, prompt):
        log(f"Calling _filtering_chat with abstract_z_zcr={abstract_z_zcr}, prompt={prompt[:40]}...")
        """
        输入：
            abstract_z_zcr，其中包含z: 可能侵权图像的摘要和zcr: 版权图像的摘要
            prompt: 过滤提示词
        输出：
            "Image1 Unique Elements:zc, Image2 Unique Elements:zccr"
            其中，zc: 过滤后可能侵权图像的摘要，zccr: 过滤后版权图像的摘要
        """
        input_text = f"Image decompositions: {abstract_z_zcr}\n{prompt}"
        if "gpt-4o" in self.agent_type:
            content = [
                {"type": "text", "text": input_text}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        elif self.agent_type in ("qwen2.5-vl", "qwen3-vl-thinking"):
            # 本地Qwen2.5-VL文本推理
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device if torch.cuda.is_available() else "cpu")
            max_new_tokens = 512
            if self.agent_type == "qwen3-vl-thinking":
                max_new_tokens = 2048
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
            # 本地Qwen2.5-VL文本推理
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 81920
                }
            )
            return completion.choices[0].message.content.strip()
        else:
            raise NotImplementedError

    # 专用聊天接口：辩论
    def _debate_chat(self, image_x, image_xcr, filter_zc_zccr, prompt, human_refs):
        log(f"Calling _debate_chat with image_x={image_x}, image_xcr={image_xcr}, prompt={prompt[:40]}...")
        # human_refs: List[Tuple[Any, Any, float]]
        in_context_examples = human_refs if human_refs and len(human_refs) > 0 else []
        messages = []
        mbti_prefix = self.get_mbti_prefix()
        # Add in-context examples as user messages
        for gen_img, real_img, score in in_context_examples:
            if "gpt-4o" in self.agent_type:
                with open(gen_img, "rb") as f:
                    img_base64_x = base64.b64encode(f.read()).decode("utf-8")
                with open(real_img, "rb") as f:
                    img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                    {"type": "text", "text": f"Similarity Score: {score}"}
                ]
                messages.append({"role": "user", "content": content})
            elif self.agent_type in ("qwen2.5-vl", "qwen3-vl-thinking"):
                img1 = Image.open(gen_img).convert("RGB")
                img2 = Image.open(real_img).convert("RGB")
                content = [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": f"Similarity Score: {score}"}
                ]
                messages.append({"role": "user", "content": content})
            elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
                with open(gen_img, "rb") as f:
                    img_base64_x = base64.b64encode(f.read()).decode("utf-8")
                with open(real_img, "rb") as f:
                    img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                    {"type": "text", "text": f"Similarity Score: {score}"}
                ]
                messages.append({"role": "user", "content": content})
            else:
                raise NotImplementedError
        # Add the target image pair as the last message
        if "gpt-4o" in self.agent_type:
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                {"type": "text", "text": mbti_prefix + f"{filter_zc_zccr}\n{prompt}"}
            ]
            messages.append({"role": "user", "content": content})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        elif self.agent_type == "qwen2.5-vl" or self.agent_type == "qwen3-vl-thinking":
            img1 = Image.open(image_x).convert("RGB")
            img2 = Image.open(image_xcr).convert("RGB")
            content = [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": mbti_prefix + f"{filter_zc_zccr}\n{prompt}"}
            ]
            messages.append({"role": "user", "content": content})
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device if torch.cuda.is_available() else "cpu")
            max_new_tokens = 512
            if self.agent_type == "qwen3-vl-thinking":
                max_new_tokens = 2048
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        elif self.agent_type == "qwen3-vl-plus" or self.agent_type == "qwen3-vl-plus-2025-09-23":
            with open(image_x, "rb") as f:
                img_base64_x = base64.b64encode(f.read()).decode("utf-8")
            with open(image_xcr, "rb") as f:
                img_base64_xcr = base64.b64encode(f.read()).decode("utf-8")
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_x}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64_xcr}"}},
                {"type": "text", "text": mbti_prefix + f"{filter_zc_zccr}\n{prompt}"}
            ]
            messages.append({"role": "user", "content": content})
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 81920
                }
            )
            return completion.choices[0].message.content.strip()
        else:
            raise NotImplementedError

    # ---- 三个核心接口 ----
    def abstract(self, image_x: Any, image_xcr: Any, prompt: str) -> str:
        log(f"abstract() called with image_x={image_x}, image_xcr={image_xcr}")
        abstract_z_zcr = self._abstraction_chat(image_x, image_xcr, prompt)
        return abstract_z_zcr # text的格式为‘Image1: XXX, Image2: XXX’

    def filter(self, abstract_z_zcr: str, prompt: str) -> str:
        log(f"filter() called with abstract_z_zcr={abstract_z_zcr}")
        filter_zc_zccr = self._filtering_chat(abstract_z_zcr, prompt)
        return filter_zc_zccr # text的格式为‘Image1 Unique Elements: XXX, Image2 Unique Elements: XXX’

    def compare(self, image_x: Any, image_xcr: Any, filter_zc_zccr: str, prompt: str, human_refs) -> Judgment:
        log(f"compare() called with image_x={image_x}, image_xcr={image_xcr}, prompt={prompt[:40]}...")
        out = self._debate_chat(image_x, image_xcr, filter_zc_zccr, prompt, human_refs)
        try:
            parts = out.split("Reason:")
            head, reason = parts[0], parts[1]
            score_str, conf_str = head.split(",")[:2]
            score = float(score_str.split(":")[1].strip())
            conf = float(conf_str.split(":")[1].strip())
        except Exception as e:
            # Log the parsing failure with contextual information, then re-raise
            try:
                log(f"Failed to parse judge comparison output for score/conf/reason: {e}. Raw output (truncated): {str(out)[:1000]}")
            except Exception:
                # If logging the raw output fails for any reason, still log the exception
                log(f"Failed to parse judge comparison output for score/conf/reason: {e}. Also failed to stringify raw output.")
            raise
        return Judgment(score=score, confidence=conf, rationale=reason.strip())

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

class Court:
    """Court orchestrator that maintains per-case global_history and performs atomic saves.

    Usage: create Court(cfg, judge, expert, plaintiff, defendant) and call run_trial for each case.
    After each case completes, per-case artifacts are saved under outputs/<run_ts>/<case_id>/ and a
    single-line JSON is appended to final_results.jsonl in the run folder.
    """

    def __init__(self, cfg: Dict[str, Any], judge: 'JudgeAgent', expert: 'ExpertAgent', plaintiff: 'LawyerAgent', defendant: 'LawyerAgent', run_dir: Optional[str] = None):
        self.cfg = cfg
        self.judge = judge
        self.expert = expert
        self.plaintiff = plaintiff
        self.defendant = defendant
        # run_dir may be provided by caller (main); if not, compute here to preserve previous behavior
        if run_dir:
            self.run_dir = run_dir
        else:
            # create top-level run dir
            self.ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            # include dataset name in the run prefix for easier identification
            dataset_dir = cfg.get('dataset_dir', '.')
            dataset_name = cfg.get('dataset_name') or os.path.basename(os.path.normpath(dataset_dir)) or 'dataset'
            # sanitize dataset_name (replace os.sep and spaces)
            dataset_name = dataset_name.replace(os.sep, '_').replace(' ', '_')
            prefix = f"dataset-{dataset_name}_courtroom-agent_type-{cfg.get('agent_type','NA')}_max_rounds-{cfg.get('max_rounds',0)}_gamma-{cfg.get('gamma',0.5)}_judge_confidence-{cfg.get('judge_confidence_threshold', 0.75)}"
            self.run_dir = os.path.join('./outputs', f"{self.ts}_{prefix}")
            os.makedirs(self.run_dir, exist_ok=True)
        # ensure run dir exists
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.run_dir, 'final_results.jsonl')

    def run_trial(self, image_x: str, image_xcr: str, human_refs: Optional[List[tuple]] = None, expert_outputs: Optional[dict] = None) -> FinalDecision:
        """Run a trial and persist per-case artifacts atomically.

        Returns the FinalDecision object.
        """
        human_refs = human_refs or []
        # prepare case id
        case_id = os.path.splitext(os.path.basename(image_x))[0]
        # ensure expert outputs computed
        if expert_outputs is None:
            expert_outputs = self.expert.analyze(image_x, image_xcr, human_refs, case_id=case_id)
        expert_outputs.setdefault('case_id', case_id)

        # per-case global history list
        global_history: List[Dict[str, Any]] = []
        # attach expert output as first entry
        try:
            global_history.append({'speaker': 'expert', 'role': 'expert', 'content': {
                'abstraction': expert_outputs.get('abstraction'),
                'filtration': expert_outputs.get('filtration'),
                'judgment': {
                    'score': getattr(expert_outputs.get('judgment'), 'score', None),
                    'confidence': getattr(expert_outputs.get('judgment'), 'confidence', None),
                    'rationale': getattr(expert_outputs.get('judgment'), 'rationale', None),
                }
            }})
        except Exception as e:
            log(f"Failed to attach expert output to global_history for case {case_id}: {e}")
            raise

        # call judge.run_trial which now accepts global_history and will append intermediate events
        final = self.judge.run_trial(image_x, image_xcr, self.expert, self.plaintiff, self.defendant, human_refs=human_refs, expert_outputs=expert_outputs, global_history=global_history)

        # After trial ends, persist per-case artifacts
        case_dir = os.path.join(self.run_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        # save global_history
        try:
            with open(os.path.join(case_dir, 'global_history.json'), 'w', encoding='utf-8') as fh:
                json.dump(global_history, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"Failed to save global_history for {case_id}: {e}")

        # save final decision
        try:
            final_dict = {
                'score_final': final.score_final,
                'confidence_final': final.confidence_final,
                'rationale_final': final.rationale_final,
                'is_infringement': final.is_infringement,
                'case_id': case_id,
                'timestamp': datetime.now().isoformat(),
            }
            with open(os.path.join(case_dir, 'final_decision.json'), 'w', encoding='utf-8') as fh:
                json.dump(final_dict, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"Failed to save final decision for {case_id}: {e}")

        # append to run-level jsonl
        try:
            with open(self.jsonl_path, 'a', encoding='utf-8') as fj:
                fj.write(json.dumps({'case_id': case_id, 'score_final': final.score_final, 'confidence_final': final.confidence_final, 'rationale_final': final.rationale_final, 'is_infringement': final.is_infringement, 'timestamp': datetime.now().isoformat()} , ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"Failed to append final result for {case_id} to jsonl: {e}")

        return final


class LawyerAgent:
    """Lawyer agent (plaintiff or defendant) that produces statements using LVLM."""

    def __init__(self, cfg: dict, role: str = 'plaintiff'):
        assert role in ('plaintiff', 'defendant')
        self.role = role
        self.agent = LLMAdapterAgent(cfg)
        # Knowledge store for lawyer: K_l = {E_l, C}
        # E_l: list of brief experiences / reflections (strings)
        # C: list of case_ids the lawyer has seen
        self.K_l = {'E_l': [], 'C': []}
        self.cfg = cfg

    def update_knowledge_base(self, expert_outputs: dict, case_id: str):
        """Update the lawyer's knowledge base with experience and case summaries."""
        # Generate experience summary
        experience_summary = self.agent._chat(
            f"You are a lawyer. Summarize the experience from this case:\n"
            f"Expert Filtration: {expert_outputs['filtration']}\n"
            f"Role: {self.role}\n"
            f"Provide a concise summary."
        )
        self.K_l['E_l'].append({'case_id': case_id, 'summary': experience_summary})

        # Generate case summary
        case_summary = self.agent._chat(
            f"You are a lawyer. Summarize the case:\n"
            f"Expert Filtration: {expert_outputs['filtration']}\n"
            f"Role: {self.role}\n"
            f"Provide a concise case summary."
        )
        if case_id not in self.K_l['C']:
            self.K_l['C'].append({'case_id': case_id, 'summary': case_summary})

    def opening_statement(self, expert_outputs: dict, gen_image: str, real_image: str) -> str:
        """Produce an opening statement. Includes both images so the lawyer can reference visual evidence."""
        template = (
            "You are an experienced copyright lawyer representing the {role}.\n"
            "Based on the Expert's analysis below and the two images provided, produce a concise opening statement (2-5 sentences).\n"
            "Focus on legal relevance and persuasive points and, if useful, reference visible elements from the images.\n\n"
            "Expert Abstraction:\n{abstraction}\n\n"
            "Expert Filtration:\n{filtration}\n\n"
            "Expert Judgment:\nScore={score:.3f}, Confidence={conf:.3f}\n"
        )
        prompt = template.format(
            role=self.role,
            abstraction=expert_outputs.get('abstraction', ''),
            filtration=expert_outputs.get('filtration', ''),
            score=getattr(expert_outputs.get('judgment'), 'score', 0.0),
            conf=getattr(expert_outputs.get('judgment'), 'confidence', 0.0),
        )
        # steer plaintiff to emphasize infringement, defendant to emphasize non-infringement
        if self.role == 'plaintiff':
            prompt += "\nYour goal: emphasize ways the accused image copies protectable elements and supports an infringement finding."
        else:
            prompt += "\nYour goal: emphasize differences and lawful/independent creation to rebut infringement claims."

        log(f"{self.role.capitalize()} lawyer: generating opening statement with images")
        out = self.agent._image_chat(gen_image, real_image, prompt)
        return out.strip()

    def rebuttal(self, expert_outputs: dict, opponent_statement: str, gen_image: str, real_image: str, judge_feedback: Optional[str] = None) -> str:
        """Produce a rebuttal that can reference both images."""
        template = (
            "You are a copyright lawyer (role={role}).\n"
            "Opponent said:\n{opponent}\n\n"
            "Expert Abstraction:\n{abstraction}\n\n"
            "Expert Filtration:\n{filtration}\n\n"
            "Expert Judgment:\nScore={score:.3f}, Confidence={conf:.3f}\n"
            "Provide a rebuttal (2-4 sentences), focusing on legal counter-arguments and image evidence if relevant."
        )
        prompt = template.format(
            role=self.role,
            opponent=opponent_statement,
            abstraction=expert_outputs.get('abstraction', ''),
            filtration=expert_outputs.get('filtration', ''),
            score=getattr(expert_outputs.get('judgment'), 'score', 0.0),
            conf=getattr(expert_outputs.get('judgment'), 'confidence', 0.0),
        )
        if judge_feedback:
            prompt += f"\nJudge asked: {judge_feedback}\nRespond to the judge's concern."

        log(f"{self.role.capitalize()} lawyer: generating rebuttal with images")
        out = self.agent._image_chat(gen_image, real_image, prompt)
        return out.strip()

    def answer_question(self, question: str, expert_outputs: dict, gen_image: str, real_image: str, opponent_statement: Optional[str] = None) -> str:
        """Answer a judge's concise question. Includes both images so the lawyer can reference visual evidence. Return a short (1-3 sentence) reply."""
        template = (
            "You are a copyright lawyer representing the {role}.\n"
            "The judge asked the following question: \n{question}\n\n"
            "Based on the Expert's filtration and your client's position, answer concisely (1-3 sentences)."\
            " If relevant, cite which unique elements or differences support your answer.\n\n"
            "Expert Abstraction:\n{abstraction}\n\n"
            "Expert Filtration:\n{filtration}\n\n"
            "Expert Judgment:\nScore={score:.3f}, Confidence={conf:.3f}\n"
            "Opponent Statement (if available):\n{opponent}\n"
        )
        prompt = template.format(
            role=self.role,
            question=question,
            abstraction=expert_outputs.get('abstraction', ''),
            filtration=expert_outputs.get('filtration', ''),
            score=getattr(expert_outputs.get('judgment'), 'score', 0.0),
            conf=getattr(expert_outputs.get('judgment'), 'confidence', 0.0),
            opponent=(opponent_statement or "None"),
        )
        log(f"{self.role.capitalize()} lawyer: answering judge question with images")
        out = self.agent._image_chat(gen_image, real_image, prompt)
        return out.strip()

    def reflect_and_summary(self, case_id: Optional[str] = None, global_history: Optional[List[dict]] = None, final_decision: Optional[FinalDecision] = None) -> str:
        """Lawyer reflects on the case using only the full-case `global_history` and the `final_decision`.

        Stores a concise experience summary into K_l['E_l'] and returns it.
        """
        if not self.cfg.get('enable_reflection_summary', True):
            return ""
        prompt = (
            f"You are a senior copyright lawyer representing the {self.role}.\n"
            "Reflect briefly (2-3 sentences) on which arguments across the full-case dialog were most persuasive, any weak points, and one recommendation for future similar cases.\n\n"
        )
        try:
            if global_history and isinstance(global_history, list):
                tail = global_history[-12:]
                prompt += "Recent dialog/events (last few):\n" + json.dumps(tail, ensure_ascii=False, indent=0) + "\n\n"
        except Exception as e:
            log(f"Failed to include global_history tail in lawyer reflect prompt for case {case_id}: {e}")
            raise
        try:
            if final_decision:
                prompt += f"Final Decision: is_infringement={getattr(final_decision, 'is_infringement', '')}, score={getattr(final_decision, 'score_final', '')}, confidence={getattr(final_decision, 'confidence_final', '')}, rationale={getattr(final_decision, 'rationale_final', '')}\n"
        except Exception as e:
            log(f"Failed to include final_decision in lawyer reflect prompt for case {case_id}: {e}")
            raise
        try:
            summary = self.agent._chat(prompt).strip()
        except Exception as e:
            log(f"{self.role.capitalize()} lawyer reflect_and_summary failed: {e}")
            summary = ""

        # store
        try:
            self.K_l['E_l'].append({'case_id': case_id or f'case_{len(self.K_l["E_l"]) + 1}', 'summary': summary})
            if case_id and case_id not in self.K_l['C']:
                self.K_l['C'].append(case_id)
        except Exception as e:
            log(f"Failed to store lawyer reflection in K_l for {self.role} case {case_id}: {e}")
            raise
        return summary


class JudgeAgent:
    """Judge agent that listens to expert and lawyers and makes decisions."""

    def __init__(self, cfg: dict):
        self.agent = LLMAdapterAgent(cfg)
        self.confidence_threshold = cfg.get('judge_confidence_threshold', 0.75)
        self.max_rounds = cfg.get('max_rounds', 3)
        self.gamma = cfg.get('gamma', 0.5)
        # single flag controlling reflection+summary behavior
        self.enable_reflection_summary = cfg.get('enable_reflection_summary', True)
        # Judge knowledge store K_j = {E_j, C}
        self.K_j = {'E_j': [], 'C': []}
        self.cfg = cfg

    def update_knowledge_base(self, expert_outputs: dict, case_id: str, final_decision: FinalDecision):
        """Update the judge's knowledge base with reflection-based experience and case summaries."""
        # Generate reflection summary
        reflection_summary = self.agent._chat(
            f"You are the Judge. Reflect on this case:\n"
            f"Expert Filtration: {expert_outputs['filtration']}\n"
            f"Plaintiff Statement: {expert_outputs.get('plaintiff', '')}\n"
            f"Defendant Statement: {expert_outputs.get('defendant', '')}\n"
            f"Final Decision: {final_decision.rationale_final}\n"
            f"Provide a concise reflection summary."
        )
        self.K_j['E_j'].append({'case_id': case_id, 'summary': reflection_summary})

        # Generate case summary
        case_summary = self.agent._chat(
            f"You are the Judge. Summarize the case:\n"
            f"Expert Filtration: {expert_outputs['filtration']}\n"
            f"Plaintiff Statement: {expert_outputs.get('plaintiff', '')}\n"
            f"Defendant Statement: {expert_outputs.get('defendant', '')}\n"
            f"Final Decision: {final_decision.rationale_final}\n"
            f"Provide a concise case summary."
        )
        if case_id not in self.K_j['C']:
            self.K_j['C'].append({'case_id': case_id, 'summary': case_summary})

    def evaluate(self, image_x: str, image_xcr: str, expert_outputs: dict, plaintiff_stmt: str, defendant_stmt: str) -> dict:
        """Ask the judge-agent to evaluate arguments and either (A) issue a final verdict or (B) ask a concise question to one party.

        The agent MUST respond in one of the two strict formats (examples included):

        1) Final verdict format (when ready to decide):
           Action: Verdict\n
           Verdict: [Infringement|Not Infringement]\n
           Confidence: [0-1]\n
           Reason: [concise explanation]\n
        2) Question format (when judge needs more information):
           Action: Question\n
           Target: [Plaintiff|Defendant|Both]\n
           Question: [concise question to the target(s)]\n
        The judge should prefer a verdict when sufficiently confident; otherwise it may ask a single clarifying question.
        """
        prompt = (
            "You are the judge in a copyright infringement case.\n"
            "Given the accused image, the copyrighted image, the expert analysis, the plaintiff's and defendant's statements, you must either: \n"
            "  - Issue a final verdict (Infringement or Not Infringement) with a confidence score, OR\n"
            "  - Ask a single concise clarifying question directed to the Plaintiff or the Defendant (or Both) that will help you reach a decision.\n\n"
            "OUTPUT STRICT FORMAT (one of the two):\n"
            "1) Action: Verdict\n"
            "   Verdict: [Infringement|Not Infringement]\n"
            "   Confidence: [0-1]\n"
            "   Reason: [text]\n\n"
            "2) Action: Question\n"
            "   Target: [Plaintiff|Defendant|Both]\n"
            "   Question: [text]\n\n"
            "Expert Filtration:\n{filtration}\n\n"
            "Plaintiff Statement:\n{plaintiff}\n\n"
            "Defendant Statement:\n{defendant}\n\n"
            "(Image paths: Accused={image_x}, Copyrighted={image_xcr})\n"
        ).format(filtration=expert_outputs['filtration'], plaintiff=plaintiff_stmt, defendant=defendant_stmt, image_x=image_x, image_xcr=image_xcr)

        log("Judge: evaluating the round (may verdict or ask question)")
        # Use image-aware chat so the judge can reference visual evidence when evaluating
        try:
            out = self.agent._image_chat(image_x, image_xcr, prompt)
        except Exception:
            # fallback to text-only chat if image chat fails
            out = self.agent._chat(prompt)
        log(f"Judge raw output: {out[:800]}")

        # parsing
        result = {'action': 'undecided', 'raw': out}
        try:
            lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
            # find Action line
            action_line = next((l for l in lines if l.lower().startswith('action:')), None)
            if action_line:
                action = action_line.split(':', 1)[1].strip().lower()
                if action == 'verdict':
                    # extract verdict, confidence, reason
                    vline = next((l for l in lines if l.lower().startswith('verdict:')), '')
                    cline = next((l for l in lines if l.lower().startswith('confidence:')), '')
                    rline = next((l for l in lines if l.lower().startswith('reason:')), '')
                    verdict = vline.split(':', 1)[1].strip() if vline else 'Undecided'
                    confidence = float(cline.split(':', 1)[1].strip()) if cline else 0.0
                    reason = rline.split(':', 1)[1].strip() if rline else ''
                    result.update({'action': 'verdict', 'verdict': verdict, 'confidence': confidence, 'reason': reason})
                    return result
                elif action == 'question':
                    tline = next((l for l in lines if l.lower().startswith('target:')), '')
                    qline = next((l for l in lines if l.lower().startswith('question:')), '')
                    target = tline.split(':', 1)[1].strip() if tline else 'Both'
                    question = qline.split(':', 1)[1].strip() if qline else ''
                    result.update({'action': 'question', 'target': target, 'question': question})
                    return result
        except Exception as e:
            log(f"Judge parsing error: {e}")

        # fallback: try old-style verdict parse for compatibility
        try:
            parts = out.split('Reason:')
            head = parts[0]
            reason = parts[1].strip() if len(parts) > 1 else ''
            head_parts = head.split(',')
            vpart = head_parts[0]
            cpart = head_parts[1] if len(head_parts) > 1 else ''
            verdict = vpart.split(':', 1)[1].strip()
            confidence = float(cpart.split(':', 1)[1].strip()) if cpart else 0.0
            return {'action': 'verdict', 'verdict': verdict, 'confidence': confidence, 'reason': reason}
        except Exception:
            return result

    def run_trial(self, image_x: str, image_xcr: str, expert: ExpertAgent, plaintiff: LawyerAgent, defendant: LawyerAgent, human_refs: Optional[List[tuple]] = None, expert_outputs: Optional[dict] = None, global_history: Optional[List[dict]] = None):
        """
        Run a single trial.

        Parameters:
        - image_x, image_xcr: paths to the two images (accused, copyrighted)
        - expert, plaintiff, defendant: agent instances
        - human_refs: optional in-context examples
        - expert_outputs: OPTIONAL precomputed expert outputs (dict with keys 'abstraction','filtration','judgment').

        If `expert_outputs` is provided, it will be used instead of calling expert.analyze again. This
        is useful for batch runners that want to save or reuse expert outputs without double computation.
        """
        human_refs = human_refs or []
        # ensure we have a global_history to append to
        if global_history is None:
            global_history = []
        if expert_outputs is None:
            expert_outputs = expert.analyze(image_x, image_xcr, human_refs)

        # attach case id where possible
        case_id = os.path.splitext(os.path.basename(image_x))[0]
        try:
            expert_outputs.setdefault('case_id', case_id)
        except Exception as e:
            log(f"Failed to set default case_id in expert_outputs for case {case_id}: {e}")
            raise

        # Opening statements (pass image paths so lawyers can reference them)
        plaintiff_stmt = plaintiff.opening_statement(expert_outputs, image_x, image_xcr)
        defendant_stmt = defendant.opening_statement(expert_outputs, image_x, image_xcr)
        # record into global history
        try:
            global_history.append({'speaker': 'plaintiff', 'role': 'plaintiff', 'content': plaintiff_stmt})
            global_history.append({'speaker': 'defendant', 'role': 'defendant', 'content': defendant_stmt})
        except Exception as e:
            log(f"Failed to append opening statements to global_history for case {case_id}: {e}")
            raise
        
        # Judge evaluates
        for round_idx in range(1, self.max_rounds + 1):
            log(f"--- Round {round_idx} ---")
            result = self.evaluate(image_x, image_xcr, expert_outputs, plaintiff_stmt, defendant_stmt)
            log(f"Judge result: {result}")
            # append judge output to history
            try:
                global_history.append({'speaker': 'judge', 'role': 'judge', 'content': result})
            except Exception as e:
                log(f"Failed to append judge output to global_history for case {case_id}, round {round_idx}: {e}")
                raise

            if result.get('action') == 'verdict':
                v = result['verdict']
                conf = result.get('confidence', 0.0)
                reason = result.get('reason', '')
                if v in ('Infringement', 'Not Infringement') and conf >= self.confidence_threshold:
                    is_infringement = (v == 'Infringement')
                    final = FinalDecision(score_final=1.0 if is_infringement else 0.0,
                                         confidence_final=conf,
                                         rationale_final=reason,
                                         is_infringement=is_infringement)
                    # Let agents reflect and store experience before returning
                    try:
                        # provide global_history as part of expert_outputs for backward compatibility
                        plaintiff.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                        defendant.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                        expert.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                        self.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                    except Exception as e:
                        log(f"Agent reflection failed during finalization for case {case_id}: {e}")
                        raise
                    # append final decision
                    try:
                        global_history.append({'speaker': 'judge', 'role': 'judge', 'content': {'final_decision': {'score_final': final.score_final, 'confidence_final': final.confidence_final, 'rationale_final': final.rationale_final, 'is_infringement': final.is_infringement}}})
                    except Exception as e:
                        log(f"Failed to append final decision to global_history for case {case_id}: {e}")
                        raise
                    return final
                # low confidence - treat as undecided and allow rebuttal
                judge_feedback = f"Round {round_idx} feedback: {reason[:300]}"
                plaintiff_stmt = plaintiff.rebuttal(expert_outputs, defendant_stmt, image_x, image_xcr, judge_feedback)
                # record rebuttal
                try:
                    global_history.append({'speaker': 'plaintiff', 'role': 'plaintiff', 'content': plaintiff_stmt, 'meta': {'round': round_idx, 'type': 'rebuttal'}})
                except Exception as e:
                    log(f"Failed to append plaintiff rebuttal to global_history for case {case_id}, round {round_idx}: {e}")
                    raise
                defendant_stmt = defendant.rebuttal(expert_outputs, plaintiff_stmt, image_x, image_xcr, judge_feedback)
                try:
                    global_history.append({'speaker': 'defendant', 'role': 'defendant', 'content': defendant_stmt, 'meta': {'round': round_idx, 'type': 'rebuttal'}})
                except Exception as e:
                    log(f"Failed to append defendant rebuttal to global_history for case {case_id}, round {round_idx}: {e}")
                    raise
                # If this was the final allowed round, let the judge re-evaluate once more
                if round_idx == self.max_rounds:
                    log("Final round rebuttal submitted — performing one last judge evaluation")
                    final_eval = self.evaluate(image_x, image_xcr, expert_outputs, plaintiff_stmt, defendant_stmt)
                    log(f"Judge re-evaluation (final round) result: {final_eval}")
                    if final_eval.get('action') == 'verdict':
                        v2 = final_eval.get('verdict')
                        c2 = final_eval.get('confidence', 0.0)
                        r2 = final_eval.get('reason', '')
                        if v2 in ('Infringement', 'Not Infringement') and c2 >= self.confidence_threshold:
                            is_infringement = (v2 == 'Infringement')
                            final = FinalDecision(score_final=1.0 if is_infringement else 0.0,
                                                 confidence_final=c2,
                                                 rationale_final=r2,
                                                 is_infringement=is_infringement)
                            try:
                                plaintiff.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                defendant.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                expert.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                self.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                            except Exception:
                                raise Exception("Final round judge evaluation produced invalid verdict")
                            try:
                                global_history.append({'speaker': 'judge', 'role': 'judge', 'content': {'final_decision': {'score_final': final.score_final, 'confidence_final': final.confidence_final, 'rationale_final': final.rationale_final, 'is_infringement': final.is_infringement}}})
                            except Exception as e:
                                log(f"Failed to append final decision to global_history after final round for case {case_id}: {e}")
                                raise
                            return final

            elif result.get('action') == 'question':
                target = result.get('target', 'Both').lower()
                question = result.get('question', '')
                # direct question to the target(s) and collect answers
                plaintiff_answer = None
                defendant_answer = None
                if target in ('plaintiff', 'both'):
                    plaintiff_answer = plaintiff.answer_question(question, expert_outputs, image_x, image_xcr, opponent_statement=defendant_stmt)
                    # update plaintiff statement to the answer (short answer becomes the plaintiff's latest statement)
                    plaintiff_stmt = plaintiff_answer
                    try:
                        global_history.append({'speaker': 'plaintiff', 'role': 'plaintiff', 'content': plaintiff_answer, 'meta': {'round': round_idx, 'type': 'answer'}})
                    except Exception as e:
                        log(f"Failed to append plaintiff answer to global_history for case {case_id}, round {round_idx}: {e}")
                        raise
                if target in ('defendant', 'both'):
                    defendant_answer = defendant.answer_question(question, expert_outputs, image_x, image_xcr, opponent_statement=plaintiff_stmt)
                    defendant_stmt = defendant_answer
                    try:
                        global_history.append({'speaker': 'defendant', 'role': 'defendant', 'content': defendant_answer, 'meta': {'round': round_idx, 'type': 'answer'}})
                    except Exception as e:
                        log(f"Failed to append defendant answer to global_history for case {case_id}, round {round_idx}: {e}")
                        raise
                # After answers, judge will re-evaluate in next loop iteration (counts as one round)
                # If this was the final allowed round, perform one immediate re-evaluation so the judge can decide
                if round_idx == self.max_rounds:
                    log("Final round question answered — performing one last judge evaluation")
                    final_eval = self.evaluate(image_x, image_xcr, expert_outputs, plaintiff_stmt, defendant_stmt)
                    log(f"Judge re-evaluation (final round) result: {final_eval}")
                    if final_eval.get('action') == 'verdict':
                        v2 = final_eval.get('verdict')
                        c2 = final_eval.get('confidence', 0.0)
                        r2 = final_eval.get('reason', '')
                        if v2 in ('Infringement', 'Not Infringement') and c2 >= self.confidence_threshold:
                            is_infringement = (v2 == 'Infringement')
                            final = FinalDecision(score_final=1.0 if is_infringement else 0.0,
                                                 confidence_final=c2,
                                                 rationale_final=r2,
                                                 is_infringement=is_infringement)
                            try:
                                plaintiff.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                defendant.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                expert.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                self.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                            except Exception:
                                raise Exception("Final round judge evaluation produced invalid verdict")
                            try:
                                global_history.append({'speaker': 'judge', 'role': 'judge', 'content': {'final_decision': {'score_final': final.score_final, 'confidence_final': final.confidence_final, 'rationale_final': final.rationale_final, 'is_infringement': final.is_infringement}}})
                            except Exception as e:
                                log(f"Failed to append final decision to global_history after final round for case {case_id}, round {round_idx}: {e}")
                                raise
                            return final

            else:
                # fallback: if judge didn't produce an actionable output, let lawyers rebut once
                judge_feedback = f"Round {round_idx} feedback: {result.get('raw', '')[:300]}"
                plaintiff_stmt = plaintiff.rebuttal(expert_outputs, defendant_stmt, image_x, image_xcr, judge_feedback)
                defendant_stmt = defendant.rebuttal(expert_outputs, plaintiff_stmt, image_x, image_xcr, judge_feedback)
                # If this was the final round, perform one final evaluation after rebuttals
                if round_idx == self.max_rounds:
                    log("Final round fallback rebuttal submitted — performing one last judge evaluation")
                    final_eval = self.evaluate(image_x, image_xcr, expert_outputs, plaintiff_stmt, defendant_stmt)
                    log(f"Judge re-evaluation (final round) result: {final_eval}")
                    if final_eval.get('action') == 'verdict':
                        v2 = final_eval.get('verdict')
                        c2 = final_eval.get('confidence', 0.0)
                        r2 = final_eval.get('reason', '')
                        if v2 in ('Infringement', 'Not Infringement') and c2 >= self.confidence_threshold:
                            is_infringement = (v2 == 'Infringement')
                            final = FinalDecision(score_final=1.0 if is_infringement else 0.0,
                                                 confidence_final=c2,
                                                 rationale_final=r2,
                                                 is_infringement=is_infringement)
                            try:
                                plaintiff.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                defendant.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                expert.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                                self.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
                            except Exception:
                                raise Exception("Final round judge evaluation produced invalid verdict")
                            return final

        # After max rounds, fallback to expert judgment or aggregate
        log("Max rounds reached or judge undecided - using expert judgment as tie-breaker")
        ej = expert_outputs['judgment']
        is_infr = ej.score > self.gamma
        final = FinalDecision(score_final=ej.score,
                             confidence_final=ej.confidence,
                             rationale_final=f"Fallback to expert comparison: score={ej.score:.3f}, reason={ej.rationale}",
                             is_infringement=is_infr)
        # Ask agents to reflect and store this outcome
        try:
            plaintiff.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
            defendant.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
            expert.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
            self.reflect_and_summary(case_id=case_id, global_history=global_history, final_decision=final)
        except Exception:
            raise Exception("Final round judge evaluation produced invalid verdict")
        try:
            global_history.append({'speaker': 'judge', 'role': 'judge', 'content': {'final_decision': {'score_final': final.score_final, 'confidence_final': final.confidence_final, 'rationale_final': final.rationale_final, 'is_infringement': final.is_infringement}}})
        except Exception as e:
            log(f"Failed to append final decision to global_history for case {case_id} at end of run: {e}")
            raise
        return final

    def reflect_and_summary(self, case_id: Optional[str] = None, global_history: Optional[List[dict]] = None, final_decision: Optional[FinalDecision] = None) -> str:
        """Judge reflects on the case using only the full-case `global_history` and the `final_decision`, stores brief experience in K_j['E_j'], and returns a short summary."""
        if not self.enable_reflection_summary:
            log("Reflection is disabled in the configuration.")
            return ""
        prompt = (
            "You are the Judge. Reflect briefly (2-4 sentences) on what facts, patterns in the dialogue, and arguments mattered most in reaching (or failing to reach) a decision.\n"
            "Also mention one pattern the court should watch in future cases.\n\n"
        )
        try:
            if global_history and isinstance(global_history, list):
                tail = global_history[-24:]
                prompt += "Recent dialog/events (last few):\n" + json.dumps(tail, ensure_ascii=False, indent=0) + "\n\n"
        except Exception as e:
            log(f"Judge reflect_and_summary failed while preparing global_history tail for case {case_id}: {e}")
            raise
        try:
            if final_decision:
                prompt += f"Final Decision: is_infringement={getattr(final_decision, 'is_infringement', '')}, score={getattr(final_decision, 'score_final', '')}, confidence={getattr(final_decision, 'confidence_final', '')}, rationale={getattr(final_decision, 'rationale_final', '')}\n"
        except Exception as e:
            log(f"Judge reflect_and_summary failed while formatting final_decision for case {case_id}: {e}")
            raise
        try:
            summary = self.agent._chat(prompt).strip()
        except Exception as e:
            log(f"Judge reflect_and_summary failed: {e}")
            summary = ""
        if self.enable_reflection_summary:
            try:
                self.K_j['E_j'].append({'case_id': case_id or f'case_{len(self.K_j["E_j"]) + 1}', 'summary': summary})
                if case_id and case_id not in self.K_j['C']:
                    self.K_j['C'].append(case_id)
            except Exception as e:
                log(f"Judge reflect_and_summary failed to store summary in K_j for case {case_id}: {e}")
                raise
        return summary


def batch_run_trials(cfg: Dict[str, Any], judge: JudgeAgent, expert: ExpertAgent, plaintiff: LawyerAgent, defendant: LawyerAgent, human_refs: Optional[list] = None, run_dir: Optional[str] = None):
    """
    Batch-run courtroom simulations over the configured test set and save results.

    This function mirrors the evaluation flow in `judge.py` but uses the courtroom
    simulator's `JudgeAgent.run_trial`. For each test case it:
      1. Builds image paths from cfg["dataset_dir"] and the test set entry.
      2. Runs the Expert analysis once and passes the outputs into `run_trial` to
         avoid duplicated work.
      3. Collects per-case detailed results and a compact final_results mapping.
      4. Writes JSON outputs and a metrics text file into a timestamped output dir.

    Important fields saved per case:
      - image_name, gen_path, real_path
      - expert_outputs: abstraction, filtration, judgment (score/conf/reason)
      - final_decision: score_final, confidence_final, rationale_final, is_infringement

    Returns: path to output directory that contains saved artifacts.
    """
    human_refs = human_refs or []
    dataset_dir = cfg.get('dataset_dir', '.')
    test_set_path = cfg.get('test_set_path')
    if not test_set_path or not os.path.exists(test_set_path):
        raise FileNotFoundError(f"test_set_path not found in config or does not exist: {test_set_path}")

    # load test set (jsonl with entries containing at least image_name)
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = [json.loads(line) for line in f if line.strip()]

    # create Court orchestrator to get per-case atomic saves (allow run_dir override)
    court = Court(cfg, judge, expert, plaintiff, defendant, run_dir=run_dir)
    detailed_results = []
    final_results = {}

    for item in test_set:
        img_name = item.get('image_name')
        gen_img_path = os.path.join(dataset_dir, 'Test', 'gen', f'gen_{img_name}.jpg')
        real_img_path = os.path.join(dataset_dir, 'Test', 'real', f'real_{img_name}.jpg')
        log(f"Processing case: {img_name}")

        # Run expert analysis once and pass into run_trial to avoid duplicate work
        try:
            expert_out = expert.analyze(gen_img_path, real_img_path, human_refs)
        except Exception as e:
            log(f"Expert analysis failed for {img_name}: {e}")
            expert_out = {'abstraction': '', 'filtration': '', 'judgment': Judgment(score=0.0, confidence=0.0, rationale=str(e))}

        # Run the full courtroom flow using precomputed expert outputs via Court.run_trial (atomic per-case save)
        try:
            final = court.run_trial(gen_img_path, real_img_path, human_refs=human_refs, expert_outputs=expert_out)
        except Exception as e:
            log(f"run_trial failed for {img_name}: {e}")
            final = FinalDecision(score_final=expert_out['judgment'].score, confidence_final=expert_out['judgment'].confidence, rationale_final=f"Error during trial: {e}", is_infringement=expert_out['judgment'].score > cfg.get('gamma', 0.5))

        # store results
        detailed = {
            'image_name': img_name,
            'gen_path': gen_img_path,
            'real_path': real_img_path,
            'expert_outputs': {
                'abstraction': expert_out.get('abstraction', ''),
                'filtration': expert_out.get('filtration', ''),
                'judgment': {
                    'score': getattr(expert_out.get('judgment'), 'score', None),
                    'confidence': getattr(expert_out.get('judgment'), 'confidence', None),
                    'rationale': getattr(expert_out.get('judgment'), 'rationale', None),
                }
            },
            'final_decision': {
                'score_final': final.score_final,
                'confidence_final': final.confidence_final,
                'rationale_final': final.rationale_final,
                'is_infringement': final.is_infringement,
            }
        }
        detailed_results.append(detailed)
        final_results[img_name] = 1 if final.is_infringement else 0

    # determine out_dir: if run_dir passed from caller (main), use it; otherwise compute here
    if run_dir:
        out_dir = run_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # include dataset name in the run-level output folder
        dataset_dir = cfg.get('dataset_dir', '.')
        dataset_name = cfg.get('dataset_name') or os.path.basename(os.path.normpath(dataset_dir)) or 'dataset'
        dataset_name = dataset_name.replace(os.sep, '_').replace(' ', '_')
        dir_prefix = f"dataset-{dataset_name}_courtroom-agent_type-{cfg.get('agent_type','NA')}_max_rounds-{cfg.get('max_rounds',0)}_gamma-{cfg.get('gamma',0.5)}_judge_confidence-{cfg.get('judge_confidence_threshold', 0.75)}"
        out_dir = os.path.join('./outputs', f"{timestamp}_{dir_prefix}")
        os.makedirs(out_dir, exist_ok=True)

    # write artifacts
    with open(os.path.join(out_dir, 'detailed_results.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'final_results.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # compute simple metrics if labels provided
    metrics_txt = os.path.join(out_dir, 'metrics.txt')
    test_label = cfg.get('test_label_path')
    if test_label and os.path.exists(test_label):
        with open(test_label, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        TP = sum(1 for k, v in final_results.items() if labels.get(k) == 1 and v == 1)
        TN = sum(1 for k, v in final_results.items() if labels.get(k) == 0 and v == 0)
        FP = sum(1 for k, v in final_results.items() if labels.get(k) == 0 and v == 1)
        FN = sum(1 for k, v in final_results.items() if labels.get(k) == 1 and v == 0)
        total = len(labels)
        accuracy = sum(1 for k, v in final_results.items() if labels.get(k) == v) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        with open(metrics_txt, 'w', encoding='utf-8') as f:
            f.write(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
    else:
        with open(metrics_txt, 'w', encoding='utf-8') as f:
            f.write("No test_label_path provided or file missing; only per-case results saved.\n")

    log(f"Batch run finished. Results saved to: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default="./config.json", required=False, help='Path to JSON config (see judge.py style)')
    args = parser.parse_args()
    cfg_path = args.cfg_path
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # Ensure max_rounds exists
    # initialize agents
    log('Initializing agents')
    expert = ExpertAgent(cfg)
    # randomly assign plaintiff/defendant as two distinct LVLM-based lawyers
    roles = ['plaintiff', 'defendant']
    random.shuffle(roles)
    plaintiff = LawyerAgent(cfg, role='plaintiff')
    defendant = LawyerAgent(cfg, role='defendant')
    judge = JudgeAgent(cfg)
    # compute a single run_dir here and pass into subroutines / Court so path is defined once
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_dir = cfg.get('dataset_dir', '.')
    dataset_name = cfg.get('dataset_name') or os.path.basename(os.path.normpath(dataset_dir)) or 'dataset'
    dataset_name = dataset_name.replace(os.sep, '_').replace(' ', '_')
    agent_type = cfg.get('agent_type', 'NA')
    # If using a local Qwen-based agent, prefer the provided local path and include the model folder name in run prefix
    model_suffix = ''
    if agent_type in ('qwen2.5-vl', 'qwen3-vl-thinking'):
        qwen_path = cfg.get('qwen_local_path')
        if qwen_path:
            model_name = os.path.basename(os.path.normpath(qwen_path))
            model_name = model_name.replace(os.sep, '_').replace(' ', '_')
            model_suffix = f"_qwen_model-{model_name}"
            log(f"Agent type is {agent_type}: will use local Qwen model at {qwen_path} (model name: {model_name})")
        else:
            log(f"Agent type is {agent_type} but no 'qwen_local_path' found in config; falling back to cfg settings.")
    dir_prefix = f"dataset-{dataset_name}_courtroom-agent_type-{agent_type}{model_suffix}_max_rounds-{cfg.get('max_rounds',0)}_gamma-{cfg.get('gamma',0.5)}_judge_confidence-{cfg.get('judge_confidence_threshold', 0.75)}"
    run_dir = os.path.join('./outputs', f"{timestamp}_{dir_prefix}")
    os.makedirs(run_dir, exist_ok=True)
    # save a copy of the config used for this run into the run_dir for reproducibility
    try:
        shutil.copy(cfg_path, os.path.join(run_dir, os.path.basename(cfg_path)))
        log(f"Saved config {cfg_path} to run dir: {run_dir}")
    except Exception as e:
        log(f"Failed to copy config {cfg_path} to run dir {run_dir}: {e}")

    # If test_set_path exists in config, run batch evaluation and save outputs
    if not cfg.get("single_test"):
        log('Running batch courtroom simulations (this may take a while)')
        out_dir = batch_run_trials(cfg, judge, expert, plaintiff, defendant, human_refs=[], run_dir=run_dir)
        log(f'Batch outputs are in: {out_dir}')
    else:
        # demo single-run (paths below are placeholders; adjust as needed)
        gen_img_path = cfg.get('demo_gen_path', "/data1/humw/Codes/Image_Copy_Detection/PDF-Embedding/D-Rep/Test/gen/gen_CAP000008.jpg")
        real_img_path = cfg.get('demo_real_path', "/data1/humw/Codes/Image_Copy_Detection/PDF-Embedding/D-Rep/Test/real/real_CAP000008.jpg")
        log('Starting single courtroom simulation (demo)')
        import pdb; pdb.set_trace()
        # use Court to run single trial and persist artifacts
        court = Court(cfg, judge, expert, plaintiff, defendant, run_dir=run_dir)
        final = court.run_trial(gen_img_path, real_img_path, human_refs=[])
        print('\n=== FINAL VERDICT ===')
        print(f'Is infringement: {final.is_infringement}')
        print(f'Score: {final.score_final:.3f}, Confidence: {final.confidence_final:.3f}')
        print(f'Reason: {final.rationale_final}')


if __name__ == '__main__':
    main()
