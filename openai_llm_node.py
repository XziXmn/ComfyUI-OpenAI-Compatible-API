"""
OpenAI Compatible API Node for ComfyUI
支持文本和图片输入，调用兼容 OpenAI 的 /v1/chat/completions 接口

功能亮点：
  • 纯文本对话
  • 视觉模型支持（图片输入）
  • 灵活种子控制（random / fixed / increment / decrement）
  • 可开关 seed 发送（兼容 Ollama、LM Studio 等不支持 seed 的后端）
  • 详细 tooltip 提示 + 规避 ComfyUI 自动注入
"""
import base64
import io
import json
import requests
from PIL import Image
import torch
import numpy as np


class OpenAICompatibleLLM:
    """
    OpenAI 兼容 API 调用节点
    支持纯文本对话和带图片的视觉对话
    """
    def __init__(self):
        self.last_seed = 233  # 用于 increment / decrement 记忆

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "hello",
                    "display": "textarea",
                    "tooltip": "输入的提示词，支持多行文本"
                }),
                "endpoint": ("STRING", {
                    "default": "http://localhost:3010/v1/chat/completions",
                    "tooltip": "兼容 OpenAI 的 API 地址，例如 http://localhost:3010/v1/chat/completions"
                }),
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "指定使用的模型名称，若留空则由后端决定"
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 1,
                    "max": 32000,
                    "step": 1,
                    "tooltip": "生成文本的最大 token 数，范围 1~32000"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "采样温度，0.0 更确定性，2.0 更随机"
                }),
                # 规避 ComfyUI 自动注入：改名为 seed_value
                "seed_value": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 9223372036854775807,
                    "step": 1,
                    "tooltip": "固定种子值，仅在“种子模式”为 fixed 时使用"
                }),
                "seed_control": (["random", "fixed", "increment", "decrement"], {
                    "default": "random",
                    "tooltip": "种子行为控制：\n"
                               "- random：每次随机生成\n"
                               "- fixed：使用固定 seed_value\n"
                               "- increment：基于上次种子 +1\n"
                               "- decrement：基于上次种子 -1"
                }),
                "use_seed": (["no", "yes"], {
                    "default": "no",
                    "tooltip": "是否向 API 发送 seed 参数（部分后端不支持，启用后可提高可复现性）"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "可选图片输入，仅视觉模型支持，格式为 ComfyUI tensor"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API 密钥（Bearer Token），若后端需要认证请填写"
                }),
                "image_detail": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "图片细节级别（仅视觉模型）：\n"
                               "- auto：自动决定\n"
                               "- low：低分辨率快速处理\n"
                               "- high：高分辨率详细分析"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "OpenAI Compatible"

    # ==================== 核心函数 ====================
    def tensor_to_base64(self, tensor_image):
        """将 ComfyUI tensor 转为 base64 图片字符串"""
        if len(tensor_image.shape) == 4:
            tensor_image = tensor_image[0]
        img_np = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    def generate(self, prompt, endpoint, model, max_tokens, temperature,
                 seed_value, seed_control, use_seed,
                 image=None, api_key=None, image_detail="auto"):
        """
        调用 OpenAI 兼容 API 生成文本
        """
        # ------------------- 种子控制逻辑 -------------------
        if seed_control == "random":
            import random
            actual_seed = random.randint(0, 9223372036854775807)
        elif seed_control == "fixed":
            actual_seed = seed_value
        elif seed_control == "increment":
            actual_seed = min(9223372036854775807, self.last_seed + 1)
        elif seed_control == "decrement":
            actual_seed = max(0, self.last_seed - 1)
        else:
            actual_seed = seed_value
        self.last_seed = actual_seed  # 记忆用于下次 increment/decrement

        # ------------------- 构建消息内容 -------------------
        if image is not None:
            image_base64 = self.tensor_to_base64(image)
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64, "detail": image_detail}}
            ]
        else:
            message_content = prompt

        # ------------------- 构建请求体 -------------------
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        # 仅在启用时添加 seed
        if use_seed == "yes":
            payload["seed"] = actual_seed

        # ------------------- 请求头 -------------------
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"

        # ------------------- 发送请求 -------------------
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            if result.get("choices") and len(result["choices"]) > 0:
                return (result["choices"][0]["message"]["content"],)
            else:
                return ("[错误] API 返回空响应",)
        except requests.exceptions.RequestException as e:
            error_msg = f"[请求错误] {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    detail = e.response.json()
                    error_msg += f"\n详情: {json.dumps(detail, indent=2, ensure_ascii=False)}"
                except:
                    error_msg += f"\n原始响应: {e.response.text}"
            print(error_msg)
            return (error_msg,)
        except Exception as e:
            error_msg = f"[未知错误] {str(e)}"
            print(error_msg)
            return (error_msg,)


# ==================== 注册节点（ComfyUI 必须）===================
NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleLLM": OpenAICompatibleLLM
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleLLM": "OpenAI 兼容 LLM"
}
