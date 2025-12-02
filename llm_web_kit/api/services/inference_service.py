# vLLM 作为可选依赖：导入失败时保持模块可用，实际使用时再报错
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llm_web_kit.config.cfg_reader import load_config

from ..dependencies import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    model_path: str = ''
    data_path: str = ''
    output_path: str = ''
    use_logits_processor: bool = True
    num_workers: int = 8
    max_tokens: int = 32768
    temperature: float = 0
    top_p: float = 0.95
    max_output_tokens: int = 8192
    tensor_parallel_size: int = 1
    # 测试环境修改为float16
    dtype: str = 'float16'
    template: bool = True


config = InferenceConfig(
    model_path='',  # checkpoint-3296路径
    output_path='',
    use_logits_processor=True,  # 启用逻辑处理器确保JSON格式输出
    num_workers=8,  # 并行工作进程数
    max_tokens=26000,  # 最大输入token数
    temperature=0,  # 确定性输出
    top_p=0.95,
    max_output_tokens=8192,  # 最大输出token数
    tensor_parallel_size=1,  # 张量并行大小
    template=True  # 启用聊天模板
)

PROMPT = """As a front-end engineering expert in HTML, your task is to analyze the given HTML structure and accurately classify elements with the _item_id attribute as either "main" (primary content) or "other" (supplementary content). Your goal is to precisely extract the primary content of the page, ensuring that only the most relevant information is labeled as "main" while excluding navigation, metadata, and other non-essential elements.
Guidelines for Classification:
Primary Content ("main")
Elements that constitute the core content of the page should be classified as "main". These typically include:
✅ For Articles, News, and Blogs:
The main text body of the article, blog post, or news content.
Images embedded within the main content that contribute to the article.
✅ For Forums & Discussion Threads:
The original post in the thread.
Replies and discussions that are part of the main conversation.
✅ For Q&A Websites:
The question itself posted by a user.
Answers to the question and replies to answers that contribute to the discussion.
✅ For Other Content-Based Pages:
Any rich text, paragraphs, or media that serve as the primary focus of the page.
Supplementary Content ("other")
Elements that do not contribute to the primary content but serve as navigation, metadata, or supporting information should be classified as "other". These include:
❌ Navigation & UI Elements:
Menus, sidebars, footers, breadcrumbs, and pagination links.
"Skip to content" links and accessibility-related text.
❌ Metadata & User Information:
Article titles, author names, timestamps, and view counts.
Like counts, vote counts, and other engagement metrics.
❌ Advertisements & Promotional Content:
Any section labeled as "Advertisement" or "Sponsored".
Social media sharing buttons, follow prompts, and external links.
❌ Related & Suggested Content:
"Read More", "Next Article", "Trending Topics", and similar sections.
Lists of related articles, tags, and additional recommendations.
Task Instructions:
You will be provided with a simplified HTML structure containing elements with an _item_id attribute. Your job is to analyze each element's function and determine whether it should be classified as "main" or "other".
Response Format:
Return a JSON object where each key is the _item_id value, and the corresponding value is either "main" or "other", as in the following example:
{{"1": "other","2": "main","3": "other"}}
🚨 Important Notes:
Do not include any explanations in the output—only return the JSON.
Ensure high accuracy by carefully distinguishing between primary content and supplementary content.
Err on the side of caution—if an element seems uncertain, classify it as "other" unless it clearly belongs to the main content.

Input HTML:
{alg_html}

Output format should be a JSON-formatted string representing a dictionary where keys are item_id strings and values are either 'main' or 'other'. Make sure to include ALL item_ids from the input HTML.
"""


def create_prompt(alg_html: str) -> str:
    return PROMPT.format(alg_html=alg_html)


def add_template(prompt: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )
    return chat_prompt


class State(Enum):
    Left_bracket = 0
    Right_bracket = 1
    Space_quote = 2
    Quote_colon_quote = 3
    Quote_comma = 4
    Main_other = 5
    Number = 6


class Token_state:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        token_id_map = {
            State.Left_bracket: ['{'],
            State.Right_bracket: ['}'],
            State.Space_quote: [' "'],
            State.Quote_colon_quote: ['":"'],
            State.Quote_comma: ['",'],
            State.Main_other: ['main', 'other'],
            State.Number: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        }
        self.token_id_map = {k: [self.tokenizer.encode(v)[0] for v in token_id_map[k]] for k in token_id_map}

    def mask_other_logits(self, logits: torch.Tensor, remained_ids: List[int]):
        remained_logits = {ids: logits[ids].item() for ids in remained_ids}
        new_logits = torch.ones_like(logits) * -float('inf')
        for id in remained_ids:
            new_logits[id] = remained_logits[id]
        return new_logits

    def calc_max_count(self, prompt_token_ids: List[int]):
        pattern_list = [716, 1203, 842, 428]
        for idx in range(len(prompt_token_ids) - len(pattern_list), -1, -1):
            if all(prompt_token_ids[idx + i] == pattern_list[i] for i in range(len(pattern_list))):
                num_idx = idx + len(pattern_list)
                num_ids = []
                while num_idx < len(prompt_token_ids) and prompt_token_ids[num_idx] in self.token_id_map[State.Number]:
                    num_ids.append(prompt_token_ids[num_idx])
                    num_idx += 1
                # return int(self.tokenizer.decode(num_ids)) + 1
                return int(self.tokenizer.decode(num_ids))
        return 1

    def find_last_complete_number(self, input_ids: List[int]):
        if not input_ids:
            return -1, 'null', -1

        tail_number_ids = []
        last_idx = len(input_ids) - 1
        while last_idx >= 0 and input_ids[last_idx] in self.token_id_map[State.Number]:
            tail_number_ids.insert(0, input_ids[last_idx])
            last_idx -= 1

        tail_number = int(self.tokenizer.decode(tail_number_ids)) if tail_number_ids else -1

        while last_idx >= 0 and input_ids[last_idx] not in self.token_id_map[State.Number]:
            last_idx -= 1

        if last_idx < 0:
            return tail_number, 'tail', tail_number

        last_number_ids = []
        while last_idx >= 0 and input_ids[last_idx] in self.token_id_map[State.Number]:
            last_number_ids.insert(0, input_ids[last_idx])
            last_idx -= 1

        last_number = int(self.tokenizer.decode(last_number_ids))

        if tail_number == last_number + 1:
            return tail_number, 'tail', tail_number
        return last_number, 'non_tail', tail_number

    def process_logit(self, prompt_token_ids: List[int], input_ids: List[int], logits: torch.Tensor):
        if not input_ids:
            return self.mask_other_logits(logits, self.token_id_map[State.Left_bracket])

        last_token = input_ids[-1]

        if last_token == self.token_id_map[State.Right_bracket][0]:
            return self.mask_other_logits(logits, [151645])
        elif last_token == self.token_id_map[State.Left_bracket][0]:
            return self.mask_other_logits(logits, self.token_id_map[State.Space_quote])
        elif last_token == self.token_id_map[State.Space_quote][0]:
            last_number, _, _ = self.find_last_complete_number(input_ids)
            # next_char = str(last_number + 1)[0]
            if last_number == -1:
                next_char = '1'
            else:
                next_char = str(last_number + 1)[0]

            return self.mask_other_logits(logits, self.tokenizer.encode(next_char))
        elif last_token in self.token_id_map[State.Number]:
            last_number, state, tail_number = self.find_last_complete_number(input_ids)
            if state == 'tail':
                return self.mask_other_logits(logits, self.token_id_map[State.Quote_colon_quote])
            else:
                next_str = str(last_number + 1)
                next_char = next_str[len(str(tail_number))]
                return self.mask_other_logits(logits, self.tokenizer.encode(next_char))
        elif last_token == self.token_id_map[State.Quote_colon_quote][0]:
            return self.mask_other_logits(logits, self.token_id_map[State.Main_other])
        elif last_token in self.token_id_map[State.Main_other]:
            return self.mask_other_logits(logits, self.token_id_map[State.Quote_comma])
        elif last_token == self.token_id_map[State.Quote_comma][0]:
            last_number, _, _ = self.find_last_complete_number(input_ids)
            max_count = self.calc_max_count(prompt_token_ids)
            if last_number >= max_count:
                return self.mask_other_logits(logits, self.token_id_map[State.Right_bracket])
            else:
                return self.mask_other_logits(logits, self.token_id_map[State.Space_quote])

        return logits


def reformat_map(text):
    try:
        data = json.loads(text)
        return {'item_id ' + k: 1 if v == 'main' else 0 for k, v in data.items()}
    except json.JSONDecodeError:
        return {}


def main(simplified_html: str, model: object, tokenizer: object, model_path: str):
    # tokenizer = AutoTokenizer.from_pretrained("/share/liukaiwen/models/qwen3-0.6b/checkpoint-3296", trust_remote_code=True)
    # simplified_html = simplify_html(ori_html)
    # print("sim_html length", len(simplified_html))
    if SamplingParams is None:
        raise RuntimeError(
            '当前环境未安装 vLLM 或安装失败，无法执行模型推理。建议在 Linux+NVIDIA GPU 环境安装 vLLM，' +
            '或在 API 中使用占位/替代推理实现。原始导入错误: {}'.format('_VLLM_IMPORT_ERROR')
        )
    prompt = create_prompt(simplified_html)
    chat_prompt = add_template(prompt, tokenizer)

    if config.use_logits_processor:
        token_state = Token_state(model_path)
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_output_tokens,
            logits_processors=[token_state.process_logit]
        )
    else:
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_output_tokens
        )

    output = model.generate(chat_prompt, sampling_params)
    output_json = clean_output(output)
    return output_json


def clean_output(output):
    prediction = output[0].outputs[0].text

    # Extract JSON from prediction
    start_idx = prediction.rfind('{')
    end_idx = prediction.rfind('}') + 1

    if start_idx != -1 and end_idx != -1:
        json_str = prediction[start_idx:end_idx]
        json_str = re.sub(r',\s*}', '}', json_str)  # Clean JSON
        try:
            json.loads(json_str)  # Validate
        except Exception:
            json_str = '{}'
    else:
        json_str = '{}'

    return json_str


class InferenceService:
    """对外暴露的推理服务封装，供 HTMLService 调用。"""

    def __init__(self):
        """初始化推理服务，延迟加载模型."""
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None  # 新增采样参数成员
        self._initialized = False
        self._init_lock = None  # 用于异步初始化锁
        self._model_path = None

    async def warmup(self):
        """在服务启动阶段主动预热模型（异步初始化）。"""
        await self._ensure_initialized()

    async def _ensure_initialized(self):
        """确保模型已初始化（异步安全）"""
        if not self._initialized:
            if self._init_lock is None:
                import asyncio
                self._init_lock = asyncio.Lock()

            async with self._init_lock:
                if not self._initialized:  # 双重检查
                    await self._init_model()
                    self._initialized = True

    async def _init_model(self):
        """初始化模型和tokenizer."""
        try:
            llm_config = load_config(suppress_error=True)
            self.model_path = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else llm_config.get('model_path',
                                                                                                         None)
            if self.model_path is None:
                raise RuntimeError('model_path为空，未配置模型路径')
            if SamplingParams is None:
                raise RuntimeError(
                    '当前环境未安装 vLLM 或安装失败，无法执行模型推理。建议在 Linux+NVIDIA GPU 环境安装 vLLM，' +
                    '或在 API 中使用占位/替代推理实现。原始导入错误: {}'.format('_VLLM_IMPORT_ERROR')
                )

            # 初始化 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 初始化 LLM 模型
            self._llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                dtype=config.dtype,
                tensor_parallel_size=config.tensor_parallel_size,
                # 测试环境取消注释
                max_model_len=config.max_tokens,  # 减少序列长度避免内存不足
            )

            # 在初始化时创建采样参数
            if config.use_logits_processor:
                token_state = Token_state(self.model_path)
                self._sampling_params = SamplingParams(
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_output_tokens,
                    logits_processors=[token_state.process_logit]
                )
            else:
                self._sampling_params = SamplingParams(
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_tokens=config.max_output_tokens
                )

            logger.info(f'模型和采样参数初始化成功: {self.model_path}')

        except Exception as e:
            logger.error(f'模型初始化失败: {e}')
            # 如果模型初始化失败，保持为 None，后续调用会返回占位结果
            self._llm = None
            self._tokenizer = None

    async def inference(self, simplified_html: str, options: dict | None = None) -> dict:
        """执行推理，如果模型未初始化则返回占位结果."""
        try:
            await self._ensure_initialized()

            if self._llm is None or self._tokenizer is None:
                logger.error('模型未初始化，返回占位结果')
                return self._get_placeholder_result()

            # 执行真实推理
            return await self._run_real_inference(simplified_html, options)

        except Exception as e:
            logger.error(f'推理过程出错: {e}')
            return self._get_placeholder_result()

    async def _run_real_inference(self, simplified_html: str, options: dict | None = None) -> dict:
        """执行真实的模型推理."""
        try:
            # 创建 prompt
            prompt = create_prompt(simplified_html)
            chat_prompt = add_template(prompt, self._tokenizer)

            # 直接使用初始化好的采样参数
            if self._sampling_params is None:
                logger.error("采样参数未初始化，返回占位结果")
                return self._get_placeholder_result()

            # 执行推理
            start_time = time.time()
            output = self._llm.generate(chat_prompt, self._sampling_params)
            end_time = time.time()
            output_json = clean_output(output)

            # 格式化结果
            result = reformat_map(output_json)
            logger.info(f'推理完成，结果：{result}')
            logger.info(f'推理完成， 耗时: {end_time - start_time}秒')
            return result

        except Exception as e:
            logger.error(f'真实推理失败: {e}')
            return self._get_placeholder_result()

    def _get_placeholder_result(self) -> dict:
        """返回占位结果."""
        return {}


if __name__ == '__main__':
    config = InferenceConfig(
        model_path='',
        output_path='',
        use_logits_processor=True,
        num_workers=8,
        max_tokens=26000,
        temperature=0,
        top_p=0.95,
        max_output_tokens=8192,
        tensor_parallel_size=1,
        template=True,
    )
    try:
        llm_config = load_config(suppress_error=True)
        model_path = llm_config.get('model_path', None)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LLM(model=model_path,
                    trust_remote_code=True,
                    dtype=config.dtype,
                    # 设置最大模型长度
                    max_model_len=config.max_tokens,
                    tensor_parallel_size=config.tensor_parallel_size)

        simplified_html = '<html><body><h1>Hello World</h1></body></html>'
        response_json = main(simplified_html, model, tokenizer)
        llm_response_dict = reformat_map(response_json)
    except Exception:
        raise
    finally:
        import torch.distributed as dist

        # 在程序结束前添加
        if dist.is_initialized():
            dist.destroy_process_group()
