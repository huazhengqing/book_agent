import os
import json
import copy
import time
from typing import List, Dict, Any, Optional
from loguru import logger
import litellm
from litellm.caching.caching import Cache
from litellm.exceptions import Timeout, AuthenticationError, APIError, RateLimitError


class LiteLLMProxy:
    def __init__(self):
        litellm.cache = Cache(type='disk')
        # litellm.input_callback = ["lunary"]
        # litellm.success_callback = ["lunary"]
        # litellm.failure_callback = ["lunary"]

    def call(self, model, messages=None, tools=None, temperature=None, **kwargs):
        messages = copy.deepcopy(messages)
        litellm_params = {
            'model': model,
            'messages': messages,
            'temperature': 0.0,
            'caching': True,
            'max_tokens': 131072,
            'max_completion_tokens': 131072,
            'timeout': 900,
            'num_retries': 2,
            'respect_retry_after': True
        }
        if temperature is not None:
            litellm_params['temperature'] = temperature
        if tools is not None:
            litellm_params['tools'] = tools
        for key, value in kwargs.items():
            if key not in litellm_params and value is not None:
                litellm_params[key] = value
        logger.info(f"LiteLLMProxy call() {litellm_params}")
        response = litellm.completion(**litellm_params)
        return response.choices

    def call_reasoning(self, messages=None, tools=None, temperature=None, **kwargs):
        messages = copy.deepcopy(messages)
        litellm_params = {
            'model': 'openrouter/deepseek/deepseek-r1-0528:free',
            # 'model': 'openai/deepseek-ai/DeepSeek-R1-0528',
            'messages': messages,
            'temperature': 0.2,
            'caching': True,
            'max_tokens': 131072,
            'max_completion_tokens': 131072,
            'timeout': 900,
            'num_retries': 5,
            'respect_retry_after': True,
            'fallbacks': [
                'openai/deepseek-ai/DeepSeek-R1-0528', 

                # 'openrouter/deepseek/deepseek-r1-0528:free', 

                # 'openrouter/deepseek/deepseek-r1-0528-qwen3-8b', 
                # 'openrouter/qwen/qwen3-32b', 
                # 'openrouter/qwen/qwen3-30b-a3b', 
                # 'openrouter/deepseek/deepseek-r1-distill-llama-70b', 
                ]
        }
        if temperature is not None:
            litellm_params['temperature'] = temperature
        if tools is not None:
            litellm_params['tools'] = tools
        for key, value in kwargs.items():
            if key not in litellm_params and value is not None:
                litellm_params[key] = value
        logger.info(f"LiteLLMProxy call_reasoning() {litellm_params}")
        response = litellm.completion(**litellm_params)
        return response.choices

    def call_fast_zh(self, messages, tools=None, temperature=None, **kwargs):
        messages = copy.deepcopy(messages)
        litellm_params = {
            'model': 'openai/deepseek-ai/DeepSeek-V3',
            # 'model': 'openrouter/deepseek/deepseek-chat-v3-0324:free',
            'messages': messages,
            'temperature': 0.0,
            'caching': True,
            'max_tokens': 163840,
            'max_completion_tokens': 163840,
            'timeout': 300,
            'num_retries': 5,
            'respect_retry_after': True,
            'fallbacks': [
                # 'openai/deepseek-ai/DeepSeek-V3',

                'openrouter/deepseek/deepseek-chat-v3-0324:free',

                # 'openrouter/meta-llama/llama-3.2-3b-instruct', 
                # 'openrouter/liquid/lfm-7b', 
                # 'openrouter/meta-llama/llama-3.1-8b-instruct', 
                # 'openrouter/deepseek/deepseek-r1-0528-qwen3-8b', 
            ]
        }
        if temperature is not None:
            litellm_params['temperature'] = temperature
        if tools is not None:
            litellm_params['tools'] = tools
        for key, value in kwargs.items():
            if key not in litellm_params and value is not None:
                litellm_params[key] = value
        logger.info(f"LiteLLMProxy call_fast_zh() {litellm_params}")
        response = litellm.completion(**litellm_params)
        return response.choices

    def call_fast_en(self, messages, tools=None, temperature=None, **kwargs):
        messages = copy.deepcopy(messages)
        litellm_params = {
            'model': 'openai/deepseek-ai/DeepSeek-V3',
            # 'model': 'openrouter/deepseek/deepseek-chat-v3-0324:free',
            'messages': messages,
            'temperature': 0.0,
            'caching': True,
            'max_tokens': 163840,
            'max_completion_tokens': 163840,
            'timeout': 300,
            'num_retries': 5,
            'respect_retry_after': True,
            'fallbacks': [
                # 'openai/deepseek-ai/DeepSeek-V3',

                'openrouter/deepseek/deepseek-chat-v3-0324:free',

                # 'openrouter/meta-llama/llama-3.2-3b-instruct', 
                # 'openrouter/liquid/lfm-7b', 
                # 'openrouter/meta-llama/llama-3.1-8b-instruct', 
                # 'openrouter/deepseek/deepseek-r1-0528-qwen3-8b', 
            ]
        }
        if temperature is not None:
            litellm_params['temperature'] = temperature
        if tools is not None:
            litellm_params['tools'] = tools
        for key, value in kwargs.items():
            if key not in litellm_params and value is not None:
                litellm_params[key] = value
        logger.info(f"LiteLLMProxy call_fast_en() {litellm_params}")
        response = litellm.completion(**litellm_params)
        return response.choices


###############################################################################


llm_client = LiteLLMProxy()




