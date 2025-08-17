import os
import json
import copy
import time
from typing import List, Dict, Any, Optional
from loguru import logger
import litellm
from litellm.caching.caching import Cache


class LiteLLMProxy:
    def __init__(self):
        litellm.cache = Cache(type='disk')
        # litellm.input_callback = ["lunary"]
        # litellm.success_callback = ["lunary"]
        # litellm.failure_callback = ["lunary"]

    def call(self, model=None, messages=None, no_cache=False, overwrite_cache=False, tools=None, temperature=None, **kwargs):
        
        messages = copy.deepcopy(messages)
        
        litellm_params = {
            'model': model,
            'messages': messages,
            'caching': True,
            'max_tokens': 131072,
            'max_completion_tokens': 131072,
            'timeout': 300,
            'num_retries': 2,
            'respect_retry_after': True,
            'fallbacks': [
                'openrouter/deepseek/deepseek-r1-0528:free',
                'openai/deepseek-ai/DeepSeek-R1-0528'
                ]
        }
        
        if temperature is not None:
            litellm_params['temperature'] = temperature
        if tools is not None:
            litellm_params['tools'] = tools
        
        for key, value in kwargs.items():
            if key not in litellm_params and value is not None:
                litellm_params[key] = value
        
        # logger.info(f"LiteLLMProxy litellm_params={litellm_params}")
        response = litellm.completion(**litellm_params)
        return response.choices






    def call_fast(self, model=None, messages=None, no_cache=False, overwrite_cache=False, tools=None, temperature=None, **kwargs):
            
        messages = copy.deepcopy(messages)
        
        litellm_params = {
            'model': model,
            'messages': messages,
            'caching': True,
            'max_tokens': 131072,
            'max_completion_tokens': 131072,
            'timeout': 300,
            'num_retries': 2,
            'respect_retry_after': True,
            'fallbacks': [
                'openai/deepseek-ai/DeepSeek-V3',
                'openrouter/deepseek/deepseek-chat-v3-0324:free'
                ]
        }
        
        if temperature is not None:
            litellm_params['temperature'] = temperature
        if tools is not None:
            litellm_params['tools'] = tools
            
        for key, value in kwargs.items():
            if key not in litellm_params and value is not None:
                litellm_params[key] = value
        
        # logger.info(f"call_fast() litellm_params={litellm_params}")
        response = litellm.completion(**litellm_params)
        return response.choices





