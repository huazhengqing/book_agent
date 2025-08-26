#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()


"""
请整体评估 `search_merge_result.py` 的提示词，并指出其最大的优势和可以进一步强化的方向。
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


根据你的分析，直接修改 `search_merge_result.py` 文件并提供 diff。
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


改进 这段提示词
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


你的输出被截断了，请从截断的地方继续
"""


@prompt_register.register_module()
class BookSearchMergeResultZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = """
# 角色
你是一名顶级的搜索结果整合专家。


# 任务
根据用户提供的【总任务】、【子任务】和【搜索任务】，整合【搜索结果】。


# 规则
- 忠实原文: 内容必须完全来自【搜索结果】，禁止任何推理、杜撰或外部信息。
- 标注来源: 每条信息后，必须以 `webpage[索引号]` 格式注明来源。
- 聚焦任务: 只整合与【搜索任务】目标高度相关的信息。
- 保留细节: 在相关的前提下，保留所有关键细节。


# 输出
- 将所有整合内容放入 `<result>` 标签内。
- 除 `<result>` 标签和其内容外，禁止任何其他输出。
""".strip()


        content_template = """
# 当前日期
{today_date}


# 待整合的搜索结果
{to_run_search_results}


# 写作总任务
{to_run_root_question}


# 当前子任务
{to_run_outer_write_task}


# 本次搜索任务
{to_run_search_task}
""".strip()


        super().__init__(system_message, content_template)
