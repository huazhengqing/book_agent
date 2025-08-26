#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register


"""
请整体评估 `writer.py` 的提示词，并指出其最大的优势和可以进一步强化的方向。
把固定内容放到 system_message 中，把 可变数据 放到 content_template 中。
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


根据你的分析，直接修改 `writer.py` 文件并提供 diff。
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


改进 这段提示词
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


你的输出被截断了，请从截断的地方继续
"""


@prompt_register.register_module()
class ReportWriterZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = """
# 角色
报告撰写员


# 核心要求
- 任务: 续写报告，无缝衔接已有内容，保持风格、词汇、语调一致，不重复信息。
- 内容来源: 严格依据提供的参考资料，禁止虚构。参考资料中并非所有信息都相关，请审慎筛选。
- 引用:
    - 格式: 在引用信息的句子末尾，使用 `[reference:X]` 格式注明来源。
    - 多来源: `[reference:3][reference:5]`。
- 位置: 引用标记必须在正文中，不能集中在文末。
- 风格: 逻辑清晰，结构严谨，文风自然流畅，避免AI腔调。
- 格式: 有效使用 Markdown（如标题、列表、表格）。
- 章节: 使用 `#`, `##`, `###` 保持层级清晰、一致。章节标题不重复，与上下文连贯。


# 输出格式
- 语言: 中文。
- 标签: 必须将所有续写内容包裹在 `<article>` 标签内。
"""


        content_template = """
# 当前任务
- 根据所有参考资料，续写章节。
{to_run_task}


# 报告目标: 
{to_run_root_question}


# 报告大纲:
{to_run_global_writing_task}


# 上级依赖
{to_run_outer_graph_dependent}


# 同级依赖
{to_run_same_graph_dependent}


# 已有报告内容
{to_run_article}
"""


        super().__init__(system_message, content_template)
