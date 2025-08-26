#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register


"""
请整体评估 `atom.py` 的提示词，并指出其最大的优势和可以进一步强化的方向。
把固定内容放到 system_message 中，把 可变数据 放到 content_template 中。
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


根据你的分析，直接修改 `atom.py` 文件并提供 diff。
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


改进 这段提示词
要求：清晰、精确、易于理解，在保持质量的同时，尽可能简洁，不要有各种“黑话”和比喻，最好以关键词为主


你的输出被截断了，请从截断的地方继续
"""


###############################################################################


ATOMIC_JUDGMENT_RULES = """
# 评估规则 (满足任一即为“复杂”)
- 定义:
    - 原子任务: 可直接执行，无需分解。
    - 复杂任务: 需要规划、搜索或额外写作。
- 判定:
    1.  分析缺失: 任务所需的设计、分析（如提纲、逻辑）在上下文中不存在。
    2.  信息缺失: 任务所需的外部信息（如数据、文献）在上下文中不存在。
    3.  篇幅过长: 预计产出文本超过 1000字。
""".strip()


###############################################################################


@prompt_register.register_module()
class BookAtomZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = f"""
# 角色
报告任务分析智能体。


# 核心任务
判断“待评估任务”是“原子任务”还是“复杂任务”。


{ATOMIC_JUDGMENT_RULES}


# 输出格式
- 严格按照以下格式输出，不要有任何额外解释。
<result>
<atomic_task_determination>
atomic/complex
</atomic_task_determination>
</result>
""".strip()


        content_template = """
# 当前任务
- 待评估和更新的任务
{to_run_task}


# 总目标
{to_run_root_question}


# 整体写作计划
{to_run_full_plan}


# 上级依赖
- 已完成的搜索与分析任务结果
{to_run_outer_graph_dependent}


# 同级依赖
- 已完成的搜索与分析任务结果
{to_run_same_graph_dependent}


# 已完成的报告内容
{to_run_article}
""".strip()


        super().__init__(system_message, content_template)


###############################################################################


@prompt_register.register_module()
class BookAtomUpdateZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = f"""
# 角色
报告任务优化与分解智能体。


# 核心工作流
1.  更新任务: 基于上下文，优化“待评估和更新的任务”。
    - 精确化: 结合上下文，使任务更具体。
    - 修正: 修正任务中的事实错误。
    - 去重: 移除“已完成的报告内容”中已包含的部分。
    - 保持: 如果任务已清晰、准确，则无需改动。
2.  评估更新后的任务: 判断其为“原子任务”或“复杂任务”。


# 评估规则 (对更新后的任务，满足任一即为“复杂”)
{ATOMIC_JUDGMENT_RULES}


# 任务与输出
严格按照以下步骤和格式执行，不要有任何额外解释。

1.  更新目标: 依据`上下文`（特别是`上级设计`和`同级设计`），优化`当前待评估任务`。
    - 输出到: `<goal_updating>`

2.  判定任务: 对更新后的目标进行判定。
    - 输出到: `<atomic_task_determination>`

<result>
<goal_updating>
优化后的目标描述；如无需优化则输出原始目标
</goal_updating>
<atomic_task_determination>
atomic/complex
</atomic_task_determination>
</result>
""".strip()


        content_template = """
# 当前日期
{today_date}


# 当前任务
- 待评估和更新的任务
{to_run_task}


# 总目标
{to_run_root_question}


# 整体写作计划
{to_run_full_plan}


# 上级依赖
- 已完成的搜索与分析任务结果
{to_run_outer_graph_dependent}


# 同级依赖
- 已完成的搜索与分析任务结果
{to_run_same_graph_dependent}


# 已完成的报告内容
{to_run_article}
""".strip()


        super().__init__(system_message, content_template)
