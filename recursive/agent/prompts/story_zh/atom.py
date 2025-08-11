#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register


ATOMIC_JUDGMENT_LOGIC = """
# 任务复杂度判定

## 核心原则
- 原子任务（atomic）：LLM能在单次调用中高质量完成的任务
- 复杂任务（complex）：需要进一步分解才能保证质量的任务

## 判定标准

## write任务
- 字数≤2000字 且 有明确依赖（dependency不为空） 且 设计结果详细 → 原子
- 字数>3000字 或 无依赖（dependency为空） 或 目标宽泛或抽象 或 多个不同类型  → 复杂

## think任务
- 单一设计要素 且 信息充足 且 范围有限 → 原子
- 多要素协调 或 系统性规划 或 设计范围大 或 需要分层思考 → 复杂

## 判定原则
- 确保每个原子任务都能产出高质量结果
- 平衡效率与质量，避免过度分解
""".strip()


CONTENT_TEMPLATE = """
# 创作目标与进度
## 终极目标
<root_question>
{to_run_root_question}
</root_question>

## 已完成内容概要
<story_context>
{to_run_mem0_content}
</story_context>

## 最新章节（接续点）
<article_latest>
{to_run_article_latest}
</article_latest>

# 规划层次与依赖
## 整体写作计划
<full_plan>
{to_run_full_plan}
</full_plan>

## 上级设计蓝图
<outer_graph_dependent>
{to_run_outer_graph_dependent}
</outer_graph_dependent>

## 同级已完成设计
<same_graph_dependent>
{to_run_same_graph_dependent}
</same_graph_dependent>

# 当前待评估任务
<target_task>
{to_run_task}
</target_task>

请基于以上信息，判定当前任务的复杂度：
""".strip()


@prompt_register.register_module()
class StoryAtomZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = f"""
# 角色
你是小说任务复杂度评估专家，精确判定任务是否适合作为原子任务直接执行。

{ATOMIC_JUDGMENT_LOGIC}

# 输出格式
- 不要有多余解释：
<result>
<atomic_task_determination>
atomic/complex
</atomic_task_determination>
</result>
""".strip()

        super().__init__(system_message, CONTENT_TEMPLATE)
        

@prompt_register.register_module()
class StoryAtomUpdateZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = f"""
# 角色
你是小说写作规划系统中的目标优化与任务复杂度评估专家

# 核心职责
1. 基于上下文优化任务目标，提高可执行性
2. 判定优化后任务的复杂度

# 优化原则
- 整合已有结论，避免重复工作
- 消除冲突矛盾，确保目标一致
- 具体化目标，保持核心意图

{ATOMIC_JUDGMENT_LOGIC}

# 输出格式
- 不要有多余解释：
<result>
<goal_updating>
优化后的目标描述；如无需优化则输出原始目标
</goal_updating>
<atomic_task_determination>
atomic/complex
</atomic_task_determination>
</result>
""".strip()

        super().__init__(system_message, CONTENT_TEMPLATE)