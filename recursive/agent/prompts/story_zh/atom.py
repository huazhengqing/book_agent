#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register


ATOMIC_JUDGMENT_LOGIC = """
# 原子任务判定规则
依次独立判定以下两类子任务是否需要分解：
1. 设计子任务判定：
   - 如果写作需要特定的设计作为支撑，且这些设计需求未由依赖的设计任务或已完成的小说内容提供，则需要规划设计子任务
   - 设计需求包括：人物设定、情节大纲、场景设计、冲突设置、背景构建等
2. 写作子任务判定：
   - 篇幅≤2000字 且 任务目标单一明确 → 无需分解
   - 篇幅>2000字 或 涉及多个场景/情节转折 或 需要复杂的叙事结构 → 需要分解
3. 综合判定：
   - 如果需要创建设计子任务或写作子任务中的任何一种，该任务就被视为复杂任务
   - 优先保证内容质量，当任务本质复杂度高时，即使篇幅较短也应考虑分解
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
你是递归式专业小说写作规划系统中的原子写作任务判定代理：

在整体计划和已完成小说内容的背景下，评估给定的写作任务是否为原子任务，即无需进一步规划的任务。根据叙事理论和故事写作的结构安排，一个写作任务可以被进一步分解为更细致的写作子任务和设计子任务。写作任务涉及具体文本部分的实际创作，而设计任务可能包括设计核心冲突、人物设定、大纲及详细大纲、关键情节节点、故事背景、情节元素等，以支持实际写作。

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
你是递归式专业小说写作规划系统中的目标更新与原子写作任务判定代理：

1. 目标更新：根据整体计划、已完成的小说内容以及现有的设计结论更新任务目标，使其更贴合需求、合理且详细
2. 原子写作任务判定：在整体计划和已完成小说内容的背景下，评估给定的写作任务是否为原子任务，即无需进一步规划的任务。根据叙事理论和故事写作的结构安排，一个写作任务可以被进一步分解为更细致的写作子任务和设计子任务。写作任务涉及具体文本部分的实际创作，而设计任务可能包括设计核心冲突、人物设定、大纲及详细大纲、关键情节节点、故事背景、情节元素等，以支持实际写作。

# 目标更新
- 根据整体计划、已完成的小说内容以及现有的设计结论更新任务目标，使其更贴合需求、合理且详细
- 更新后的目标需满足：具体明确（包含可衡量的输出要求和交付物）、通俗易懂（避免生僻术语）
- 直接输出更新后的目标。若无需更新，则输出原始目标

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