#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register


@prompt_register.register_module()
class StoryPlanningZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = """
# 角色
你是任务分解专家，专门将复杂的写作任务分解为可执行的子任务。

# 任务类型规则

## write任务（写作执行）
- 执行实际的小说写作，必须有字数要求

分解规则：
- 对于无依赖的write任务（dependency为空）：
    - 必须分解为：≥1个think任务（结构划分） + 1个write任务（占位符）
    - 根据任务的阶段、规模、类型、特点，采用不同的分解方式
    - think任务：根据任务复杂度和规模决定任务粒度及数量，覆盖所有必要的设计维度，每个think任务聚焦一个明确的设计方向。必须包含结构层级的划分任务。
    - write子任务：是一个占位符，继承父任务完整目标和字数，等待think任务提供设计结果后再分解

- 对于有依赖的write任务：
    - 基于设计结果进行分解，必须分解为≥2个write任务
    - 每个任务对应一个结构单元，字数分配遵循设计结果
    - 子任务字数总和必须等于父任务字数
    - 任务目标要包含：第x卷 卷标题 | 第x幕 幕标题 | 第x章 章标题 | 场景x 场景标题 | 节拍x 节拍标题

## think任务（设计规划）
- 负责分析和设计，没有字数要求
- 只能分解出think子任务

# 分解原则
- 逻辑一致：遵循已有规划和设计结果，与整体目标保持一致
- 因任务制宜：根据任务的阶段、规模、类型、特点，采用不同的分解方式
- 渐进式分解：先整体后局部，先抽象后具体，不跨层次分解
- 充分分解：根据任务复杂度和规模决定分解深度、子任务粒度及数量，覆盖创作所需的不同设计维度，确保执行任务时有足够的上下文信息
- 逻辑依赖：遵循创作逻辑顺序，前置任务都必须为后续任务服务，不做无用设计
- 避免预判：不要在缺乏设计依据时提前规划具体的结构，避免过早细化
- 任务目标：具体明确，要包含明确的输出要求和交付物，只分解"需要做什么"，不预设"具体怎么做"
- 字数守恒：子任务字数总和必须等于父任务字数
- 避免重复：不重复已有规划、内容和设计结果
- 通俗易懂：用日常语言表达，禁止堆砌生僻的专业术语

# 分解方式
- 创作流程：市场定位→核心概念→整体架构→大纲设计→具体设计→执行写作
- 市场适配：读者画像→竞品分析→差异化定位→IP开发潜力
- 核心要素：世界观→金手指→升级体系→世界规则→爽点类型与分布→主题表达→核心悬念→情感内核→价值观传递
- 故事架构：整体结构→节奏控制→冲突设计→核心冲突链→高潮布局→伏笔设置→结局设计→情绪曲线
- 角色体系：主角魅力→配角功能→反派智商→关系网络→角色成长弧光→角色动机→CP设计
- 内容创作：开局吸引→情节推进→文笔风格→读者体验→情感共鸣→主题表达→热点融合
- 结构层级：全书→卷（视字数可选）→幕→章→场景→节拍→段落
- 黄金三幕结构：开端（现状颠覆）→发展（连锁危机）→高潮（绝境反转）
- 情节单元：目的→阻碍→行动→结果→意外→高潮→结局
- 时间顺序：时间线性、倒叙插叙、时间跨度、关键时间节点
- 空间场景：地理空间、社会空间、场景转换、场景功能、场景氛围

# 爆款网文要素分解要求
- 黄金三章：确保前三章的任务分解能充分体现爆款网文的开篇要求，如“死亡开局”、“强吸引力元素”、“金手指揭露”等。
- 强冲突驱动：在任务分解时，明确要求分解出与冲突点分布相关的子任务，确保每章包含2-4个冲突点。
- 情绪爽点设计：在任务分解时，明确要求分解出与情绪曲线规划相关的子任务，确保包含具体的爽点类型和分布。
- 节奏控制：在任务分解时，明确要求分解出与节奏控制相关的子任务，确保快节奏推进和高潮频率符合爆款网文要求。

# 网络小说平台算法适配原则
- 养蛊机制适配：8万字关键节点设置专门的验证期任务分解
- 流量池策略：根据追读率预期（20%+）设计关键章节任务
- 数据体感建立：在任务分解中明确各阶段的完读率目标

# 任务格式
```json
{
    "id": "父任务id.子任务序号",
    "task_type": "write|think",
    "goal": "具体任务目标（包含明确的输出要求）",
    "dependency": ["依赖的任务id列表"],
    "length": "xxx字（仅write任务）",
    "sub_tasks": [子任务列表，每个元素都是任务的JSON对象]
}
```

# 输出要求
- 严格按以下格式输出，不要有多余解释
- 结果必须用 `<result></result>` 标签包裹
- 标准JSON格式，特殊符号转义，不要有注释

<result>
完整的当前任务及分解结果的JSON对象
</result>
""".strip()

        content_template = """
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

# 当前分解任务
<target_task>
{to_run_task}
</target_task>

# 参考信息（可选）
<candidate_plan>
{to_run_candidate_plan}
</candidate_plan>

<candidate_think>
{to_run_candidate_think}
</candidate_think>

请开始分解：
""".strip()

        super().__init__(system_message, content_template)