# 主聊天系统提示词组装与轨迹记录（一期）

## 已确认的范围

- 主聊天始终使用 Fusion 基础规则；用户 `system_prompt` 只是可选偏好。
- 基础规则、能力边界、执行模式规则在 Python 模块维护，使用 Git 管理变更。无新增依赖、数据库表、配置服务或平台。
- 修复历史年份被禁止搜索，以及用户偏好包含规则标题时抑制真实系统规则的问题。
- 保留计划门禁、工具权限、知识库、续跑及上下文裁剪行为；不实现 Skills。
- 仅记录组装成功或失败，没有准备中、加载动画、虚构状态或历史补填。
- 一个 Run 只有一条系统提示词组装记录；真实模型请求另行携带有效 system 消息指纹。
- 不将提示词全文或用户偏好原文写入事件、日志或前端详情。

## 代码与运行边界

`app/ai/prompts/agent_loop.py` 继续维护模板；新增纯本地组装模块，把可信段落身份与提示词文本分离。文件、URL、知识库 IO 不计入组装耗时。

主聊天基础、工具边界、计划规则与续跑模板切为代码来源。收尾摘要、工具 description、标题、推荐问题及文件分析保留现有解析路径。2026-08-26（Asia/Shanghai）已只读核对 dev：前述迁移项均与代码一致；`limit_summary` 的线上内容没有代码默认的附加非披露段，因此本期不迁移它。PromptHub 的其他消费者与同步服务不受影响。

## 协议

新增 `system_prompt_prepared`，沿用 Agent envelope 与 `protocol_version: 2`，载荷为：

```typescript
{
  status: 'ready' | 'failed';
  source: 'code';
  template_version: string;
  section_ids: string[];
  fingerprint?: string | null;
  char_count?: number | null;
  duration_ms: number;
  error_code?: string | null;
  message?: string | null; // 固定安全文案，不使用异常原文
}
```

`ready` 必须来自真实组装结果，`failed` 只用于组装本身抛错。知识库无证据而直接回复、组装前失败、旧 Run 均不凭空产生记录。

`llm_round_started` 新增可选 `system_prompt_fingerprint`（SHA-256）。哈希对象是最终语言规则和上下文管理之后、交给主聊天 LLM 调用的有序 system 消息数组，保留消息边界和内容结构。主循环与收尾请求都覆盖；不宣称包含用户/工具消息或工具 schema，也不推断历史指纹。

前端 normalizer、Redux/SSE 消费和历史 API 共用相同契约。Trajectory 按 Run 投影为一条上下文记录，成功文案“系统提示词已组装”，失败“系统提示词组装失败”；详情展示来源、模板版本、段落、指纹、字符数及耗时。模型请求详情展示该请求的实际 system 指纹。普通聊天视图不新增状态，原 token 上下文事件语义不变。

## 验收

无偏好仍有基础规则；偏好标题碰撞不影响规则；组装选择与模式一致；日期保留历史查询；事件只带安全元数据；成功、失败、未执行三分支准确；两类 LLM 请求记录最终有效指纹；实时与历史事件投影一致且只一行；旧事件兼容。

代码门禁通过不等于模型效果或浏览器验收。默认不启动本地服务，真实浏览器仅复用已有匹配 Chrome 标签；提交与特性分支 CI 按仓库约定执行，合并和部署单独报告，未经授权不做生产发布。
