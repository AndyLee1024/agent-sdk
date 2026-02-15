## 1) 工作方式（硬规则）
- 先方案后编码：任何任务先给 2-3 个方案（Quick / Balanced / Long-term）+ 风险 + 你需要问我的关键问题；我明确批准前禁止改代码。
- 不清楚就问：缺信息/不确定/有歧义，必须先问我，禁止猜。
- 小步提交：一次只做一个小步骤（优先 ≤3 个文件、≤200 行净新增），超过就拆步。

## 2) Anti-God-Object（硬护栏）
- 单文件上限：任何改动后，任何文件不得 >700 LOC。
  - 如果目标文件已 >700：本次禁止净增行数；要么抽取拆分，要么先给拆分方案让我选。
- self 状态耦合：当一个类已经“很多 self.*”时，禁止继续往里加状态/方法；必须先外置状态或拆分。
- 触发以上任一条：立刻停止编码，输出拆分方案（不写代码），包含：新模块/文件清单、职责边界、State/Config 方案、分步迁移、回归风险。

## 3) 工程化修复要求（排查/修 bug 必须）
- 先全局分析：根因 + 影响面 + 连锁反应 + 回滚/缓解（测试点/隔离/适配），再动手改。

## 4) Python 规则（硬规则）
- 必须用 f-string；必须用 logging；禁止 print。
- 运行用 `uv run python xxx.py`；装包用 `uv add 包名`。
- 禁止硬编码路径/配置：用环境变量或配置文件集中到 Config。
- 遵循 PEP8；用 black 格式化。

## 5) Prompt_toolkit / TUI 铁律（硬规则）
- 禁止 History TUI：聊天历史只写入终端 scrollback；UI layout 只允许底部输入/补全/状态栏；历史输出唯一通道是 `run_in_terminal(...)` 追加。
- 禁止任何承载历史的 multiline read-only 组件（例如 TextArea(multiline=True, read_only=True) 或 messages/history/chat 命名的窗口/控件）。
- 禁止回写已输出到 scrollback 的历史行（只追加不可变）。
- 涉及 prompt_toolkit：必须使用 context7 mcp 的 resolve-library-id + query-docs 校验用法；查不到就停下问我。

## 6) 其他
- 不要给我生成“总结文档/报告类新文件”，除非我明确要求。
- 规则冲突或无法同时满足：停止并说明冲突点，向我提问。
