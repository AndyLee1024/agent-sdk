## global rule
0. 设计应该尽可能简单，而不是更简单。 Keep It Simple, Stupid。
1. "Simplicity is the ultimate sophistication." — Leonardo da Vinci
2. 如果你遇到不清楚的问题或者缺少什么信息，MUST及时向我询问。  
3. 先给出你的方案，MUST经我审阅后再编写代码。
4. 给出几版方案选择，构建方案时使用问答的方式逐步确认我的需求。 this is MUST
5. 不要为我生成任何总结文档, 除非我主动告诉你.
6. 当用户让你排查或者修复问题时，我希望你站在全局角度工程化考虑问题,结合当前架构,分析问题原因,给出解决方案。而不是简单止血。需要确认修改这个会不会造成其他依赖这个的功能的连锁反应和冲突. this is MUST
7. 涉及到 Prompt_toolkit 这个TUI框架, 你必须使用 context7 mcp里面的  query-docs和resolve-library-id 功能来查询相关文档, 以确保你对这个框架的理解是正确的. this is MUST

## python coding rule
1. 代码必须使用 f-string 进行字符串格式化
2. 代码必须使用 logging 模块进行日志记录，禁止使用 print 语句
3. 运行 python必须使用 uv run python 脚本名.py 的方式运行
4. 安装pip包必须使用 uv add 包名 的方式安装
5. 禁止硬编码任何路径或者配置，必须使用环境变量或者配置文件的方式进行配置
6. 代码必须遵循 PEP 8 风格指南，使用黑色（black）进行代码格式化


## KISS 原则
 KISS 的好处

   1 易于理解 — 代码一看就懂
   2 易于维护 — 修改时不会牵一发而动全身
   3 易于测试 — 同步代码测试简单
   4 性能更好 — 没有不必要的异步开销
   5 减少 Bug — 简单代码出错概率低
  
  相关原则

   原则   含义                                     场景
   ────────────────────────────────────────────────────────────────────────────
   KISS   保持简单                                 通用设计
   YAGNI  You Ain't Gonna Need It（你不会需要它）  不要提前实现可能用不到的功能
   DRY    Don't Repeat Yourself（不要重复自己）    代码复用
   SOLID  单一职责、开闭原则等                     面向对象设计

## TUI 编程铁律
 
不要做 History TUI。历史永远只写入终端 scrollback，UI 只负责底部输入、补全菜单和状态栏。任何“把聊天历史放进可滚动窗口/HSplit 顶部 TextArea”的实现，一律视为回归并拒绝合并。

1. History 输出的唯一通道

* 只允许 `run_in_terminal(console.print(...))` 或等价的“安全打印”通道。
* 禁止在 prompt_toolkit 布局里存在 `history_area` / `chat_window` / `messages_view` 之类承载历史文本的组件（TextArea/Window/FormattedTextControl 都不行）。
* 禁止为历史实现滚动、重绘、diff、虚拟列表、分页等 UI 行为。所有历史回看依赖终端 scrollback 或外部日志文件。

2. 布局铁律（三行模型）

* Application 布局只允许：`loading_line`（可选） + `input_line` + `status_line`，外加 `CompletionsMenu` 作为 float。
* 任何 HSplit 上方的“内容区”高度 > 1 都视为历史 TUI 试图复活。

3. 输出频率铁律

* 任何 history 追加必须“批量打印”，每个 tick 最多一次 `run_in_terminal()`。
* 禁止按 token/按事件高频打印导致 UI 暂停/恢复频繁切换。

 
 