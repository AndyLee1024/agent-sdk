## 项目概述

这是一个 agent sdk 项目， 用于以后各种agent开发的基础设施。
这是一个非常重要的项目，必须有非常优秀的的 上下文工程 实践的沉淀。 请仔细思考每一个细节。

## global rule
0. 设计应该尽可能简单，而不是更简单。 Keep It Simple, Stupid。
1. 和你交流时，MUST使用中文。 
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

- Chat history 永远不进入 prompt_toolkit layout；历史唯一输出通道是 run_in_terminal(...) 追加到 scrollback。

- layout 允许存在“底部交互带”，其内容仅包括：loading（可选）、输入相关（input/问答 UI/补全菜单等短暂交互组件）、status。

- 禁止出现任何承载历史消息的 multiline read-only 组件（典型：TextArea(multiline=True, read_only=True)、或任何“messages/history/chat”命名的 Window/TextArea/FormattedTextControl）。

- 禁止“回写/修改”已经写进 scrollback 的历史行（比如工具开始打一行，结束回去改成绿色）。scrollback 只允许追加不可变日志。
 
 