## 项目概述

这是一个 agent sdk 项目， 用于以后各种agent开发的基础设施。 这是一个非常重要的项目，必须有非常优秀的的 上下文工程 实践的沉淀。 请仔细思考每一个细节。


## global rule

1. 如果你遇到不清楚的问题或者缺少什么信息，MUST及时向我询问。  
2. 先给出你的方案，MUST经我审阅后再编写代码。
3. 给出几版方案选择，构建方案时使用问答的方式逐步确认我的需求。 this is MUST
4. 不要为我生成任何总结文档, 除非我主动告诉你.
5. 当用户让你排查或者修复问题时，我希望你站在全局角度工程化考虑问题,结合当前架构,分析问题原因,给出解决方案。而不是简单止血。需要确认修改这个会不会造成其他依赖这个的功能的连锁反应和冲突. this is MUST
6. make code simple, readable, maintainable, and extensible
7. 涉及到Textual 这个TUI框架, 你必须使用 context7 mcp里面的  query-docs和resolve-library-id 功能来查询相关文档, 以确保你对这个框架的理解是正确的. this is MUST

## python coding rule
1. 代码必须使用 f-string 进行字符串格式化
2. 代码必须使用 logging 模块进行日志记录，禁止使用 print 语句
3. 运行 python必须使用 uv run python 脚本名.py 的方式运行
4. 安装pip包必须使用 uv add 包名 的方式安装
