# Context FileSystem 实现完成

## 实现概述

已成功实现 Context FileSystem 上下文卸载功能，允许将大型上下文内容持久化到文件系统，避免占用过多 token，同时保持上下文结构完整。

## 完成的步骤

### ✅ Step 1: 数据结构扩展
- **文件**: `bu_agent_sdk/context/items.py`
- **修改**: ContextItem 新增 `offload_path` 和 `offloaded` 字段

### ✅ Step 2: 核心模块
- **文件**: `bu_agent_sdk/context/offload.py`
  - 实现 `OffloadPolicy` 配置类
- **文件**: `bu_agent_sdk/context/fs.py`
  - 实现 `ContextFileSystem` 核心类
  - 实现 `OffloadedMeta` 元数据类
  - 实现文件系统操作（offload, load, get_placeholder）
  - 实现索引管理

### ✅ Step 3: Compaction 集成
- **文件**: `bu_agent_sdk/context/compaction.py`
- **修改**:
  - SelectiveCompactionPolicy 新增 `fs` 和 `offload_policy` 字段
  - 实现 `_should_offload()` 判断方法
  - 在 TRUNCATE 时调用 offload
  - 跳过已 destroyed 的 items（避免与 ephemeral 重复）

### ✅ Step 4: Ephemeral 协作
- **文件**: `bu_agent_sdk/agent/service.py`
- **修改**:
  - 移除 `_destroy_ephemeral_messages` 中的直接文件写入逻辑
  - 替换为调用 ContextFileSystem.offload()
  - 同步更新 message.offloaded 和 offload_path

### ✅ Step 5: Serializer 修改
- **文件**:
  - `bu_agent_sdk/llm/messages.py`: ToolMessage 新增字段
  - `bu_agent_sdk/llm/anthropic/serializer.py`
  - `bu_agent_sdk/llm/openai/serializer.py`
  - `bu_agent_sdk/llm/google/serializer.py`
- **修改**: destroyed 时检查 offload_path，显示完整路径

### ✅ Step 6: AgentService 集成
- **文件**: `bu_agent_sdk/agent/service.py`
- **修改**:
  - 新增配置字段：`offload_enabled`, `offload_token_threshold`, `offload_root_path`, `ephemeral_keep_recent`
  - 新增内部字段：`_context_fs`, `_session_id`
  - 生成 session_id（timestamp + random）
  - 初始化 ContextFileSystem
  - 在 `_check_and_compact` 中创建 OffloadPolicy 并传递给压缩策略

### ✅ Step 7: 导出新组件
- **文件**: `bu_agent_sdk/context/__init__.py`
- **修改**: 导出 `ContextFileSystem`, `OffloadedMeta`, `OffloadPolicy`

## 测试结果

### 基本功能测试 ✅
- ContextFileSystem 创建和初始化
- 条目卸载（offload）
- 内容加载（load）
- 占位符生成（get_placeholder）
- 索引管理

### 卸载策略测试 ✅
- 类型启用/禁用检查
- Token 阈值判断
- COMPACTION_SUMMARY 不卸载

### Ephemeral + Offload 集成测试 ✅
- Ephemeral 机制触发卸载
- 文件系统存储验证
- ContextItem 状态标记（destroyed, offloaded）
- Message 对象同步更新

## 关键设计决策

### 1. 分工明确
- **Ephemeral 机制**: 决定何时销毁（基于 keep_recent）
- **Context FileSystem**: 负责持久化存储

### 2. 避免重复处理
- Compaction 时跳过已 destroyed 的 items
- 防止 ephemeral 和 compaction 重复卸载同一条目

### 3. 消息结构完整性
- 保留 destroyed 条目在列表中
- Serializer 生成占位符，保持 API 兼容性

### 4. 用户可配置
- Session ID 自动生成（timestamp + random）
- 存储路径可自定义（默认 `~/.agent/context/{session_id}`）
- Token 阈值可调整（默认 2000）
- 类型级别的启用/禁用控制

## 文件系统结构

```
~/.agent/context/{session_id}/
├── index.json              # 索引文件（记录所有卸载条目）
├── tool_result/
│   ├── Read/
│   │   └── {item_id}.json
│   └── Bash/
│       └── {item_id}.json
├── assistant_message/
│   └── {item_id}.json
└── user_message/
    └── {item_id}.json
```

## 占位符示例

```
[Content offloaded]
Path: /home/user/.agent/context/session_20240115_abc/tool_result/Read/abc123.json
Use Read tool to view details.
```

## 配置示例

```python
agent = Agent(
    llm=ChatAnthropic(),
    tools=[...],
    offload_enabled=True,              # 启用卸载
    offload_token_threshold=2000,      # Token 阈值
    offload_root_path=None,            # 使用默认路径
    ephemeral_keep_recent=None,        # 使用工具的默认值
)
```

## 后续改进建议

1. **自动清理**：定期清理过期 session 文件
2. **压缩存储**：使用 gzip 压缩卸载文件
3. **摘要生成**：卸载时生成简短摘要嵌入占位符
4. **恢复机制**：支持从卸载文件恢复完整上下文
5. **统计监控**：记录卸载统计信息（节省 token 数等）

## 文档

详细文档已创建：`docs/context_filesystem.md`

## 总结

✅ 所有计划步骤已完成
✅ 所有测试通过
✅ 与现有功能（Ephemeral, Compaction）协作良好
✅ 代码结构清晰，易于扩展

实现完全符合设计文档要求！
