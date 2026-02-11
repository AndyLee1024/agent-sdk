"""上下文视图格式化器

将 ContextInfo 格式化为可视化的字符串输出。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.context.info import ContextInfo


def format_context_view(info: ContextInfo) -> str:
    """格式化上下文信息为可视化字符串

    输出示例：
    ```
      Context Usage
      ⛁ ⛁ ⛁ ⛁ ⛁ ⛁ ⛁ ⛁ ⛀ ⛀   claude-opus-4-5 · 24k/200k tokens (12%)
      ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶
      ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶   Estimated usage by category
      ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶   ⛁ System Prompt: 3.0k tokens (1.5%)
      ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶   ⛁ Tool Definitions: 17.1k tokens (8.6%)
      ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶   ⛁ Memory: 3.6k tokens (1.8%)
      ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶ ⛶   ⛁ Messages: 8 tokens (0.0%)
      ⛶ ⛶ ⛶ ⛝ ⛝ ⛝ ⛝ ⛝ ⛝ ⛝   ⛶ Free space: 131k (65.6%)
      ⛝ ⛝ ⛝ ⛝ ⛝ ⛝ ⛝ ⛝ ⛝ ⛝   ⛝ Compaction buffer: 45.0k tokens (22.5%)
    ```

    Args:
        info: 上下文信息

    Returns:
        格式化后的字符串
    """
    lines = []

    # 标题
    lines.append("  Context Usage")

    # 构建网格
    grid_lines = _build_grid(info)
    lines.extend(grid_lines)

    # 压缩状态提示（如果禁用）
    if not info.compaction_enabled:
        lines.append("  [compaction disabled]")

    return "\n".join(lines)


def _build_grid(info: ContextInfo) -> list[str]:
    """构建上下文使用情况的网格可视化

    网格规则：
    - 10 列固定宽度
    - 自动计算行数以适应 context_limit
    - 符号：⛁ 已用内容 | ⛀ 工具定义 | ⛶ 可用空间 | ⛝ 压缩缓冲区

    Args:
        info: 上下文信息

    Returns:
        网格行列表（每行包含网格符号和可选的右侧标注）
    """
    COLS = 10

    # 计算合适的行数（确保每格代表的 token 数合理）
    # 目标：每格约 2k-5k tokens
    target_tokens_per_cell = 2000
    total_cells_needed = max(1, info.context_limit // target_tokens_per_cell)
    rows = max(1, (total_cells_needed + COLS - 1) // COLS)

    tokens_per_cell = info.context_limit / (COLS * rows)

    # 计算各部分占用的 cell 数（主口径：next-step 估算）
    used_cells = int(info.primary_used_tokens / tokens_per_cell)
    free_cells = int(info.free_tokens / tokens_per_cell)
    buffer_cells = int(info.buffer_tokens / tokens_per_cell)

    # 确保总数不超过网格容量
    total_cells = COLS * rows
    assigned_cells = used_cells + free_cells + buffer_cells
    if assigned_cells > total_cells:
        # 按比例缩减
        scale = total_cells / assigned_cells
        used_cells = int(used_cells * scale)
        free_cells = int(free_cells * scale)
        buffer_cells = total_cells - used_cells - free_cells

    # 构建 cell 序列
    cells = []
    cells.extend(['⛁'] * used_cells)
    cells.extend(['⛶'] * free_cells)
    cells.extend(['⛝'] * buffer_cells)

    # 填充到总容量
    while len(cells) < total_cells:
        cells.append('⛝')

    # 构建行
    lines = []

    # 第一行：网格 + 模型信息
    first_row_cells = " ".join(cells[:COLS])
    model_short = _shorten_model_name(info.model_name)
    lines.append(
        f"  {first_row_cells}   {model_short} · "
        f"next-step {_format_token_count(info.primary_used_tokens)}/"
        f"{_format_token_count(info.context_limit)} tokens ({info.utilization_percent:.1f}%)"
    )

    # 后续行：网格 + 类别明细
    category_labels = _build_category_labels(info)

    for i in range(1, rows):
        row_start = i * COLS
        row_end = row_start + COLS
        row_cells = " ".join(cells[row_start:row_end])

        # 获取对应的类别标注（如果有）
        label = category_labels[i - 1] if i - 1 < len(category_labels) else ""

        if label:
            lines.append(f"  {row_cells}   {label}")
        else:
            lines.append(f"  {row_cells}")

    return lines


def _build_category_labels(info: ContextInfo) -> list[str]:
    """构建类别明细标签列表

    Args:
        info: 上下文信息

    Returns:
        标签列表（用于网格右侧并排显示）
    """
    labels = []

    # 第一行：类别标题
    labels.append("Estimated usage by category (IR)")

    next_step_percent = (info.next_step_estimated_tokens / info.context_limit * 100) if info.context_limit > 0 else 0
    labels.append(
        f"⛁ Next-step estimate: {_format_token_count(info.next_step_estimated_tokens)} "
        f"tokens ({next_step_percent:.1f}%)"
    )
    if info.last_step_reported_tokens > 0:
        reported_percent = (info.last_step_reported_tokens / info.context_limit * 100) if info.context_limit > 0 else 0
        labels.append(
            f"⛂ Last-step reported: {_format_token_count(info.last_step_reported_tokens)} "
            f"tokens ({reported_percent:.1f}%)"
        )

    msg_only_percent = (info.used_tokens_message_only / info.context_limit * 100) if info.context_limit > 0 else 0
    labels.append(
        f"⛃ Message-only used: {_format_token_count(info.used_tokens_message_only)} tokens ({msg_only_percent:.1f}%)"
    )
    with_tools_percent = (info.used_tokens_with_tools / info.context_limit * 100) if info.context_limit > 0 else 0
    labels.append(
        f"⛀ Message+tools used: {_format_token_count(info.used_tokens_with_tools)} tokens ({with_tools_percent:.1f}%)"
    )

    # 各类别明细
    for cat in info.categories:
        percent = (cat.token_count / info.context_limit * 100) if info.context_limit > 0 else 0
        labels.append(
            f"⛁ {cat.label}: {_format_token_count(cat.token_count)} tokens ({percent:.1f}%)"
        )

    # 可用空间
    free_percent = (info.free_tokens / info.context_limit * 100) if info.context_limit > 0 else 0
    labels.append(f"⛶ Free space: {_format_token_count(info.free_tokens)} ({free_percent:.1f}%)")

    # 压缩缓冲区
    buffer_percent = (info.buffer_tokens / info.context_limit * 100) if info.context_limit > 0 else 0
    labels.append(f"⛝ Compaction buffer: {_format_token_count(info.buffer_tokens)} tokens ({buffer_percent:.1f}%)")

    return labels


def _format_token_count(count: int) -> str:
    """格式化 token 数为易读形式（如 3.0k, 17.1k, 200k）

    Args:
        count: token 数

    Returns:
        格式化后的字符串
    """
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def _shorten_model_name(model: str) -> str:
    """缩短模型名称为简洁形式

    示例：
    - claude-opus-4-5-20251101 -> claude-opus-4-5
    - gpt-4o-2024-05-13 -> gpt-4o

    Args:
        model: 完整模型名称

    Returns:
        简化后的模型名称
    """
    # 移除日期后缀（YYYYMMDD 或 YYYY-MM-DD）
    parts = model.split('-')
    filtered = []
    for part in parts:
        # 跳过纯数字且长度为 8 的部分（YYYYMMDD）
        if part.isdigit() and len(part) == 8:
            break
        # 跳过看起来像日期的部分
        if part.isdigit() and len(part) == 4 and int(part) >= 2020:
            break
        filtered.append(part)

    return '-'.join(filtered) if filtered else model
