"""TUI logging adapter - 将日志输出到 prompt_toolkit TUI 而不破坏界面"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prompt_toolkit.application import run_in_terminal

if TYPE_CHECKING:
    from terminal_agent.event_renderer import EventRenderer


class TUILoggingHandler(logging.Handler):
    """自定义 logging handler，将日志友好地显示在 TUI 中

    特性：
    - WARNING/ERROR 显示给用户，带颜色标识
    - 首次显示机制：相同消息只显示一次
    - 使用 run_in_terminal() 不破坏 prompt_toolkit 界面
    """

    def __init__(self, renderer: EventRenderer) -> None:
        super().__init__()
        self._renderer = renderer
        self._shown_messages: set[str] = set()  # 跟踪已显示的消息（首次显示）

    def emit(self, record: logging.LogRecord) -> None:
        """处理日志记录"""
        try:
            # 只处理 WARNING 和 ERROR
            if record.levelno < logging.WARNING:
                return

            # 生成友好消息
            msg = self._format_message(record)
            if not msg:
                return

            # 首次显示检查
            msg_key = self._get_message_key(record)
            if msg_key in self._shown_messages:
                return
            self._shown_messages.add(msg_key)

            # 通过 run_in_terminal 输出（不破坏 TUI）
            is_error = record.levelno >= logging.ERROR
            self._append_to_tui(msg, is_error=is_error)

        except Exception:
            self.handleError(record)

    def _format_message(self, record: logging.LogRecord) -> str:
        """格式化日志消息为用户友好格式"""
        msg = record.getMessage()

        # 特殊处理：token count fallback
        if "token count failed" in msg.lower() and "fallback" in msg.lower():
            return "⚠️ Token estimation using fallback (working normally)"

        # ERROR 级别
        if record.levelno >= logging.ERROR:
            return f"❌ {self._simplify_message(msg)}"

        # WARNING 级别
        return f"⚠️ {self._simplify_message(msg)}"

    def _simplify_message(self, msg: str) -> str:
        """简化技术细节"""
        # 去掉常见的技术参数
        if "exc_type=" in msg:
            # 截断到第一个技术参数之前
            parts = msg.split("exc_type=")
            msg = parts[0].rstrip(": ,")

        if "timeout_ms=" in msg:
            parts = msg.split("timeout_ms=")
            msg = parts[0].rstrip(": ,")

        # 限制长度
        max_len = 100
        if len(msg) > max_len:
            msg = msg[:max_len] + "..."

        return msg

    def _get_message_key(self, record: logging.LogRecord) -> str:
        """生成消息的唯一键（用于首次显示检查）"""
        # 使用 logger 名称 + 消息内容的前 50 个字符
        msg = record.getMessage()
        # 对于 token count，使用固定 key
        if "token count failed" in msg.lower():
            return "token_count_fallback"
        # 其他消息用完整内容作为 key
        return f"{record.name}:{msg[:50]}"

    def _append_to_tui(self, msg: str, *, is_error: bool = False) -> None:
        """将消息追加到 TUI（通过 run_in_terminal 避免破坏界面）"""
        def _append() -> None:
            self._renderer.append_system_message(msg, is_error=is_error)

        # 使用 run_in_terminal 确保不破坏 prompt_toolkit 的输入
        # 如果没有运行中的真实 Application，直接调用（测试或初始化阶段）
        try:
            from prompt_toolkit.application import get_app
            from prompt_toolkit.application.dummy import DummyApplication
            app = get_app()
            if isinstance(app, DummyApplication):
                # DummyApplication，直接调用
                _append()
            else:
                # 真实 app，使用 run_in_terminal
                run_in_terminal(_append, in_executor=False)
        except Exception:
            # 没有 app 或导入失败，直接调用
            _append()


def setup_tui_logging(renderer: EventRenderer) -> None:
    """统一日志初始化：文件 + TUI 双通道。

    - 所有日志（含 traceback）写入 ~/.comate/logs/agent.log（RotatingFileHandler）
    - WARNING/ERROR 以用户友好格式显示在 TUI scrollback
    - 清除 root logger 默认的 stderr handler，阻止 traceback 泄漏到终端
    """
    import os
    from logging.handlers import RotatingFileHandler

    # 1. 日志文件 handler（完整调试信息，含 traceback）
    log_dir = os.path.join(os.path.expanduser("~"), ".comate", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "agent.log")
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    # 2. 清除 root logger 的所有 handler，阻止 stderr 泄漏
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.setLevel(logging.DEBUG)

    # 3. TUI handler 挂到 root（覆盖所有命名空间的 WARNING/ERROR）
    tui_handler = TUILoggingHandler(renderer)
    tui_handler.setLevel(logging.WARNING)
    root.addHandler(tui_handler)
