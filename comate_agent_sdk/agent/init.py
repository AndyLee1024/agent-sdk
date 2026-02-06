from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.compaction import CompactionConfig, CompactionService
from comate_agent_sdk.agent.llm_levels import LLMLevel
from comate_agent_sdk.context import ContextIR
from comate_agent_sdk.context.fs import ContextFileSystem
from comate_agent_sdk.context.offload import OffloadPolicy
from comate_agent_sdk.tokens import TokenCost
from comate_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("comate_agent_sdk.agent")

if TYPE_CHECKING:
    from comate_agent_sdk.agent.core import Agent


def _setup_env_info(agent: "Agent") -> None:
    opts = agent.env_options
    if opts is None:
        return
    if not opts.system_env and not opts.git_env:
        return

    from comate_agent_sdk.context.env import EnvProvider

    working_dir = opts.working_dir or agent.project_root or Path.cwd()
    provider = EnvProvider(
        git_status_limit=opts.git_status_limit,
        git_log_limit=opts.git_log_limit,
    )

    if opts.system_env:
        agent._context.set_system_env(provider.get_system_env(working_dir))
        logger.info(f"已注入 SYSTEM_ENV（working_dir={working_dir}）")

    if opts.git_env:
        git_env = provider.get_git_env(working_dir)
        if git_env:
            agent._context.set_git_env(git_env)
            logger.info(f"已注入 GIT_ENV（working_dir={working_dir}）")
        else:
            logger.info(f"未注入 GIT_ENV：working_dir={working_dir} 不在 git 仓库中或无法读取")


def agent_post_init(agent: "Agent") -> None:
    # ====== 自动推断 tools 和 tool_registry ======
    # 设计目标：
    # - tools 支持 list[Tool | str]（str 为按名选择，支持 mcp__... 延迟解析）
    # - tool_registry 必须是“可变的、Agent 私有”的 registry（用于后续 MCP 动态注入，避免污染全局 built-in registry）
    from comate_agent_sdk.tools import ToolRegistry, get_default_registry

    builtin = get_default_registry()

    if agent.tool_registry is None:
        local_registry = ToolRegistry()
        # seed：拷贝 built-in tools（避免后续动态注入污染全局）
        for t in builtin.all():
            if t.name not in local_registry:
                local_registry.register(t)
        agent.tool_registry = local_registry
        logger.debug(f"Initialized agent-local registry with {len(local_registry)} built-in tool(s)")

    # 解析 tools：
    # - tools is None：默认使用 registry 中的所有工具（后续 MCP 会动态扩展）
    # - tools 为 list：允许 Tool 与 str 混合；str 按名解析，mcp__... 允许 pending
    if agent.tools is None:
        agent._tools_allowlist_mode = False
        agent.tools = agent.tool_registry.all()
        logger.debug(f"Using all {len(agent.tools)} tool(s) from registry")
    else:
        agent._tools_allowlist_mode = True

        resolved: list[Tool] = []
        pending_mcp: list[str] = []
        requested_names: list[str] = []

        # 先注册 Tool 实例，保证后续 str 可解析到这些工具
        for item in agent.tools:
            if isinstance(item, Tool):
                # 检查是否已经注册，避免重复注册警告
                if item.name not in agent.tool_registry:  # type: ignore[operator]
                    try:
                        agent.tool_registry.register(item)  # type: ignore[union-attr]
                    except Exception:
                        logger.debug(f"注册工具失败（忽略覆盖/重复）：{item.name}", exc_info=True)

        for item in agent.tools:
            if isinstance(item, Tool):
                resolved.append(item)
                continue

            if isinstance(item, str):
                name = item
                requested_names.append(name)

                if name in agent.tool_registry:  # type: ignore[operator]
                    resolved.append(agent.tool_registry.get(name))  # type: ignore[union-attr]
                    continue

                if name.startswith("mcp__"):
                    pending_mcp.append(name)
                    continue

                raise ValueError(f"Unknown tool name: {name}")

            raise TypeError(f"tools 仅支持 Tool 或 str，收到：{type(item).__name__}")

        agent._mcp_pending_tool_names = pending_mcp
        agent._requested_tool_names = requested_names
        agent.tools = resolved

    # ====== Tool 名冲突检查 ======
    # 约定：若未显式禁用 subagent（agents != []），则 'Task' 为保留名。
    # 即使当前未发现 subagent，也禁止用户静默提供同名工具，避免后续开启 subagent 时出现不可预期行为。
    if agent.agents != []:
        user_task_tools = [
            t
            for t in agent.tools
            if isinstance(t, Tool)
            and t.name == "Task"
            and getattr(t, "_comate_agent_sdk_internal", False) is not True
        ]
        if user_task_tools:
            raise ValueError(
                "检测到用户提供了同名工具 'Task'。"
                "'Task' 为 subagent 调度保留名（除非显式禁用 subagent：agents=[]）。"
                "解决方式：1) 将你的工具改名（不要叫 'Task'）；"
                "2) 显式禁用 subagent（例如 agents=[]）。"
            )

    # ====== Subagent 自动发现和 Merge ======
    # 只在非 subagent 时执行自动发现（subagent 不支持嵌套）
    if not agent._is_subagent:
        # 约定：
        # - agents is None：允许自动发现（纯自动 / 混合模式的一部分）
        # - agents 为非空 list：允许自动发现并 merge（同名时以代码传入为准）
        # - agents == []：显式禁用自动发现（用于测试隔离或完全手动模式）
        if agent.agents == []:
            logger.debug("Subagent auto-discovery disabled because agents=[]")
        else:
            from comate_agent_sdk.subagent import discover_subagents

            discovered = discover_subagents(project_root=agent.project_root)
            user_agents = agent.agents or []

            if discovered or user_agents:
                # Merge：代码传入的覆盖自动发现的同名 subagent
                user_agent_names = {a.name for a in user_agents}
                merged = [a for a in discovered if a.name not in user_agent_names]
                merged.extend(user_agents)
                agent.agents = merged if merged else None

                if discovered:
                    logger.info(f"Auto-discovered {len(discovered)} subagent(s)")

    # 检查嵌套 - Subagent 不能再定义 agents
    if agent._is_subagent and agent.agents:
        raise ValueError("Subagent 不能再定义 agents（不支持嵌套）")

    # Validate that all tools are Tool instances
    for t in agent.tools:
        assert isinstance(t, Tool), (
            f"Expected Tool instance, got {type(t).__name__}. Did you forget to use the @tool decorator?"
        )

    # Build tool lookup map
    agent._tool_map = {t.name: t for t in agent.tools}

    # Generate session_id (uuid by default)
    agent._session_id = agent.session_id or str(uuid.uuid4())

    # ====== Settings 配置加载 ======
    from comate_agent_sdk.agent.settings import discover_agents_md, discover_user_agents_md, resolve_settings

    settings = resolve_settings(
        sources=agent.setting_sources,
        project_root=agent.project_root,
    )

    # AGENTS.md 自动发现 → 注入 memory（用户未手动设置 memory 时）
    # 优先级：project > user（project 有则忽略 user）
    if agent.setting_sources and agent.memory is None:
        agents_md_files: list[Path] = []
        source_used: str | None = None

        # 尝试 project 级 AGENTS.md
        if "project" in agent.setting_sources:
            agents_md_files = discover_agents_md(agent.project_root)
            if agents_md_files:
                source_used = "project"

        # project 未找到时，fallback 到 user 级
        if not agents_md_files and "user" in agent.setting_sources:
            agents_md_files = discover_user_agents_md()
            if agents_md_files:
                source_used = "user"

        # 找到任何文件就注入 memory
        if agents_md_files and source_used:
            from comate_agent_sdk.context.memory import MemoryConfig

            agent.memory = MemoryConfig(files=agents_md_files)
            logger.info(f"自动发现 {source_used} AGENTS.md，注入 memory: {agents_md_files}")

    # Resolve LLM levels (LOW/MID/HIGH)
    from comate_agent_sdk.agent.llm_levels import resolve_llm_levels

    agent.llm_levels = resolve_llm_levels(explicit=agent.llm_levels, settings=settings)  # type: ignore[assignment]

    # Resolve main LLM:
    # llm= instance > level=档位 > 默认 MID
    if agent.llm is None:
        effective_level: LLMLevel = agent.level or "MID"
        if agent.llm_levels is None or effective_level not in agent.llm_levels:
            raise ValueError(f"level='{effective_level}' 不在 llm_levels 中")
        agent.llm = agent.llm_levels[effective_level]
    elif agent.level is not None:
        logger.warning(f"同时指定了 llm= 和 level='{agent.level}'，llm= 优先")

    # Initialize token cost service (Subagent 复用父级实例)
    if agent._parent_token_cost is not None:
        agent._token_cost = agent._parent_token_cost
    else:
        agent._token_cost = TokenCost(include_cost=agent.include_cost)

    # Initialize ContextFileSystem if offload enabled
    if agent.offload_enabled:
        if agent.offload_root_path:
            root_path = Path(agent.offload_root_path).expanduser()
        else:
            root_path = (
                Path.home() / ".agent" / "sessions" / agent._session_id / "offload"
            )
        agent._context_fs = ContextFileSystem(
            root_path=root_path,
            session_id=agent._session_id,
        )
        logger.info(f"Context FileSystem enabled at {root_path}")

        # Initialize offload policy (default: only tool_call/tool_result enabled)
        if agent.offload_policy is None:
            agent.offload_policy = OffloadPolicy(
                enabled=True,
                token_threshold=agent.offload_token_threshold,
            )

    # Initialize ContextIR
    agent._context = ContextIR()

    # Initialize compaction service (enabled by default)
    # Use provided config or create default (which has enabled=True)
    compaction_config = (
        agent.compaction if agent.compaction is not None else CompactionConfig()
    )
    agent._compaction_service = CompactionService(
        config=compaction_config,
        llm=agent.llm,
        token_cost=agent._token_cost,
    )

    # Initialize tool strategy (writes to IR header)
    agent._setup_tool_strategy()

    # Initialize agent loop control (writes to IR header)
    agent._setup_agent_loop()

    # Initialize subagent support (writes to IR header)
    if agent.agents:
        agent._setup_subagents()

    # Initialize skill support (writes to IR header)
    agent._setup_skills()

    # 解析 system_prompt 并写入 IR header
    resolved_prompt = agent._resolve_system_prompt()
    if resolved_prompt:
        agent._context.set_system_prompt(resolved_prompt, cache=True)

    # Initialize memory support
    if agent.memory:
        agent._setup_memory()

    # Initialize environment info (writes to IR header; initialization snapshot)
    _setup_env_info(agent)
