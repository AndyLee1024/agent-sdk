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
    from comate_agent_sdk.agent.core import AgentRuntime, AgentTemplate


def _setup_env_info(runtime: "AgentRuntime") -> None:
    opts = runtime.env_options
    if opts is None:
        return
    if not opts.system_env and not opts.git_env:
        return

    from comate_agent_sdk.context.env import EnvProvider

    working_dir = opts.working_dir or runtime.project_root or Path.cwd()
    provider = EnvProvider(
        git_status_limit=opts.git_status_limit,
        git_log_limit=opts.git_log_limit,
    )

    if opts.system_env:
        runtime._context.set_system_env(provider.get_system_env(working_dir))
        logger.info(f"已注入 SYSTEM_ENV（working_dir={working_dir}）")

    if opts.git_env:
        git_env = provider.get_git_env(working_dir)
        if git_env:
            runtime._context.set_git_env(git_env)
            logger.info(f"已注入 GIT_ENV（working_dir={working_dir}）")
        else:
            logger.info(f"未注入 GIT_ENV：working_dir={working_dir} 不在 git 仓库中或无法读取")


def build_template(template: "AgentTemplate") -> None:
    """模板阶段初始化：只做只读解析与发现。"""
    from comate_agent_sdk.agent.settings import discover_agents_md, discover_user_agents_md, resolve_settings
    from comate_agent_sdk.agent.llm_levels import resolve_llm_levels

    cfg = template.config

    # ===== Subagent 自动发现和 Merge（模板期一次性） =====
    # 语义：
    # - agents is None：允许自动发现
    # - agents == ()：显式禁用自动发现
    # - agents 非空 tuple：自动发现并 merge（同名以显式配置优先）
    resolved_agents: tuple | None
    if cfg.agents == ():
        resolved_agents = ()
    else:
        from comate_agent_sdk.subagent import discover_subagents

        discovered = discover_subagents(project_root=cfg.project_root)
        user_agents = list(cfg.agents or ())

        if discovered or user_agents:
            user_names = {a.name for a in user_agents}
            merged = [a for a in discovered if a.name not in user_names]
            merged.extend(user_agents)
            resolved_agents = tuple(merged)
            if discovered:
                logger.info(f"Auto-discovered {len(discovered)} subagent(s)")
        else:
            resolved_agents = None

    # ===== Skill 自动发现和 Merge（模板期一次性） =====
    from comate_agent_sdk.skill import discover_skills

    discovered_skills = discover_skills(project_root=cfg.project_root)
    user_skills = list(cfg.skills or ())
    if discovered_skills or user_skills:
        user_skill_names = {s.name for s in user_skills}
        merged_skills = [s for s in discovered_skills if s.name not in user_skill_names]
        merged_skills.extend(user_skills)
        resolved_skills: tuple | None = tuple(merged_skills)
        if discovered_skills:
            logger.info(f"Auto-discovered {len(discovered_skills)} skill(s)")
    else:
        resolved_skills = None

    # ===== Settings + AGENTS.md memory 注入（模板期一次性） =====
    settings = resolve_settings(
        sources=cfg.setting_sources,
        project_root=cfg.project_root,
    )

    resolved_memory = cfg.memory
    if cfg.setting_sources and resolved_memory is None:
        agents_md_files: list[Path] = []
        source_used: str | None = None

        if "project" in cfg.setting_sources:
            agents_md_files = discover_agents_md(cfg.project_root)
            if agents_md_files:
                source_used = "project"

        if not agents_md_files and "user" in cfg.setting_sources:
            agents_md_files = discover_user_agents_md()
            if agents_md_files:
                source_used = "user"

        if agents_md_files and source_used:
            from comate_agent_sdk.context.memory import MemoryConfig

            resolved_memory = MemoryConfig(files=agents_md_files)
            logger.info(f"自动发现 {source_used} AGENTS.md，注入 memory: {agents_md_files}")

    # ===== LLM levels + 主 LLM 解析（模板期一次性） =====
    resolved_llm_levels = resolve_llm_levels(explicit=dict(cfg.llm_levels) if cfg.llm_levels else None, settings=settings)

    resolved_llm = template.llm
    if resolved_llm is None:
        effective_level: LLMLevel = template.level or "MID"
        if effective_level not in resolved_llm_levels:
            raise ValueError(f"level='{effective_level}' 不在 llm_levels 中")
        resolved_llm = resolved_llm_levels[effective_level]
    elif template.level is not None:
        logger.warning(f"同时指定了 llm= 和 level='{template.level}'，llm= 优先")

    object.__setattr__(template, "_resolved_agents", resolved_agents)
    object.__setattr__(template, "_resolved_skills", resolved_skills)
    object.__setattr__(template, "_resolved_memory", resolved_memory)
    object.__setattr__(template, "_resolved_settings", settings)
    object.__setattr__(template, "_resolved_llm_levels", resolved_llm_levels)
    object.__setattr__(template, "_resolved_llm", resolved_llm)


def init_runtime_from_template(runtime: "AgentRuntime") -> None:
    """运行态初始化：只初始化可变状态，不做模板发现。"""
    from comate_agent_sdk.tools import ToolRegistry, get_default_registry

    # ====== 自动推断 tools 和 tool_registry ======
    builtin = get_default_registry()

    if runtime.tool_registry is None:
        local_registry = ToolRegistry()
        for t in builtin.all():
            if t.name not in local_registry:
                local_registry.register(t)
        runtime.tool_registry = local_registry
        logger.debug(f"Initialized runtime-local registry with {len(local_registry)} built-in tool(s)")

    if runtime.tools is None:
        runtime._tools_allowlist_mode = False
        runtime.tools = runtime.tool_registry.all()
        logger.debug(f"Using all {len(runtime.tools)} tool(s) from registry")
    else:
        runtime._tools_allowlist_mode = True

        resolved: list[Tool] = []
        pending_mcp: list[str] = []
        requested_names: list[str] = []

        for item in runtime.tools:
            if isinstance(item, Tool):
                if item.name not in runtime.tool_registry:  # type: ignore[operator]
                    try:
                        runtime.tool_registry.register(item)  # type: ignore[union-attr]
                    except Exception:
                        logger.debug(f"注册工具失败（忽略覆盖/重复）：{item.name}", exc_info=True)

        for item in runtime.tools:
            if isinstance(item, Tool):
                resolved.append(item)
                continue

            if isinstance(item, str):
                name = item
                requested_names.append(name)

                if name in runtime.tool_registry:  # type: ignore[operator]
                    resolved.append(runtime.tool_registry.get(name))  # type: ignore[union-attr]
                    continue

                if name.startswith("mcp__"):
                    pending_mcp.append(name)
                    continue

                raise ValueError(f"Unknown tool name: {name}")

            raise TypeError(f"tools 仅支持 Tool 或 str，收到：{type(item).__name__}")

        runtime._mcp_pending_tool_names = pending_mcp
        runtime._requested_tool_names = requested_names
        runtime.tools = resolved

    # ====== Tool 名冲突检查 ======
    if runtime.agents != []:
        user_task_tools = [
            t
            for t in runtime.tools
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

    if runtime._is_subagent and runtime.agents:
        raise ValueError("Subagent 不能再定义 agents（不支持嵌套）")

    for t in runtime.tools:
        assert isinstance(t, Tool), (
            f"Expected Tool instance, got {type(t).__name__}. Did you forget to use the @tool decorator?"
        )

    runtime._tool_map = {t.name: t for t in runtime.tools}

    runtime._session_id = runtime.session_id or str(uuid.uuid4())

    if runtime.llm is None:
        effective_level: LLMLevel = runtime.level or "MID"
        if runtime.llm_levels is None or effective_level not in runtime.llm_levels:
            raise ValueError(f"level='{effective_level}' 不在 llm_levels 中")
        runtime.llm = runtime.llm_levels[effective_level]

    if runtime._parent_token_cost is not None:
        runtime._token_cost = runtime._parent_token_cost
    else:
        runtime._token_cost = TokenCost(include_cost=runtime.include_cost)

    if runtime.offload_enabled:
        if runtime.offload_root_path:
            root_path = Path(runtime.offload_root_path).expanduser()
        else:
            root_path = Path.home() / ".agent" / "sessions" / runtime._session_id / "offload"
        runtime._context_fs = ContextFileSystem(
            root_path=root_path,
            session_id=runtime._session_id,
        )
        logger.info(f"Context FileSystem enabled at {root_path}")

        if runtime.offload_policy is None:
            runtime.offload_policy = OffloadPolicy(
                enabled=True,
                token_threshold=runtime.offload_token_threshold,
            )

    runtime._context = ContextIR()

    compaction_config = runtime.compaction if runtime.compaction is not None else CompactionConfig()
    runtime._compaction_service = CompactionService(
        config=compaction_config,
        llm=runtime.llm,
        token_cost=runtime._token_cost,
    )

    runtime._setup_tool_strategy()
    runtime._setup_agent_loop()

    if runtime.agents:
        runtime._setup_subagents()

    runtime._setup_skills()

    resolved_prompt = runtime._resolve_system_prompt()
    if resolved_prompt:
        runtime._context.set_system_prompt(resolved_prompt, cache=True)

    if runtime.memory:
        runtime._setup_memory()

    _setup_env_info(runtime)


# 兼容旧调用路径
agent_post_init = init_runtime_from_template
