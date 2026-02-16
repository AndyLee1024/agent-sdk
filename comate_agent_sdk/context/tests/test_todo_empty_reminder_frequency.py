"""测试空 todo 列表提醒的频率控制（适配 NudgeState 系统）"""

from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.llm.messages import UserMessage


def test_reminder_at_each_turn():
    """模拟真实的对话流程，测试每一轮的提醒情况"""
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="Hello"))

    print("=== 模拟真实对话流程 ===\n")

    # 初始化空 todo
    ctx.set_todo_state([])
    ctx._nudge.todo_last_changed_turn = 1  # 从 turn 1 开始计算 gap

    print(f"初始化后:")
    print(f"  - 当前轮次: {ctx._turn_number}")
    print(f"  - todo_last_changed_turn: {ctx._nudge.todo_last_changed_turn}")

    # 测试轮次 1-17
    print("\n各轮次提醒情况:")
    for turn in range(1, 18):
        ctx.set_turn_number(turn)
        ctx._nudge.turn = turn

        # 调用 lower() 生成提醒
        msgs = ctx.lower()
        last_user_msg = next((m for m in reversed(msgs) if isinstance(m, UserMessage)), None)

        if last_user_msg:
            has_reminder = "currently empty" in last_user_msg.text
            gap = turn - ctx._nudge.todo_last_changed_turn
            status = "✓" if has_reminder else "✗"
            print(f"轮次 {turn:2d}: {status} (gap={gap}, {gap}%8={gap%8})")

    # 验证预期
    print("\n=== 预期行为 ===")
    print("第一次提醒：轮次 1 (gap=0, 0%8=0) ✓")
    print("第二次提醒：轮次 9 (gap=8, 8%8=0) ✓")
    print("第三次提醒：轮次 17 (gap=16, 16%8=0) ✓")


def test_state_change_scenario():
    """测试完整的状态变化场景"""
    ctx = ContextIR()
    ctx.add_message(UserMessage(content="Hello"))

    print("\n=== 测试状态变化场景 ===\n")

    scenarios = [
        (1, "empty", "初始化空 todo"),
        (2, "empty", "第 2 轮，不提醒"),
        (8, "empty", "第 8 轮，不提醒"),
        (9, "empty", "第 9 轮，应该提醒 (gap=8)"),
        (10, "add", "第 10 轮，添加 todo"),
        (15, "clear", "第 15 轮，清空 todo，应立即提醒 (gap=0)"),
        (16, "empty", "第 16 轮，不提醒"),
        (23, "empty", "第 23 轮 (15+8)，应该提醒"),
    ]

    # 初始化
    ctx.set_todo_state([])
    ctx._nudge.todo_last_changed_turn = 1

    for turn, action, description in scenarios:
        ctx.set_turn_number(turn)
        ctx._nudge.turn = turn

        if action == "add":
            ctx.set_todo_state([{"content": "任务1", "status": "pending"}])
        elif action == "clear":
            ctx.set_todo_state([])

        # 检查提醒
        msgs = ctx.lower()
        last_user_msg = next((m for m in reversed(msgs) if isinstance(m, UserMessage)), None)

        if last_user_msg:
            has_empty = "currently empty" in last_user_msg.text
            has_gentle = "active TODO items" in last_user_msg.text

            print(f"轮次 {turn:2d}: {description}")
            if has_empty:
                gap = turn - ctx._nudge.todo_last_changed_turn
                print(f"         空提醒: ✓ 显示 (gap={gap})")
            elif has_gentle:
                print(f"         温和提醒已注入（有 todo）")
            else:
                print(f"         无提醒")


def test_modulo_logic():
    """直接测试取模逻辑"""
    print("\n=== 测试取模逻辑 ===\n")

    start = 1
    print(f"起始轮次: {start}")
    print("应该提醒的轮次（gap % 8 == 0）:")

    for turn in range(1, 26):
        gap = turn - start
        should_remind = gap % 8 == 0
        if should_remind:
            print(f"  轮次 {turn}: gap={gap}, {gap}%8={gap%8} ✓")


if __name__ == "__main__":
    test_reminder_at_each_turn()
    test_state_change_scenario()
    test_modulo_logic()
    print("\n=== 所有测试完成 ===")
