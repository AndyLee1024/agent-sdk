"""
Example demonstrating dependency injection with Depends.

This shows how to inject shared resources (like database connections,
API clients, or configuration) into your tools using type-safe annotations.

Run with:
    python -m bu_agent_sdk.examples.dependency_injection
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Annotated

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import (
    ComateAgentOptions,
    StopEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import Depends, tool


logger = logging.getLogger(__name__)


# Simulated database
@dataclass
class Database:
    """Simulated database with some user data."""

    users: dict[int, dict]

    def get_user(self, user_id: int) -> dict | None:
        return self.users.get(user_id)

    def list_users(self) -> list[dict]:
        return list(self.users.values())


# Dependency provider
def get_database() -> Database:
    """Provide the database instance."""
    return Database(
        users={
            1: {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "role": "admin",
            },
            2: {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
            3: {
                "id": 3,
                "name": "Charlie",
                "email": "charlie@example.com",
                "role": "user",
            },
        }
    )


@tool("Get a user by their ID")
async def get_user(user_id: int, db: Annotated[Database, Depends(get_database)]) -> str:
    """Fetch a user from the database by ID."""
    user = db.get_user(user_id)
    if user:
        return f"User found: {user['name']} ({user['email']}) - Role: {user['role']}"
    return f"No user found with ID {user_id}"


@tool("List all users in the system")
async def list_users(db: Annotated[Database, Depends(get_database)]) -> str:
    """List all users in the database."""
    users = db.list_users()
    lines = [f"- {u['name']} (ID: {u['id']}, Role: {u['role']})" for u in users]
    return "Users:\n" + "\n".join(lines)


@tool("Count users with a specific role")
async def count_by_role(
    role: str, db: Annotated[Database, Depends(get_database)]
) -> str:
    """Count how many users have a given role."""
    users = db.list_users()
    count = sum(1 for u in users if u["role"] == role)
    return f"Found {count} user(s) with role '{role}'"


async def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = Agent(
        llm=ChatOpenAI(model="gpt-5.2"),
        options=ComateAgentOptions(
            tools=[get_user, list_users, count_by_role],
            system_prompt="You are a helpful assistant that can query user information from a database.",
        ),
    )

    # Ask about users with streaming
    async for event in agent.query_stream(
        "List all users and tell me how many are admins"
    ):
        match event:
            case ToolCallEvent(tool=name, args=args):
                logger.info(f"🔧 {name}({args})")
            case ToolResultEvent(tool=name, result=result):
                logger.info(f"   → {result}")
            case TextEvent(content=text):
                logger.info(f"\n✅ {text}")
            case StopEvent(reason=_reason):
                pass

    usage = await agent.get_usage()
    logger.info(f"Token usage: {usage.total_cost}")


if __name__ == "__main__":
    asyncio.run(main())
