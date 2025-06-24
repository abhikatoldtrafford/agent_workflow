from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional, List
import logging

@dataclass
class AppContext:
    store: List[str]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Manage application lifecycle with type-safe context.
    
    This function initializes the application resources and cleans them up when the app shuts down.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        AppContext: The application context containing stores and sessions
    """
    # Initialize on startup
    db = []  # General-purpose list store

    try:
        yield AppContext(store=db)
    finally:
        # Cleanup on shutdown could happen here
        pass

mcp = FastMCP("local_memory_mcp", lifespan=app_lifespan)
logger = logging.getLogger("local_memory_mcp:server")
# Add tools to the MCP
@mcp.tool(description="Add a memory to the memory MCP for future replay")
def remember(
    value: str,
) -> str:
    """
    Add a memory to the local memory MCP.

    Args:
        value: The value for the memory

    Returns:
        A message indicating success
    """
    ctx = mcp.get_context()
    store = ctx.request_context.lifespan_context.store

    logger.info(f"inside remember - value: {value}, store length = {len(store)}",)

    store.append(value)
    return f"Memory added with value: {value}"

@mcp.tool(description="Get multiple memories, returns all the memories if count is not provided, else return count from the top")
def recall(
    count: Optional[int],
) -> List[str]:
    """
    Get all memories from the local memory MCP.

    Args:
        count: Optional number of memories to return from the top

    Returns:
        Dictionary containing memories
    """
    ctx = mcp.get_context()
    store = ctx.request_context.lifespan_context.store

    logger.info(f"inside recall - count: {count}, store length = {len(store)}",)

    if count is None:
        selected = store
    elif count > 0:
        # adjust slicing to match your definition of "top"
        selected = store[-count:]
    else:
        # explicit behavior for count <= 0
        selected = []

    logger.info(f"inside recall - return count - {len(selected)}",)
    return selected

