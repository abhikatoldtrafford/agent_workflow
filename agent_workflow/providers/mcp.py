import logging
from typing import Any, Dict, List, Optional

from agent_workflow.workflow_engine.models import MCPServerSpec

logger = logging.getLogger("workflow-engine.mcp")


class MCPServerRegistry:
    """Central registry for all available MCP servers."""

    def __init__(self) -> None:
        self._servers: Dict[str, MCPServerSpec] = {}

    def register_server(self, server: MCPServerSpec) -> None:
        """Register an MCP server with the registry.

        Args:
            server: The MCP server to register
        """
        self._servers[server.name] = server
        logger.info(f"Registered MCP server: {server.name}")

    def get_server(self, name: str) -> Optional[MCPServerSpec]:
        """Get a server by name.

        Args:
            name: The name of the server to get

        Returns:
            The server if found, None otherwise
        """
        return self._servers.get(name)

    def list_servers(self) -> List[str]:
        """List all registered server names.

        Returns:
            List of server names
        """
        return list(self._servers.keys())

    def get_all_servers(self) -> Dict[str, MCPServerSpec]:
        """Get all registered servers.

        Returns:
            Dictionary of server names to server objects
        """
        return self._servers.copy()

    def get_servers_schema(self) -> List[Dict[str, Any]]:
        """Get schema representations for all servers.

        Returns:
            List of server schemas
        """
        schemas = []
        for server in self._servers.values():
            schemas.append(
                {
                    "name": server.name,
                    "params": server.params,
                    "server_type": server.server_type,
                    "cache_tools_list": server.cache_tools_list,
                    "client_session_timeout": server.client_session_timeout,
                }
            )
        return schemas


# Create a global MCP server registry instance
mcp_registry = MCPServerRegistry()


# Convenience function to register an MCP server
def register_mcp_server(server: MCPServerSpec) -> None:
    """Register an MCP server with the global registry.

    Args:
        server: The MCP server to register
    """
    mcp_registry.register_server(server)
