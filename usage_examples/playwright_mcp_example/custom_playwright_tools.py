"""
Custom tools built on top of Playwright MCP.

This module demonstrates how to create higher-level tools
that use the Playwright MCP server under the hood.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from agent_workflow.providers.tools import Tool

logger = logging.getLogger(__name__)


class PlaywrightTool(Tool):
    """Base class for Playwright MCP-based tools."""
    
    def __init__(self, name: str, description: str, mcp_server: str = "playwright-mcp-server"):
        """Initialize a Playwright tool.
        
        Args:
            name: Tool name
            description: Tool description
            mcp_server: Name of the Playwright MCP server to use
        """
        super().__init__(name=name, description=description)
        self.mcp_server = mcp_server


class WebsiteScreenshotTool(PlaywrightTool):
    """Tool to take a screenshot of a website."""
    
    def __init__(self):
        """Initialize the website screenshot tool."""
        super().__init__(
            name="website_screenshot",
            description="Takes a screenshot of a specified website URL",
        )
    
    async def _execute(
        self, url: str, filename: Optional[str] = None, wait_seconds: int = 2
    ) -> Dict[str, str]:
        """Execute the website screenshot tool.
        
        Args:
            url: Website URL to screenshot
            filename: Optional filename for the screenshot
            wait_seconds: Seconds to wait after page load before taking screenshot
            
        Returns:
            Dictionary with screenshot path
        """
        # This is where we'd use the Playwright MCP tools
        # This is a simplified example showing the flow
        
        # First navigate to the URL
        await self.call_mcp_tool(
            self.mcp_server,
            "browser_navigate",
            {"url": url}
        )
        
        # Wait for the page to load
        await self.call_mcp_tool(
            self.mcp_server,
            "browser_wait_for",
            {"time": wait_seconds}
        )
        
        # Take the screenshot
        screenshot_result = await self.call_mcp_tool(
            self.mcp_server,
            "browser_take_screenshot",
            {"filename": filename} if filename else {}
        )
        
        return {"screenshot_path": screenshot_result.get("path", "Screenshot taken")}


class WebsiteFormFillTool(PlaywrightTool):
    """Tool to fill a form on a website."""
    
    def __init__(self):
        """Initialize the form fill tool."""
        super().__init__(
            name="website_form_fill",
            description="Fills a form on a website with provided field values",
        )
    
    async def _execute(
        self, 
        url: str, 
        form_fields: Dict[str, Union[str, List[str]]],
        submit: bool = True
    ) -> Dict[str, str]:
        """Execute the form fill tool.
        
        Args:
            url: Website URL with the form
            form_fields: Dictionary mapping field identifiers to values
            submit: Whether to submit the form after filling
            
        Returns:
            Dictionary with status
        """
        # Navigate to the URL
        await self.call_mcp_tool(
            self.mcp_server,
            "browser_navigate",
            {"url": url}
        )
        
        # Get the page snapshot to identify form elements
        snapshot_result = await self.call_mcp_tool(
            self.mcp_server,
            "browser_snapshot",
            {}
        )
        
        # In a real implementation, we would parse the snapshot
        # to find the form elements and their references
        
        # Fill each field
        for field_name, value in form_fields.items():
            # In a real implementation, we would find the correct element reference
            # from the snapshot based on the field name
            element_ref = f"example_ref_for_{field_name}"
            
            # Type the value into the field
            await self.call_mcp_tool(
                self.mcp_server,
                "browser_type",
                {
                    "element": f"Form field {field_name}",
                    "ref": element_ref,
                    "text": value,
                    "submit": submit and field_name == list(form_fields.keys())[-1]
                }
            )
        
        return {"status": "Form filled successfully"}


class WebsiteClickFlowTool(PlaywrightTool):
    """Tool to perform a series of clicks on a website."""
    
    def __init__(self):
        """Initialize the click flow tool."""
        super().__init__(
            name="website_click_flow",
            description="Performs a series of clicks on a website following a specified flow",
        )
    
    async def _execute(
        self,
        url: str,
        click_sequence: List[str],
        take_screenshots: bool = False,
        wait_seconds_between: float = 1.0
    ) -> Dict[str, Any]:
        """Execute the click flow tool.
        
        Args:
            url: Starting website URL
            click_sequence: List of element descriptions to click in sequence
            take_screenshots: Whether to take screenshots after each click
            wait_seconds_between: Seconds to wait between clicks
            
        Returns:
            Dictionary with results and optional screenshot paths
        """
        # Navigate to the URL
        await self.call_mcp_tool(
            self.mcp_server,
            "browser_navigate",
            {"url": url}
        )
        
        screenshots = []
        
        # Perform each click in sequence
        for i, element_description in enumerate(click_sequence):
            # Get the page snapshot
            snapshot_result = await self.call_mcp_tool(
                self.mcp_server,
                "browser_snapshot",
                {}
            )
            
            # In a real implementation, we would analyze the snapshot to find
            # the element that matches the description
            
            # For this example, we'll assume a reference format
            element_ref = f"example_ref_for_{i}"
            
            # Click the element
            await self.call_mcp_tool(
                self.mcp_server,
                "browser_click",
                {
                    "element": element_description,
                    "ref": element_ref
                }
            )
            
            # Wait between clicks
            await self.call_mcp_tool(
                self.mcp_server,
                "browser_wait_for",
                {"time": wait_seconds_between}
            )
            
            # Take a screenshot if requested
            if take_screenshots:
                screenshot_result = await self.call_mcp_tool(
                    self.mcp_server,
                    "browser_take_screenshot",
                    {"filename": f"click_step_{i+1}.png"}
                )
                screenshots.append(screenshot_result.get("path", f"Screenshot {i+1} taken"))
        
        result = {"status": "Click flow completed successfully"}
        if take_screenshots:
            result["screenshots"] = screenshots
        
        return result


# Example of how to call the MCP tool from a custom tool
# This is a simplified implementation
async def call_mcp_tool(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call an MCP tool with the given parameters.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        parameters: Parameters for the tool
        
    Returns:
        Tool execution result
    """
    # In a real implementation, this would use the MCP client to call the tool
    logger.info(f"Calling {tool_name} on server {server_name} with parameters: {parameters}")
    
    # Mock implementation for example purposes
    return {"success": True, "path": f"/path/to/output/{tool_name}_result.png"}
    
    
# Add this method to the PlaywrightTool class for actual implementation
PlaywrightTool.call_mcp_tool = call_mcp_tool