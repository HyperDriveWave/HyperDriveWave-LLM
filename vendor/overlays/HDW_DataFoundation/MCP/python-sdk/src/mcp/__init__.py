"""Top-level MCP exports.

This local vendored copy is used by HyperDriveWave primarily for server-side
MCP integration. Some client transports pull in optional platform-specific
dependencies at import time on Windows. To keep the server usable without
forcing those imports, top-level client symbols are exposed lazily.
"""

from .server.session import ServerSession
from .server.stdio import stdio_server
from .shared.exceptions import MCPError, UrlElicitationRequiredError
from .types import (
    CallToolRequest,
    ClientCapabilities,
    ClientNotification,
    ClientRequest,
    ClientResult,
    CompleteRequest,
    CreateMessageRequest,
    CreateMessageResult,
    CreateMessageResultWithTools,
    ErrorData,
    GetPromptRequest,
    GetPromptResult,
    Implementation,
    IncludeContext,
    InitializedNotification,
    InitializeRequest,
    InitializeResult,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsResult,
    LoggingLevel,
    LoggingMessageNotification,
    Notification,
    PingRequest,
    ProgressNotification,
    PromptsCapability,
    ReadResourceRequest,
    ReadResourceResult,
    Resource,
    ResourcesCapability,
    ResourceUpdatedNotification,
    RootsCapability,
    SamplingCapability,
    SamplingContent,
    SamplingContextCapability,
    SamplingMessage,
    SamplingMessageContentBlock,
    SamplingToolsCapability,
    ServerCapabilities,
    ServerNotification,
    ServerRequest,
    ServerResult,
    SetLevelRequest,
    StopReason,
    SubscribeRequest,
    Tool,
    ToolChoice,
    ToolResultContent,
    ToolsCapability,
    ToolUseContent,
    UnsubscribeRequest,
)
from .types import Role as SamplingRole

__all__ = [
    "CallToolRequest",
    "Client",
    "ClientCapabilities",
    "ClientNotification",
    "ClientRequest",
    "ClientResult",
    "ClientSession",
    "ClientSessionGroup",
    "CompleteRequest",
    "CreateMessageRequest",
    "CreateMessageResult",
    "CreateMessageResultWithTools",
    "ErrorData",
    "GetPromptRequest",
    "GetPromptResult",
    "Implementation",
    "IncludeContext",
    "InitializeRequest",
    "InitializeResult",
    "InitializedNotification",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "ListPromptsRequest",
    "ListPromptsResult",
    "ListResourcesRequest",
    "ListResourcesResult",
    "ListToolsResult",
    "LoggingLevel",
    "LoggingMessageNotification",
    "MCPError",
    "Notification",
    "PingRequest",
    "ProgressNotification",
    "PromptsCapability",
    "ReadResourceRequest",
    "ReadResourceResult",
    "Resource",
    "ResourcesCapability",
    "ResourceUpdatedNotification",
    "RootsCapability",
    "SamplingCapability",
    "SamplingContent",
    "SamplingContextCapability",
    "SamplingMessage",
    "SamplingMessageContentBlock",
    "SamplingRole",
    "SamplingToolsCapability",
    "ServerCapabilities",
    "ServerNotification",
    "ServerRequest",
    "ServerResult",
    "ServerSession",
    "SetLevelRequest",
    "StdioServerParameters",
    "StopReason",
    "SubscribeRequest",
    "Tool",
    "ToolChoice",
    "ToolResultContent",
    "ToolsCapability",
    "ToolUseContent",
    "UnsubscribeRequest",
    "UrlElicitationRequiredError",
    "stdio_client",
    "stdio_server",
]


def __getattr__(name):
    if name == "Client":
        from .client.client import Client

        return Client
    if name == "ClientSession":
        from .client.session import ClientSession

        return ClientSession
    if name == "ClientSessionGroup":
        from .client.session_group import ClientSessionGroup

        return ClientSessionGroup
    if name in {"StdioServerParameters", "stdio_client"}:
        from .client.stdio import StdioServerParameters, stdio_client

        return {"StdioServerParameters": StdioServerParameters, "stdio_client": stdio_client}[name]
    raise AttributeError(name)
