"""
Model Context Protocol (MCP) Schemas
Defines standardized formats for agent communication.
"""

from .input import PromptSchema, DocumentSchema, ContextSchema
from .output import ProposalSchema, VerificationSchema, StatusSchema

__all__ = [
    "PromptSchema",
    "DocumentSchema",
    "ContextSchema",
    "ProposalSchema",
    "VerificationSchema",
    "StatusSchema"
] 