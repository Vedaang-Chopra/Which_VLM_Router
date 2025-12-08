"""
Artemis Router Module

VLM model selection and routing logic.

This module provides:
- Multiple routing strategies (Reward-based, Pairwise, Classical)
- Unified API for routing requests
- Checkpoint loading and inference utilities

Usage:
    >>> from router import init_router, route_request
    >>> init_router()
    >>> decision = route_request("What is in this image?", mode="balanced")

"""

from .public_api import (
    init_router,
    route_request,
    load_router_from_checkpoint,
    RouterService
)

__all__ = [
    "init_router",
    "route_request",
    "load_router_from_checkpoint",
    "RouterService"
]
