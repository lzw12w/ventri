"""Feishu / Lark channel (``use: ventri_agent.channels.feishu``; DESIGN.md 5.8).

Needs the optional ``feishu`` extra (``lark-oapi``), imported lazily: loading
this package does not import the SDK. Setup: docs/feishu-setup.md.
"""
from __future__ import annotations

from .channel import Conversation, FeishuChannel, FeishuConfig, feishu, plugin
from .transport import CardAction, Inbound

__all__ = ["CardAction", "Conversation", "FeishuChannel", "FeishuConfig", "Inbound", "feishu", "plugin"]
