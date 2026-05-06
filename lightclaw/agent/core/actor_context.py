"""ActorContext — identifies who is speaking in the current turn."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TrustLevel(str, Enum):
    OWNER = "owner"
    TRUSTED = "trusted"
    GROUP_MEMBER = "group_member"
    UNKNOWN = "unknown"
    SYSTEM = "system"


class ChatType(str, Enum):
    PRIVATE = "private"
    GROUP = "group"
    DASHBOARD = "dashboard"
    CRON = "cron"
    PROACTIVE = "proactive"


@dataclass
class ActorContext:
    """Identifies who is speaking in the current turn and their permissions."""

    agent_id: str = ""
    channel: str = ""
    account_id: str = ""

    chat_type: ChatType = ChatType.DASHBOARD
    chat_id: str = ""

    sender_id: str = ""
    sender_display_name: str = ""

    principal_id: str = ""
    is_owner: bool = False
    is_group_chat: bool = False

    session_id: str = ""
    session_key: str = ""

    user_id: str = ""

    trust_level: TrustLevel = TrustLevel.OWNER

    metadata: dict = field(default_factory=dict)

    @classmethod
    def dashboard_owner(cls, session_id: str = "", agent_id: str = "") -> ActorContext:
        return cls(
            agent_id=agent_id or "default",
            channel="dashboard",
            account_id="dashboard",
            chat_type=ChatType.DASHBOARD,
            chat_id="dashboard",
            sender_id="default",
            sender_display_name="Owner",
            principal_id="owner",
            is_owner=True,
            is_group_chat=False,
            session_id=session_id or "dashboard-default",
            session_key=f"{agent_id or 'default'}:dashboard:owner",
            user_id="owner",
            trust_level=TrustLevel.OWNER,
        )

    @classmethod
    def system(cls, job_id: str = "heartbeat", agent_id: str = "") -> ActorContext:
        return cls(
            agent_id=agent_id or "default",
            channel="system",
            account_id="system",
            chat_type=ChatType.CRON,
            chat_id=job_id,
            sender_id=f"system:{job_id}",
            sender_display_name="System",
            principal_id="system",
            is_owner=False,
            is_group_chat=False,
            session_id=f"proactive_{job_id}",
            session_key=f"{agent_id or 'default'}:system:{job_id}",
            user_id="system",
            trust_level=TrustLevel.SYSTEM,
        )

    @classmethod
    def unknown(cls, session_id: str = "", channel: str = "", agent_id: str = "") -> ActorContext:
        """Fallback context when actor identity cannot be determined.

        Grants minimal privileges: no owner_memory access, current_session only,
        no raw session search.  This is intentionally narrower than dashboard_owner.
        """
        return cls(
            agent_id=agent_id or "default",
            channel=channel or "unknown",
            account_id="",
            chat_type=ChatType.PRIVATE,
            chat_id="",
            sender_id="",
            sender_display_name="Unknown",
            principal_id="unknown",
            is_owner=False,
            is_group_chat=False,
            session_id=session_id or "",
            session_key="",
            user_id="",
            trust_level=TrustLevel.UNKNOWN,
        )

    def build_session_key(self) -> str:
        if self.chat_type == ChatType.DASHBOARD:
            return f"{self.agent_id}:dashboard:owner"
        if self.chat_type in (ChatType.CRON, ChatType.PROACTIVE):
            return f"{self.agent_id}:system:{self.chat_id}"
        if self.chat_type == ChatType.GROUP:
            if self.is_group_chat:
                return f"{self.agent_id}:{self.channel}:{self.account_id}:group:{self.chat_id}:user:{self.sender_id}"
            return f"{self.agent_id}:{self.channel}:{self.account_id}:group:{self.chat_id}:shared"
        return f"{self.agent_id}:{self.channel}:{self.account_id}:dm:{self.sender_id}"

    def __post_init__(self):
        if not self.session_key:
            self.session_key = self.build_session_key()

    @property
    def is_dashboard(self) -> bool:
        return self.chat_type == ChatType.DASHBOARD

    @property
    def is_private(self) -> bool:
        return self.chat_type == ChatType.PRIVATE

    @property
    def is_system(self) -> bool:
        return self.chat_type in (ChatType.CRON, ChatType.PROACTIVE)

    @property
    def sender_id_hash(self) -> str:
        import hashlib
        h = hashlib.sha256(self.sender_id.encode()).hexdigest()[:8] if self.sender_id else "none"
        return h


ActorContextLike = ActorContext | None


def resolve_principal(
    channel: str,
    sender_id: str,
    owner_principals: list[dict[str, str]] | None = None,
) -> tuple[str, bool, TrustLevel]:
    """Resolve principal identity from owner_principals config.

    Returns (principal_id, is_owner, trust_level).
    Falls back to trust_level=unknown when no match is found.
    """
    if not sender_id or not channel:
        return ("unknown", False, TrustLevel.UNKNOWN)

    entries = owner_principals or []
    for entry in entries:
        if isinstance(entry, dict):
            if entry.get("channel") == channel and entry.get("senderId", entry.get("sender_id")) == sender_id:
                pid = entry.get("principalId", entry.get("principal_id", "owner"))
                return (pid, pid == "owner", TrustLevel.OWNER if pid == "owner" else TrustLevel.TRUSTED)
        else:
            # Pydantic model
            if getattr(entry, "channel", "") == channel and getattr(entry, "sender_id", "") == sender_id:
                pid = getattr(entry, "principal_id", "owner")
                return (pid, pid == "owner", TrustLevel.OWNER if pid == "owner" else TrustLevel.TRUSTED)

    return ("unknown", False, TrustLevel.UNKNOWN)


def actor_context_to_dict(ctx: ActorContext) -> dict:
    """Serialize ActorContext to a JSON-safe dict for TurnRequest.context."""
    return {
        "agent_id": ctx.agent_id,
        "channel": ctx.channel,
        "account_id": ctx.account_id,
        "chat_type": ctx.chat_type.value if isinstance(ctx.chat_type, ChatType) else ctx.chat_type,
        "chat_id": ctx.chat_id,
        "sender_id": ctx.sender_id,
        "sender_display_name": ctx.sender_display_name,
        "principal_id": ctx.principal_id,
        "is_owner": ctx.is_owner,
        "is_group_chat": ctx.is_group_chat,
        "session_id": ctx.session_id,
        "session_key": ctx.session_key,
        "user_id": ctx.user_id,
        "trust_level": ctx.trust_level.value if isinstance(ctx.trust_level, TrustLevel) else ctx.trust_level,
    }


def actor_context_from_dict(d: dict | None) -> ActorContext | None:
    """Deserialize ActorContext from a dict (e.g. from TurnRequest.context)."""
    if not d:
        return None
    chat_type_raw = d.get("chat_type", "dashboard")
    try:
        chat_type = ChatType(chat_type_raw)
    except ValueError:
        chat_type = ChatType.DASHBOARD
    trust_raw = d.get("trust_level", "owner")
    try:
        trust_level = TrustLevel(trust_raw)
    except ValueError:
        trust_level = TrustLevel.OWNER
    ctx = ActorContext(
        agent_id=d.get("agent_id", ""),
        channel=d.get("channel", ""),
        account_id=d.get("account_id", ""),
        chat_type=chat_type,
        chat_id=d.get("chat_id", ""),
        sender_id=d.get("sender_id", ""),
        sender_display_name=d.get("sender_display_name", ""),
        principal_id=d.get("principal_id", ""),
        is_owner=d.get("is_owner", False),
        is_group_chat=d.get("is_group_chat", False),
        session_id=d.get("session_id", ""),
        session_key=d.get("session_key", ""),
        user_id=d.get("user_id", ""),
        trust_level=trust_level,
    )
    return ctx
