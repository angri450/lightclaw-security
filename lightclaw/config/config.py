from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lightclaw.constant import (
    HEARTBEAT_DEFAULT_EVERY,
    HEARTBEAT_DEFAULT_TARGET,
    PROACTIVE_DEFAULT_DAILY_ANALYSIS_TIME,
    PROACTIVE_DEFAULT_EVERY,
    PROACTIVE_DEFAULT_TARGET,
)

# Import ProvidersData here to embed it in Config.
# Use a TYPE_CHECKING guard for the heavy provider registry to avoid
# circular imports at module load time.
from lightclaw.infra.providers.models import ProvidersData


class BaseChannelConfig(BaseModel):
    """Base for channel config (read from lightclaw.json, no env)."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    bot_prefix: str = Field(default="", alias="botPrefix")


class DiscordConfig(BaseChannelConfig):
    bot_token: str = Field(default="", alias="botToken")
    http_proxy: str = Field(default="", alias="httpProxy")
    http_proxy_auth: str = Field(default="", alias="httpProxyAuth")


class DingTalkConfig(BaseChannelConfig):
    """DingTalk: client_id, client_secret; media_dir for received media."""

    client_id: str = Field(default="", alias="clientId")
    client_secret: str = Field(default="", alias="clientSecret")
    media_dir: str = Field(default="~/.lightclaw/media", alias="mediaDir")


class FeishuConfig(BaseChannelConfig):
    """Feishu/Lark channel: app_id, app_secret; optional encrypt_key,
    verification_token for event handler. media_dir for received media.
    """

    app_id: str = Field(default="", alias="appId")
    app_secret: str = Field(default="", alias="appSecret")
    encrypt_key: str = Field(default="", alias="encryptKey")
    verification_token: str = Field(default="", alias="verificationToken")
    media_dir: str = Field(default="~/.lightclaw/media", alias="mediaDir")


class QQConfig(BaseChannelConfig):
    app_id: str = Field(default="", alias="appId")
    client_secret: str = Field(default="", alias="clientSecret")

    # ── STT (speech-to-text) config ──
    stt: dict = Field(
        default_factory=dict,
        description="STT config: {baseUrl, apiKey, model, language, enabled}",
    )

    # ── TTS (text-to-speech) config (edge-tts, free, no API key required) ──
    tts: dict = Field(
        default_factory=dict,
        description="TTS config: {enabled, voice, rate, volume, pitch}",
    )


class YuanbaoConfig(BaseChannelConfig):
    """Tencent YuanBao bot channel: appKey + appSecret or static token."""

    app_key: str = Field(default="", alias="appKey")
    app_secret: str = Field(default="", alias="appSecret")
    token: str = Field(default="")
    identifier: str = Field(default="")
    api_domain: str = Field(default="bot.yuanbao.tencent.com", alias="apiDomain")
    ws_url: str = Field(default="wss://bot-wss.yuanbao.tencent.com/wss/connection", alias="wsUrl")
    route_env: str = Field(default="", alias="routeEnv")


class DashboardConfig(BaseChannelConfig):
    """Dashboard channel: prints agent responses to stdout."""

    enabled: bool = True


class WeComConfig(BaseChannelConfig):
    """WeCom (Enterprise WeChat) AI Bot channel: WebSocket long connection.

    Field aliases match OpenClaw wecom-openclaw-plugin for migration compat.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    bot_id: str = Field(default="", alias="botId")
    secret: str = Field(default="", alias="secret")
    name: str = Field(default="", alias="name")
    websocket_url: str = Field(default="", alias="websocketUrl")

    # Access control — aligned with OpenClaw dmPolicy / allowFrom
    dm_policy: str = Field(
        default="open",
        alias="dmPolicy",
        description='Private-chat policy: "open" | "allowlist" | "pairing" | "disabled"',
    )
    allow_from: list = Field(
        default_factory=list,
        alias="allowFrom",
        description="Private-chat allowlist (user IDs); only effective when dmPolicy=allowlist",
    )
    group_policy: str = Field(
        default="open",
        alias="groupPolicy",
        description='Group-chat policy: "open" | "allowlist" | "disabled"',
    )
    group_allow_from: list = Field(
        default_factory=list,
        alias="groupAllowFrom",
        description="Group-chat allowlist (group IDs); only effective when groupPolicy=allowlist",
    )
    groups: dict = Field(
        default_factory=dict,
        alias="groups",
        description="Per-group config, e.g. {groupId: {allowFrom: [...]}}",
    )

    # Behavior
    send_thinking: bool = Field(default=True, alias="sendThinkingMessage")
    media_dir: str = Field(default="~/.lightclaw/media", alias="mediaDir")
    media_local_roots: list[str] = Field(
        default_factory=list,
        alias="mediaLocalRoots",
        description="Extra local paths allowed for media access (supports ~)",
    )

    @model_validator(mode="before")
    @classmethod
    def _compat_send_thinking(cls, data):
        """Backward compat: accept legacy 'sendThinking' key."""
        if isinstance(data, dict) and "sendThinking" in data and "sendThinkingMessage" not in data:
            data["sendThinkingMessage"] = data.pop("sendThinking")
        return data


class WeixinAccountConfig(BaseModel):
    """Single WeChat account config."""

    model_config = ConfigDict(populate_by_name=True)

    account_id: str
    account_name: str = ""
    base_url: str = Field(default="https://ilinkai.weixin.qq.com", alias="baseUrl")
    token: str
    configured: bool = False
    bot_uin: str = Field(default="", alias="botUin")
    user_uin: str = Field(default="", alias="userUin")


class WeixinConfig(BaseChannelConfig):
    """WeChat iLink Bot channel config."""

    media_dir: str = Field(default="~/.lightclaw/media", alias="mediaDir")
    show_tool_details: bool = Field(default=True, alias="showToolDetails")
    accounts: list[WeixinAccountConfig] = Field(default_factory=list)


class LightClawBotAccountConfig(BaseModel):
    """Embedded LightClaw/OpenClaw plugin account config."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = True
    api_key: str = Field(default="", alias="apiKey")
    api_keys: list[str] = Field(default_factory=list, alias="apiKeys")
    api_base_url: str = Field(default="", alias="apiBaseUrl")
    name: str = ""
    dm_policy: str = Field(default="open", alias="dmPolicy")
    allow_from: list[str] = Field(
        default_factory=lambda: ["*"],
        alias="allowFrom",
    )
    system_prompt: str = Field(default="", alias="systemPrompt")

    @model_validator(mode="before")
    @classmethod
    def _normalize_api_keys(cls, data):
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        raw_api_keys = payload.get("apiKeys")
        api_key = payload.get("apiKey", "")
        # Sync apiKey into apiKeys when apiKeys is missing or empty
        if not raw_api_keys and api_key:
            payload["apiKeys"] = [api_key]
        # Ensure apiKey is present in apiKeys (deduplicated)
        elif api_key and isinstance(raw_api_keys, list) and api_key not in raw_api_keys:
            payload["apiKeys"] = [api_key, *raw_api_keys]
        return payload


class LightClawBotConfig(LightClawBotAccountConfig):
    """Top-level lightclawbot channel config in lightclaw.json."""

    enabled: bool = False
    accounts: dict[str, LightClawBotAccountConfig] = Field(default_factory=dict)


class ChannelConfig(BaseModel):
    """Built-in channel configs; extra keys allowed for plugin channels."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    discord: DiscordConfig = DiscordConfig()
    dingtalk: DingTalkConfig = DingTalkConfig()
    feishu: FeishuConfig = FeishuConfig()
    qqbot: QQConfig = Field(default_factory=QQConfig)
    yuanbao: YuanbaoConfig = Field(default_factory=YuanbaoConfig)
    dashboard: DashboardConfig = DashboardConfig()
    wecom: WeComConfig = WeComConfig()
    weixin: WeixinConfig = Field(default_factory=WeixinConfig)
    lightclawbot: LightClawBotConfig = Field(default_factory=LightClawBotConfig)

    @model_validator(mode="before")
    @classmethod
    def _normalize_qq_key(cls, data):
        """Backward compat: map legacy keys to current config attrs."""
        if isinstance(data, dict) and "qq" in data and "qqbot" not in data:
            data = dict(data)
            data["qqbot"] = data.pop("qq")
        if isinstance(data, dict) and "lightclaw" in data and "lightclawbot" not in data:
            data = dict(data)
            data["lightclawbot"] = data.pop("lightclaw")
        return data


class LastApiConfig(BaseModel):
    host: str | None = None
    port: int | None = None
    ssl: bool = False
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None


class ActiveHoursConfig(BaseModel):
    """Optional active window for heartbeat (e.g. 08:00–22:00)."""

    start: str = "08:00"
    end: str = "22:00"


class QuietHoursConfig(BaseModel):
    """Optional quiet window where all proactive pushes are suppressed (e.g. 22:00–08:00)."""

    start: str = "22:00"
    end: str = "08:00"


class HeartbeatConfig(BaseModel):
    """Heartbeat: run agent with HEARTBEAT.md as query at interval."""

    model_config = {"populate_by_name": True}

    every: str = Field(default=HEARTBEAT_DEFAULT_EVERY)
    target: str = Field(default=HEARTBEAT_DEFAULT_TARGET)
    active_hours: ActiveHoursConfig | None = Field(
        default=None,
        alias="activeHours",
    )


class ProactivityConfig(BaseModel):
    """Proactivity: periodic agent-initiated messages with probability gate."""

    model_config = {"populate_by_name": True}

    enabled: bool = True
    every: str = Field(default=PROACTIVE_DEFAULT_EVERY)
    target: str = Field(default=PROACTIVE_DEFAULT_TARGET)
    cooldown_hours: float = Field(
        default=8.0,
        alias="cooldownHours",
    )
    active_hours: ActiveHoursConfig | None = Field(
        default=None,
        alias="activeHours",
    )
    quiet_hours: QuietHoursConfig | None = Field(
        default_factory=QuietHoursConfig,
        alias="quietHours",
    )
    daily_analysis_time: str = Field(
        default=PROACTIVE_DEFAULT_DAILY_ANALYSIS_TIME,
        alias="dailyAnalysisTime",
    )


class DreamingConfig(BaseModel):
    """Dreaming pipeline: 3-phase nightly memory consolidation (Light/REM/Deep)."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = True
    cron: str = "0 2 * * *"
    light_ingest_enabled: bool = Field(default=True, alias="lightIngestEnabled")
    light_ingest_cron: str = Field(default="30 14 * * *", alias="lightIngestCron")
    timezone: str = "Asia/Shanghai"
    recent_days: int = Field(default=7, ge=1, le=90)
    max_daily_talk_chars: int = Field(default=10000, ge=1000, le=50000)
    max_recent_memory_chars: int = Field(default=20000, ge=1000, le=100000)
    dynamic_threshold_base: float = Field(default=0.35, ge=0.0, le=1.0)
    dynamic_threshold_max: float = Field(default=0.50, ge=0.0, le=1.0)
    max_prune_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    protected_types: list[str] = Field(
        default_factory=lambda: ["identity", "constraint", "todo", "decision"],
        alias="protectedTypes",
    )
    auto_update_user_profile: bool = Field(default=True, alias="autoUpdateUserProfile")
    auto_update_proactivity: bool = Field(default=True, alias="autoUpdateProactivity")
    auto_update_tools: bool = Field(default=True, alias="autoUpdateTools")
    auto_update_agents: bool = Field(default=True, alias="autoUpdateAgents")
    auto_update_heartbeat: bool = Field(default=False, alias="autoUpdateHeartbeat")
    write_dream_report: bool = Field(default=True, alias="writeDreamReport")


class OwnerPrincipalEntry(BaseModel):
    """Maps a channel+sender to a principal identity."""
    model_config = ConfigDict(populate_by_name=True)
    channel: str = ""
    sender_id: str = Field(default="", alias="senderId")
    principal_id: str = Field(default="", alias="principalId")


class PrincipalAliasEntry(BaseModel):
    """Aliases mapping a principal to multiple channel+sender pairs."""
    model_config = ConfigDict(populate_by_name=True)
    principal_id: str = Field(default="", alias="principalId")
    aliases: list[dict[str, str]] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    """Session / Principal Isolation & Context Budget security configuration."""
    model_config = ConfigDict(populate_by_name=True)

    # Recall scope
    recall_scope: str = Field(default="current_session", alias="recallScope")

    # Realtime injection gates
    daily_talk_realtime_injection_enabled: bool = Field(
        default=False, alias="dailyTalkRealtimeInjectionEnabled"
    )
    include_raw_sessions_in_realtime_recall: bool = Field(
        default=False, alias="includeRawSessionsInRealtimeRecall"
    )

    # Cross-scope gates
    private_to_group_recall_enabled: bool = Field(
        default=False, alias="privateToGroupRecallEnabled"
    )
    group_to_private_recall_enabled: bool = Field(
        default=False, alias="groupToPrivateRecallEnabled"
    )
    unknown_user_memory_promotion_enabled: bool = Field(
        default=False, alias="unknownUserMemoryPromotionEnabled"
    )
    group_memory_enabled: bool = Field(default=False, alias="groupMemoryEnabled")
    secret_realtime_recall_enabled: bool = Field(
        default=False, alias="secretRealtimeRecallEnabled"
    )
    old_metadata_realtime_recall_enabled: bool = Field(
        default=False, alias="oldMetadataRealtimeRecallEnabled"
    )

    # Session key settings
    group_sessions_per_user: bool = Field(default=True, alias="groupSessionsPerUser")
    group_shared_session_enabled: bool = Field(
        default=False, alias="groupSharedSessionEnabled"
    )

    # Principal identity
    owner_principals: list[OwnerPrincipalEntry] = Field(
        default_factory=list, alias="ownerPrincipals"
    )
    principal_aliases: list[PrincipalAliasEntry] = Field(
        default_factory=list, alias="principalAliases"
    )

    # Budget limits
    auto_recall_max_total_chars: int = Field(default=12000, alias="autoRecallMaxTotalChars")
    auto_recall_max_candidate_chars: int = Field(default=2000, alias="autoRecallMaxCandidateChars")
    auto_recall_max_session_result_chars: int = Field(
        default=2000, alias="autoRecallMaxSessionResultChars"
    )
    auto_recall_max_daily_talk_chars: int = Field(default=0, alias="autoRecallMaxDailyTalkChars")
    auto_recall_max_memory_chars: int = Field(default=4000, alias="autoRecallMaxMemoryChars")
    summary_block_max_chars: int = Field(default=12000, alias="summaryBlockMaxChars")
    extra_parts_max_total_chars: int = Field(default=30000, alias="extraPartsMaxTotalChars")
    tool_message_single_max_chars: int = Field(default=8000, alias="toolMessageSingleMaxChars")
    tool_message_total_max_chars: int = Field(default=24000, alias="toolMessageTotalMaxChars")


class AgentsDefaultsConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    heartbeat: HeartbeatConfig | None = None
    proactivity: ProactivityConfig | None = None
    user_timezone: str | None = Field(
        default=None,
        alias="userTimezone",
        description="User's preferred IANA timezone name (e.g. 'Asia/Shanghai', 'America/New_York').",
    )


class AgentsRunningConfig(BaseModel):
    """Agent runtime behavior configuration."""

    model_config = ConfigDict(populate_by_name=True)

    max_iters: int = Field(
        default=50,
        ge=1,
        alias="maxIters",
        description=("Maximum number of reasoning-acting iterations for ReAct agent"),
    )
    max_input_length: int = Field(
        default=128 * 1024,  # 128K = 131072 tokens
        ge=1000,
        alias="maxInputLength",
        description=("Maximum input length (tokens) for the model context window"),
    )


class CompactionAgentConfig(BaseModel):
    """Fine-grained compaction configuration.

    Loaded from ``agents.compaction`` in ``lightclaw.json``.
    All compaction uses the safeguard pipeline (structured output,
    identifier preservation, adaptive chunking, quality guard).
    """

    model_config = ConfigDict(populate_by_name=True)

    identifier_policy: str = Field(
        default="strict",
        alias="identifierPolicy",
        description="Identifier preservation policy: 'strict', 'off', or 'custom'",
    )
    identifier_instructions: str = Field(
        default="",
        alias="identifierInstructions",
        description="Custom identifier instructions (only used when identifierPolicy='custom')",
    )
    quality_guard_enabled: bool = Field(
        default=True,
        alias="qualityGuardEnabled",
        description="Enable post-compaction quality audit",
    )
    quality_guard_max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        alias="qualityGuardMaxRetries",
        description="Max retries when quality audit fails (0-3)",
    )
    recent_turns_preserve: int = Field(
        default=3,
        ge=0,
        le=12,
        alias="recentTurnsPreserve",
        description="Number of recent turns to preserve verbatim (0-12)",
    )
    max_history_share: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        alias="maxHistoryShare",
        description="Fraction of context window reserved for history (0.1-0.9)",
    )
    base_chunk_ratio: float = Field(
        default=0.4,
        ge=0.1,
        le=0.8,
        alias="baseChunkRatio",
        description="Default chunk size as fraction of context window",
    )
    min_chunk_ratio: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        alias="minChunkRatio",
        description="Floor for chunk ratio when messages are large",
    )
    safety_margin: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        alias="safetyMargin",
        description="Token estimate safety multiplier (1.0-2.0)",
    )
    custom_instructions: str = Field(
        default="",
        alias="customInstructions",
        description="Extra instructions appended to compaction prompt",
    )
    timeout_seconds: int = Field(
        default=900,
        ge=60,
        le=3600,
        alias="timeoutSeconds",
        description="Compaction LLM call timeout in seconds",
    )


class AgentsConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    defaults: AgentsDefaultsConfig = Field(
        default_factory=AgentsDefaultsConfig,
    )
    running: AgentsRunningConfig = Field(
        default_factory=AgentsRunningConfig,
    )
    language: str = Field(
        default="zh",
        description="Language for agent MD files (en/zh)",
    )
    installed_prompts_language: str | None = Field(
        default=None,
        alias="installedPromptsLanguage",
        description="Language of currently installed prompt files",
    )


class LastDispatchConfig(BaseModel):
    """Last channel/user/session that received a user-originated reply."""

    model_config = ConfigDict(populate_by_name=True)

    channel: str = ""
    user_id: str = Field(default="", alias="userId")
    canonical_user_id: str = Field(default="", alias="canonicalUserId")
    session_id: str = Field(default="", alias="sessionId")


class OpenClawConfig(BaseModel):
    """Optional pointers to the OpenClaw installation.

    When either field is set it overrides the default discovery logic used by
    the ``sync_from_openclaw`` command / API.

    Defaults (when fields are *None*):
    - ``config_path``    → ``~/.openclaw/openclaw.json``
    - ``workspace_path`` → value of ``agents.defaults.workspace`` inside that
                           openclaw.json (usually ``~/.openclaw/workspace``)
    """

    model_config = ConfigDict(populate_by_name=True)

    config_path: str | None = Field(
        default=None,
        alias="configPath",
        description="Override path to openclaw.json",
    )
    workspace_path: str | None = Field(
        default=None,
        alias="workspacePath",
        description="Override path to openclaw workspace directory",
    )


class SkillEntryConfig(BaseModel):
    """Configuration for a single skill entry (enabled/disabled)."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = True


class SkillsConfig(BaseModel):
    """Skills configuration — mirrors openclaw's plugins.entries pattern.

    Example lightclaw.json:
        {
          "skills": {
            "entries": {
              "pdf":  { "enabled": true },
              "xlsx": { "enabled": false }
            }
          }
        }

    Skills whose name does not appear in ``entries`` are treated as disabled.
    """

    model_config = ConfigDict(populate_by_name=True)

    entries: dict[str, SkillEntryConfig] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration.

    provider values:
    - ``none``   : disable vector search, use FTS keyword search only (default)
    - ``local``  : use local fastembed model (requires local-embedding extra)
    - ``custom`` : use a user-defined OpenAI-compatible embedding API
    """

    model_config = ConfigDict(populate_by_name=True)

    provider: Literal["none", "local", "custom"] = "none"

    # --- local mode ---
    local_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        alias="localModel",
        description=(
            "fastembed TextEmbedding model ID (not an arbitrary HuggingFace name); "
            "downloaded automatically on first use. See fastembed.TextEmbedding.list_supported_models()"
        ),
    )

    # --- custom mode ---
    api_key: str = Field(default="", alias="apiKey")
    base_url: str = Field(default="", alias="baseUrl")
    model_name: str = Field(default="", alias="modelName")
    dimensions: int = Field(default=512, ge=1)

    # Known model dimensions (auto-detected based on model name)
    _MODEL_DIMENSIONS: dict[str, int] = {
        # BAAI BGE series
        "BAAI/bge-small-zh-v1.5": 512,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-zh-v1.5": 768,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        # fastembed preset multilingual / CJK models
        "jinaai/jina-embeddings-v2-base-zh": 768,
        # Ollama common models
        "nomic-embed-text": 768,
        "snowflake-arctic-embed": 768,
    }

    def get_vector_dimensions(self) -> int:
        """Infer vector dimensions automatically.

        Priority:
        1. provider="custom" with explicit dimensions set — use that value.
        2. provider="local" — look up local_model in the dimension table.
        3. provider="custom" — look up model_name in the dimension table.
        4. Default: 768.
        """
        if self.provider == "custom":
            # If the user left the default 512, try to look up from model_name.
            if self.model_name and self.model_name in self._MODEL_DIMENSIONS:
                return self._MODEL_DIMENSIONS[self.model_name]
            return self.dimensions

        if self.provider == "local":
            # Look up the local_model name in the dimension table.
            if self.local_model in self._MODEL_DIMENSIONS:
                return self._MODEL_DIMENSIONS[self.local_model]
            # Heuristic fallback for BAAI models not in the lookup table.
            if "bge-small" in self.local_model:
                return 512
            if "bge-base" in self.local_model:
                return 768
            if "bge-large" in self.local_model:
                return 1024
            # Unknown model: fall back to 768.
            return 768

        # provider="none": vector dimensions are unused.
        return 768


class StarOfficeConfig(BaseModel):
    """Star Office UI integration configuration.

    When enabled, the Dashboard exposes a one-click toggle to switch between
    the normal management view and the Star Office pixel-art office view.
    The ``url`` field should point to a running Star-Office-UI backend
    (e.g. ``http://127.0.0.1:19000``).
    """

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    url: str = Field(
        default="http://127.0.0.1:19000",
        description="Star Office UI backend URL",
    )


class Config(BaseModel):
    """Root config (lightclaw.json)."""

    model_config = ConfigDict(populate_by_name=True)

    channels: ChannelConfig = ChannelConfig()
    last_api: LastApiConfig = Field(default_factory=LastApiConfig, alias="lastApi")
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    last_dispatch: LastDispatchConfig | None = Field(default=None, alias="lastDispatch")
    # When False, hide tool call/result messages entirely.
    show_tool_details: bool = Field(default=True, alias="showToolDetails")
    # When False, hide model reasoning/thinking content from users
    # (Dashboard bubble + IM channel "💭 思考过程" message). The model
    # still thinks internally; only the delivery to end-users is suppressed.
    show_thinking: bool = Field(default=True, alias="showThinking")
    # Provider/model config — merged into lightclaw.json (openclaw-compatible).
    models: ProvidersData = Field(default_factory=ProvidersData)
    # Skills enable/disable config — mirrors openclaw plugins.entries pattern.
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    # Optional openclaw integration paths.
    openclaw: OpenClawConfig = Field(default_factory=OpenClawConfig)
    # Embedding provider config (vector search for memory).
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # Proactivity config — top-level for easy access.
    # Default enabled with 30m interval; can be disabled via lightclaw.json.
    proactivity: ProactivityConfig = Field(default_factory=ProactivityConfig)
    # Dreaming pipeline config — 3-phase nightly memory consolidation.
    dreaming: DreamingConfig = Field(default_factory=DreamingConfig)
    # Star Office UI integration (pixel-art office visualization).
    star_office: StarOfficeConfig = Field(
        default_factory=StarOfficeConfig,
        alias="starOffice",
    )
    # Session / Principal Isolation & Context Budget security configuration.
    security: SecurityConfig = Field(default_factory=SecurityConfig)


# Mapping from runtime channel registry key -> ChannelConfig attribute name.
# Used when the lightclaw.json key (openclaw-compatible) differs from the
# internal channel identifier.  e.g. QQ channel runs as "qq" internally but
# stores its config under "qqbot" in lightclaw.json.
CHANNEL_REGISTRY_KEY_TO_CONFIG_ATTR: dict[str, str] = {
    "qq": "qqbot",
}


ChannelConfigUnion = (
    DiscordConfig
    | DingTalkConfig
    | FeishuConfig
    | QQConfig
    | YuanbaoConfig
    | DashboardConfig
    | WeComConfig
    | WeixinConfig
    | LightClawBotConfig
)
