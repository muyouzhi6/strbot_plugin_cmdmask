from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.config import AstrBotConfig
from astrbot.core.star.filter.custom_filter import CustomFilter
from astrbot.core.star.star import star_map

VERSION = "0.3.0"


@dataclass(frozen=True)
class MappingEntry:
    alias_raw: str
    target_raw: str
    reply_mode: str
    reply_text: str


APPLIED_KEY = "__astrbot_plugin_cmdmask:applied"
WHITESPACE_PATTERN = re.compile(r"\s+")
REPLY_MODE_MAP = {
    "silent": "silent",
    "mute": "silent",
    "no_reply": "silent",
    "keep": "keep",
    "default": "keep",
    "passthrough": "keep",
    "custom": "custom",
}
REPLY_MODE_CN_MAP = {
    "静默": "silent",
    "不回复": "silent",
    "不回": "silent",
    "无回复": "silent",
    "自定义": "custom",
    "自订": "custom",
    "保留": "keep",
    "默认": "keep",
    "原样": "keep",
}


def _normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text.strip())


def _read_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _get_wake_prefixes(cfg: AstrBotConfig) -> list[str]:
    prefixes = cfg.get("wake_prefix", [])
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    if not isinstance(prefixes, list):
        return []
    return [p for p in prefixes if isinstance(p, str) and p]


def _strip_wake_prefix(text: str, prefixes: list[str], strip_common: bool = False) -> str:
    """去掉文本开头的 wake_prefix。
    
    Args:
        text: 输入文本
        prefixes: 用户配置的 wake_prefix 列表
        strip_common: 是否同时去掉常见命令前缀（仅用于配置归一化）
    """
    # 先尝试去掉用户配置的 prefixes
    for prefix in prefixes:
        if prefix and text.startswith(prefix):
            return text[len(prefix) :].strip()
    # 仅在配置归一化时，再尝试去掉常见的命令前缀
    if strip_common:
        common_prefixes = ['/', '.', '!', '！', '。']
        for prefix in common_prefixes:
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
    return text


def _normalize_reply_mode(value: Any) -> str:
    if value is None:
        return "keep"
    text = str(value).strip()
    if not text:
        return "keep"
    lower = text.lower()
    if lower in REPLY_MODE_MAP:
        return REPLY_MODE_MAP[lower]
    if text in REPLY_MODE_CN_MAP:
        return REPLY_MODE_CN_MAP[text]
    return "keep"


def _parse_reply_option(opt_strip: str) -> tuple[str | None, str | None]:
    if not opt_strip:
        return None, None
    lower = opt_strip.lower()

    if lower in REPLY_MODE_MAP:
        return REPLY_MODE_MAP[lower], None
    if opt_strip in REPLY_MODE_CN_MAP:
        return REPLY_MODE_CN_MAP[opt_strip], None

    if lower.startswith("reply_mode=") or opt_strip.startswith("回复模式="):
        mode = opt_strip.split("=", 1)[1].strip()
        return _normalize_reply_mode(mode), None

    if (
        lower.startswith("reply=")
        or lower.startswith("reply_text=")
        or opt_strip.startswith("回复=")
        or opt_strip.startswith("回复文本=")
    ):
        return "custom", opt_strip.split("=", 1)[1].strip()
    if lower.startswith("text=") or opt_strip.startswith("文本="):
        return "custom", opt_strip.split("=", 1)[1].strip()
    if opt_strip.startswith("回复:"):
        return "custom", opt_strip.split(":", 1)[1].strip()

    return "custom", opt_strip


def _parse_mapping_line(line: str) -> MappingEntry | None:
    if not line or not isinstance(line, str):
        return None
    parts = [part.strip() for part in line.split("||") if part.strip()]
    if not parts:
        return None

    mapping_part = parts[0]
    if "=>" in mapping_part:
        target_raw, alias_raw = mapping_part.split("=>", 1)
    elif "->" in mapping_part:
        target_raw, alias_raw = mapping_part.split("->", 1)
    else:
        return None

    alias_raw = alias_raw.strip()
    target_raw = target_raw.strip()
    if not alias_raw or not target_raw:
        return None

    reply_mode = "keep"
    reply_text = ""

    for opt in parts[1:]:
        mode, text = _parse_reply_option(opt.strip())
        if mode:
            reply_mode = mode
        if text is not None:
            reply_text = text

    return MappingEntry(
        alias_raw=alias_raw,
        target_raw=target_raw,
        reply_mode=reply_mode,
        reply_text=reply_text,
    )


def _build_mapping_from_config(
    command: Any,
    alias_text: Any,
    silent: Any,
    reply_text: Any,
) -> MappingEntry | None:
    command_str = _read_str(command)
    alias_str = _read_str(alias_text)
    if not command_str or not alias_str:
        return None
    reply_text_str = _read_text(reply_text)
    reply_mode = "silent" if bool(silent) else ("custom" if reply_text_str else "keep")
    return MappingEntry(
        alias_raw=alias_str,
        target_raw=command_str,
        reply_mode=reply_mode,
        reply_text=reply_text_str,
    )


def _build_fixed_mapping(command: str, data: Any) -> MappingEntry | None:
    if not isinstance(data, dict):
        return None
    return _build_mapping_from_config(
        command,
        data.get("alias_text"),
        data.get("silent", False),
        data.get("reply_text"),
    )


def _build_custom_mappings(raw_list: Any) -> list[MappingEntry]:
    mappings: list[MappingEntry] = []
    if not isinstance(raw_list, list):
        return mappings
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        entry = _build_mapping_from_config(
            item.get("command"),
            item.get("alias_text") or item.get("alias") or item.get("mask"),
            item.get("silent", False),
            item.get("reply_text") or item.get("reply") or item.get("回复"),
        )
        if entry:
            mappings.append(entry)
    return mappings


def _build_mappings(raw_list: list[Any]) -> list[MappingEntry]:
    mappings: list[MappingEntry] = []
    for item in raw_list:
        if isinstance(item, str):
            entry = _parse_mapping_line(item)
            if entry:
                mappings.append(entry)
            continue
        if isinstance(item, dict):
            command = (
                item.get("command")
                or item.get("target")
                or item.get("真实指令")
                or item.get("真實指令")
            )
            alias = (
                item.get("alias")
                or item.get("mask")
                or item.get("伪装指令")
                or item.get("偽裝指令")
            )
            command_str = _read_str(command)
            alias_str = _read_str(alias)
            if not command_str or not alias_str:
                continue
            reply_mode = _normalize_reply_mode(
                item.get("reply_mode") or item.get("回复模式") or "keep",
            )
            reply_text = _read_text(item.get("reply_text") or item.get("回复") or "")
            if reply_text and reply_mode == "keep":
                reply_mode = "custom"
            mappings.append(
                MappingEntry(
                    alias_raw=alias_str,
                    target_raw=command_str,
                    reply_mode=reply_mode,
                    reply_text=reply_text,
                ),
            )
    return mappings


def _build_mappings_from_text(text: str) -> list[MappingEntry]:
    mappings: list[MappingEntry] = []
    if not text or not isinstance(text, str):
        return mappings
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("//") or line.startswith(";"):
            continue
        entry = _parse_mapping_line(line)
        if entry:
            mappings.append(entry)
    return mappings


def _apply_mapping(
    event: AstrMessageEvent,
    cfg: AstrBotConfig,
    enabled: bool,
    mappings: list[MappingEntry],
) -> bool:
    # 入口日志 - 用 INFO 确保可见
    logger.info(f"[CmdMask] _apply_mapping enter: enabled={enabled}, mappings={len(mappings)}")
    
    if not enabled or not mappings:
        logger.info(f"[CmdMask] skip: enabled={enabled}, mappings={len(mappings)}")
        return False
    
    # 检查是否已处理 - 使用命名空间 key 避免冲突
    if event.get_extra(APPLIED_KEY, False):
        logger.info(f"[CmdMask] skip: already applied, extra_keys={list(event._extras.keys()) if hasattr(event, '_extras') else 'unknown'}")
        return True
    
    if not event.is_at_or_wake_command:
        logger.info(f"[CmdMask] skip: is_at_or_wake_command=False, message_str={event.get_message_str()!r}")
        return False

    raw_msg = _normalize_text(event.get_message_str())
    logger.info(f"[CmdMask] raw_msg={raw_msg!r}")
    if not raw_msg:
        return False

    prefixes = _get_wake_prefixes(cfg)
    
    # 检测用户实际使用的前缀
    # 优先从 message_obj.message_str 获取原始消息（未被处理）
    original_msg = ""
    if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message_str'):
        original_msg = event.message_obj.message_str or ""
    if not original_msg:
        original_msg = raw_msg
    
    used_prefix = ""
    for prefix in prefixes:
        if prefix and original_msg.startswith(prefix):
            used_prefix = prefix
            break
    
    msg = _strip_wake_prefix(raw_msg, prefixes)  # 去掉 wake_prefix 后再匹配
    logger.info(f"[CmdMask] original_msg={original_msg!r}, prefixes={prefixes}, used_prefix={used_prefix!r}, msg={msg!r}")

    for entry in mappings:
        # 配置归一化：去掉常见命令前缀
        alias_norm = _strip_wake_prefix(_normalize_text(entry.alias_raw), prefixes, strip_common=True)
        logger.info(f"[CmdMask] checking: alias_raw={entry.alias_raw!r}, alias_norm={alias_norm!r}, msg={msg!r}")
        if not alias_norm:
            continue
        # 匹配：完全相等，或 alias 后跟任意空白字符
        if msg == alias_norm or (
            msg.startswith(alias_norm) 
            and len(msg) > len(alias_norm) 
            and msg[len(alias_norm)].isspace()
        ):
            # 配置归一化：去掉常见命令前缀
            target_norm = _strip_wake_prefix(
                _normalize_text(entry.target_raw),
                prefixes,
                strip_common=True,
            )
            logger.info(f"[CmdMask] MATCHED! target_raw={entry.target_raw!r}, target_norm={target_norm!r}")
            if not target_norm:
                continue
            suffix = msg[len(alias_norm) :].strip()
            # 重写消息：直接使用 target_norm（不加前缀）
            # 因为 AstrBot 已经在 event.message_str 中去掉了 wake_prefix
            # CommandFilter 也期望不带前缀的命令名（如 "reset" 而不是 ".reset"）
            new_msg = target_norm
            if suffix:
                new_msg = f"{new_msg} {suffix}"
            
            logger.info(f"[CmdMask] rewriting: {event.message_str!r} -> {new_msg!r}")

            event.set_extra(APPLIED_KEY, True)
            event.set_extra("__astrbot_plugin_cmdmask:reply_mode", entry.reply_mode)
            event.set_extra("__astrbot_plugin_cmdmask:reply_text", entry.reply_text)
            event.set_extra("__astrbot_plugin_cmdmask:alias", alias_norm)
            event.set_extra("__astrbot_plugin_cmdmask:target", target_norm)
            event.set_extra("__astrbot_plugin_cmdmask:original_message", event.get_message_str())

            if new_msg != event.message_str:
                event.message_str = new_msg
                if (
                    hasattr(event, "message_obj")
                    and event.message_obj
                    and hasattr(event.message_obj, "message_str")
                ):
                    event.message_obj.message_str = new_msg

            event.should_call_llm(True)
            return True

    return False


class _CommandMaskFilter(CustomFilter):
    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        plugin_md = star_map.get(__name__)
        plugin = plugin_md.star_cls if plugin_md else None
        if not isinstance(plugin, CmdMask):
            return False
        logger.debug(
            f"[CmdMask] filter called, enabled={plugin._enabled}, mappings_count={len(plugin._mappings)}",
        )
        try:
            _apply_mapping(event, cfg, plugin._enabled, plugin._mappings)
        except Exception as exc:
            logger.warning(f"[CmdMask] mapping error: {exc}")
        return False


class CmdMask(Star):
    def __init__(
        self,
        context: Context,
        config: AstrBotConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config
        self._enabled = True
        self._mappings: list[MappingEntry] = []
        self._load_config()
        logger.info(
            f"[CmdMask] loaded v{VERSION} with {len(self._mappings)} mappings",
        )

    def _cfg(self, key: str, default: Any = None) -> Any:
        if self._config is None:
            return default
        return self._config.get(key, default)

    def _load_config(self) -> None:
        enabled = bool(self._cfg("enable", True))
        reset_rule = self._cfg("reset_rule", {})
        new_rule = self._cfg("new_rule", {})
        custom_rules = self._cfg("custom_rules", [])
        rules_text = self._cfg("rules_text", "")
        raw_mappings = self._cfg("mappings", [])
        if not isinstance(raw_mappings, list):
            raw_mappings = []

        mappings: list[MappingEntry] = []
        reset_entry = _build_fixed_mapping("/reset", reset_rule)
        if reset_entry:
            mappings.append(reset_entry)
        new_entry = _build_fixed_mapping("/new", new_rule)
        if new_entry:
            mappings.append(new_entry)
        mappings.extend(_build_custom_mappings(custom_rules))
        mappings.extend(_build_mappings_from_text(rules_text))
        mappings.extend(_build_mappings(raw_mappings))

        self._enabled = enabled
        self._mappings = mappings

    @filter.custom_filter(_CommandMaskFilter)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=100000)
    async def _mapping_probe(self, event: AstrMessageEvent):
        return

    @filter.on_decorating_result(priority=100000)
    async def _override_reply(self, event: AstrMessageEvent):
        if not event.get_extra(APPLIED_KEY, False):
            return

        mode = event.get_extra("__astrbot_plugin_cmdmask:reply_mode", "keep")
        if mode == "keep":
            return

        if mode == "silent":
            event.set_result(event.make_result())
            return

        if mode == "custom":
            text = event.get_extra("__astrbot_plugin_cmdmask:reply_text", "")
            if not text or not str(text).strip():
                event.set_result(event.make_result())
                return
            event.set_result(event.plain_result(str(text)))
            return

    async def terminate(self) -> None:
        logger.info("[CmdMask] terminated")
