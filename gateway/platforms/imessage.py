"""
iMessage platform adapter via BlueBubbles Server.

BlueBubbles provides a REST API + WebSocket gateway to Apple's Messages.app,
enabling iMessage/SMS integration on macOS. This adapter communicates with a
locally-running (or tunnelled) BlueBubbles Server instance.

Features:
- Send and receive text, images, video, voice, and documents
- Tapback reactions (love, like, dislike, laugh, emphasize, question)
- Read receipts
- Group chat support
- Typing indicators

Configuration (env vars or config.yaml):
- BLUEBUBBLES_SERVER_URL: Base URL (e.g. http://localhost:1234)
- BLUEBUBBLES_PASSWORD:   Server password
- IMESSAGE_ALLOWED_USERS: Comma-separated phone numbers / emails
- IMESSAGE_HOME_CHANNEL:  Default chat GUID for cron delivery
"""

import asyncio
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as url_quote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BlueBubbles associatedMessageType codes for tapback reactions
_TAPBACK_ADDED = {
    2000: "love",
    2001: "like",
    2002: "dislike",
    2003: "laugh",
    2004: "emphasize",
    2005: "question",
}
_TAPBACK_REMOVED = {
    3000: "love",
    3001: "like",
    3002: "dislike",
    3003: "laugh",
    3004: "emphasize",
    3005: "question",
}

# iMessage has no hard per-message limit, but keep it reasonable
MAX_MESSAGE_LENGTH = 20000

# Polling interval in seconds
_POLL_INTERVAL = 1.5

# Phone number / email redaction for logs
_PHONE_RE = re.compile(r"\+?\d{7,15}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _redact(text: str) -> str:
    """Redact phone numbers and emails from log output."""
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    return text


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def check_imessage_requirements() -> bool:
    """Check if iMessage/BlueBubbles dependencies are available."""
    try:
        import httpx  # noqa: F401 -- core dependency of hermes-agent
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# iMessage Adapter
# ---------------------------------------------------------------------------

class IMessageAdapter(BasePlatformAdapter):
    """
    iMessage adapter powered by BlueBubbles Server.

    Communicates with BlueBubbles via its REST API:
    - Polls for new messages on a timer
    - Sends text/media via HTTP endpoints
    - Supports tapback reactions and read receipts
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.IMESSAGE)
        self._server_url: str = config.extra.get(
            "server_url",
            os.getenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234"),
        ).rstrip("/")
        self._password: str = config.extra.get(
            "password",
            os.getenv("BLUEBUBBLES_PASSWORD", ""),
        )
        self._has_private_api: bool = False
        self._poll_task: Optional[asyncio.Task] = None
        self._http_client: Optional[Any] = None  # httpx.AsyncClient
        # Track the last message timestamp (ms) for polling
        self._last_poll_ts: int = 0
        # Deduplication: recently seen message GUIDs
        self._seen_guids: set = set()
        self._seen_guids_max: int = 500

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _api_url(self, path: str, **extra_params: str) -> str:
        """Build a full BlueBubbles API URL with auth."""
        base = f"{self._server_url}/api/v1{path}"
        params = {"password": self._password}
        params.update(extra_params)
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{qs}"

    async def _get(self, path: str, **kwargs) -> Optional[Dict]:
        """GET request to BlueBubbles."""
        try:
            resp = await self._http_client.get(self._api_url(path), timeout=15, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("[iMessage] GET %s failed: %s", path, e)
            return None

    async def _post(self, path: str, json_body: Optional[Dict] = None, **kwargs) -> Optional[Dict]:
        """POST request to BlueBubbles (JSON body)."""
        try:
            resp = await self._http_client.post(
                self._api_url(path), json=json_body or {}, timeout=30, **kwargs
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("[iMessage] POST %s failed: %s", path, e)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to the BlueBubbles server and start polling."""
        import httpx

        if not self._password:
            logger.error("[iMessage] BLUEBUBBLES_PASSWORD is not set")
            self._set_fatal_error(
                "imessage_no_password",
                "BlueBubbles password not configured. Set BLUEBUBBLES_PASSWORD.",
                retryable=False,
            )
            return False

        self._http_client = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "HermesAgent/1.0"},
        )

        # Ping the server
        ping = await self._get("/ping")
        if not ping:
            logger.error("[iMessage] Cannot reach BlueBubbles at %s", self._server_url)
            await self._http_client.aclose()
            self._http_client = None
            self._set_fatal_error(
                "imessage_unreachable",
                f"Cannot reach BlueBubbles server at {self._server_url}. "
                "Make sure BlueBubbles Server is running.",
                retryable=True,
            )
            return False

        # Probe server info for Private API support
        info = await self._get("/server/info")
        if info and isinstance(info.get("data"), dict):
            self._has_private_api = bool(info["data"].get("private_api"))
            server_ver = info["data"].get("server_version", "unknown")
            print(f"[iMessage] BlueBubbles v{server_ver} "
                  f"(Private API: {'yes' if self._has_private_api else 'no'})")
        else:
            print("[iMessage] Connected (could not determine Private API status)")

        # Set poll start time to now (don't replay old messages)
        self._last_poll_ts = int(time.time() * 1000)

        # Start polling
        self._poll_task = asyncio.create_task(self._poll_messages())
        self._mark_connected()
        print(f"[iMessage] Connected to BlueBubbles at {self._server_url}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from BlueBubbles."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        self._poll_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._mark_disconnected()
        print("[iMessage] Disconnected")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message via BlueBubbles."""
        if not self._running or not self._http_client:
            return SendResult(success=False, error="Not connected")

        payload: Dict[str, Any] = {
            "chatGuid": chat_id,
            "tempGuid": uuid.uuid4().hex,
            "message": content,
        }
        if reply_to and self._has_private_api:
            payload["method"] = "private-api"
            payload["selectedMessageGuid"] = reply_to
            payload["partIndex"] = 0

        result = await self._post("/message/text", payload)
        if result and result.get("status") == 200:
            msg_guid = None
            data = result.get("data")
            if isinstance(data, dict):
                msg_guid = data.get("guid")
            return SendResult(success=True, message_id=msg_guid, raw_response=result)
        error = result.get("message", "Unknown error") if result else "No response"
        return SendResult(success=False, error=error)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator (requires Private API)."""
        if not self._has_private_api or not self._running or not self._http_client:
            return
        try:
            encoded = url_quote(chat_id, safe="")
            await self._http_client.post(
                self._api_url(f"/chat/{encoded}/typing"), timeout=5
            )
        except Exception:
            pass

    async def stop_typing(self, chat_id: str) -> None:
        """Stop typing indicator."""
        if not self._has_private_api or not self._running or not self._http_client:
            return
        try:
            encoded = url_quote(chat_id, safe="")
            await self._http_client.delete(
                self._api_url(f"/chat/{encoded}/typing"), timeout=5
            )
        except Exception:
            pass

    async def _send_attachment(
        self,
        chat_id: str,
        file_path: str,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        is_audio_message: bool = False,
    ) -> SendResult:
        """Send a file attachment via BlueBubbles multipart upload."""
        if not self._running or not self._http_client:
            return SendResult(success=False, error="Not connected")

        if not os.path.isfile(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        fname = filename or os.path.basename(file_path)

        try:
            import httpx

            with open(file_path, "rb") as f:
                files = {"attachment": (fname, f, "application/octet-stream")}
                data: Dict[str, str] = {
                    "chatGuid": chat_id,
                    "name": fname,
                    "tempGuid": uuid.uuid4().hex,
                }
                if is_audio_message:
                    data["isAudioMessage"] = "true"

                resp = await self._http_client.post(
                    self._api_url("/message/attachment"),
                    files=files,
                    data=data,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()

            # Send caption as follow-up text if provided
            if caption and result.get("status") == 200:
                await self.send(chat_id, caption)

            if result.get("status") == 200:
                msg_guid = None
                rdata = result.get("data")
                if isinstance(rdata, dict):
                    msg_guid = rdata.get("guid")
                return SendResult(success=True, message_id=msg_guid, raw_response=result)

            return SendResult(
                success=False,
                error=result.get("message", "Attachment upload failed"),
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download image URL and send as attachment."""
        try:
            from gateway.platforms.base import cache_image_from_url

            local_path = await cache_image_from_url(image_url)
            return await self._send_attachment(chat_id, local_path, caption=caption)
        except Exception:
            return await super().send_image(chat_id, image_url, caption, reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file."""
        return await self._send_attachment(chat_id, image_path, caption=caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send an audio file as a voice message."""
        return await self._send_attachment(
            chat_id, audio_path, caption=caption, is_audio_message=True,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video file."""
        return await self._send_attachment(chat_id, video_path, caption=caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        return await self._send_attachment(
            chat_id, file_path, filename=file_name, caption=caption,
        )

    async def send_reaction(
        self,
        chat_id: str,
        message_guid: str,
        reaction: str,
        part_index: int = 0,
    ) -> SendResult:
        """
        Send a tapback reaction (requires Private API).

        reaction: one of love, like, dislike, laugh, emphasize, question
                  prefix with '-' to remove (e.g. '-love')
        """
        if not self._has_private_api:
            return SendResult(success=False, error="Reactions require BlueBubbles Private API")

        payload = {
            "chatGuid": chat_id,
            "selectedMessageGuid": message_guid,
            "reaction": reaction,
            "partIndex": part_index,
        }
        result = await self._post("/message/react", payload)
        if result and result.get("status") == 200:
            return SendResult(success=True, raw_response=result)
        error = result.get("message", "Reaction failed") if result else "No response"
        return SendResult(success=False, error=error)

    async def mark_read(self, chat_id: str) -> bool:
        """Mark a chat as read (requires Private API)."""
        if not self._has_private_api:
            return False
        try:
            encoded = url_quote(chat_id, safe="")
            resp = await self._http_client.post(
                self._api_url(f"/chat/{encoded}/read"), timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug("[iMessage] mark_read failed for %s: %s", _redact(chat_id), e)
            return False

    # ------------------------------------------------------------------
    # Chat info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about an iMessage chat."""
        if not self._running or not self._http_client:
            return {"name": chat_id, "type": "dm", "chat_id": chat_id}

        try:
            encoded = url_quote(chat_id, safe="")
            result = await self._get(f"/chat/{encoded}")
            if result and isinstance(result.get("data"), dict):
                data = result["data"]
                is_group = ";+;" in chat_id or data.get("style") == 43
                participants = data.get("participants", [])
                display_name = data.get("displayName") or ""

                if not display_name and not is_group and participants:
                    # For DMs, use the participant's display name
                    display_name = participants[0].get("displayName", chat_id)

                return {
                    "name": display_name or chat_id,
                    "type": "group" if is_group else "dm",
                    "chat_id": chat_id,
                    "participants": [
                        p.get("address", "") for p in participants
                    ],
                }
        except Exception as e:
            logger.debug("[iMessage] get_chat_info failed: %s", e)

        # Infer type from GUID format
        is_group = ";+;" in chat_id
        return {
            "name": chat_id,
            "type": "group" if is_group else "dm",
            "chat_id": chat_id,
        }

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_messages(self) -> None:
        """Continuously poll BlueBubbles for new messages."""
        while self._running:
            try:
                payload = {
                    "limit": 50,
                    "offset": 0,
                    "after": self._last_poll_ts,
                    "sort": "ASC",
                    "with": ["chat", "attachment", "handle"],
                }
                result = await self._post("/message/query", payload)
                if result and isinstance(result.get("data"), list):
                    for msg in result["data"]:
                        msg_guid = msg.get("guid", "")

                        # Update poll timestamp (advance past this message)
                        msg_ts = msg.get("dateCreated", 0)
                        if isinstance(msg_ts, (int, float)) and msg_ts >= self._last_poll_ts:
                            # +1ms to ensure the next poll excludes this message
                            self._last_poll_ts = int(msg_ts) + 1

                        # Deduplicate by GUID
                        if msg_guid in self._seen_guids:
                            continue
                        self._seen_guids.add(msg_guid)
                        # Cap the set size to avoid unbounded growth
                        if len(self._seen_guids) > self._seen_guids_max:
                            # Discard oldest entries (sets are unordered, so just trim)
                            excess = len(self._seen_guids) - self._seen_guids_max // 2
                            for _ in range(excess):
                                self._seen_guids.pop()

                        # Skip self-messages to avoid echo loops
                        if msg.get("isFromMe"):
                            continue

                        # Build and dispatch event
                        event = await self._build_message_event(msg)
                        if event:
                            # Mark as read
                            if event.source and event.source.chat_id:
                                asyncio.create_task(self.mark_read(event.source.chat_id))
                            await self.handle_message(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[iMessage] Poll error: %s", e)
                await asyncio.sleep(5)
                continue

            await asyncio.sleep(_POLL_INTERVAL)

    async def _build_message_event(self, msg: Dict[str, Any]) -> Optional[MessageEvent]:
        """Convert a BlueBubbles message dict into a MessageEvent."""
        try:
            # --- Determine chat ---
            chats = msg.get("chats", [])
            if not chats:
                return None
            chat = chats[0]
            chat_guid = chat.get("guid", "")
            chat_name = chat.get("displayName", "")

            is_group = ";+;" in chat_guid or chat.get("style") == 43
            chat_type = "group" if is_group else "dm"

            # --- Sender ---
            handle = msg.get("handle") or {}
            sender_address = handle.get("address", "")
            sender_name = handle.get("displayName") or sender_address

            # --- Handle tapback reactions ---
            assoc_type = msg.get("associatedMessageType")
            if assoc_type is not None and assoc_type != 0:
                # This is a reaction, not a regular message. Log and skip.
                assoc_guid = msg.get("associatedMessageGuid", "")
                reaction_name = _TAPBACK_ADDED.get(assoc_type) or _TAPBACK_REMOVED.get(assoc_type, "unknown")
                is_add = assoc_type in _TAPBACK_ADDED
                logger.info(
                    "[iMessage] Tapback %s '%s' on %s from %s",
                    "added" if is_add else "removed",
                    reaction_name,
                    _redact(assoc_guid),
                    _redact(sender_address),
                )
                # We don't generate a MessageEvent for reactions (Hermes gateway
                # doesn't have a standard "reaction received" event type).
                return None

            # --- Message type & text ---
            text = msg.get("text") or ""
            msg_type = MessageType.TEXT

            attachments = msg.get("attachments", [])
            media_urls: List[str] = []
            media_types: List[str] = []

            for att in attachments:
                mime = att.get("mimeType", "") or ""
                att_guid = att.get("guid", "")

                if not att_guid:
                    continue

                # Download attachment to local cache
                cached = await self._download_attachment(att_guid, att)

                if cached:
                    media_urls.append(cached)
                    media_types.append(mime)

                    if mime.startswith("image/"):
                        msg_type = MessageType.PHOTO
                    elif mime.startswith("video/"):
                        msg_type = MessageType.VIDEO
                    elif mime.startswith("audio/") or att.get("uti", "").endswith("caf"):
                        msg_type = MessageType.VOICE
                    else:
                        msg_type = MessageType.DOCUMENT

            # If multiple attachments, prefer photo type
            if len(attachments) > 1:
                mime_set = {(a.get("mimeType") or "").split("/")[0] for a in attachments}
                if "image" in mime_set:
                    msg_type = MessageType.PHOTO

            # --- Inject text-readable document content ---
            MAX_TEXT_INJECT = 100 * 1024
            if msg_type == MessageType.DOCUMENT and media_urls:
                for doc_path in media_urls:
                    ext = Path(doc_path).suffix.lower()
                    text_exts = (
                        ".txt", ".md", ".csv", ".json", ".xml",
                        ".yaml", ".yml", ".log", ".py", ".js",
                        ".ts", ".html", ".css",
                    )
                    if ext in text_exts:
                        try:
                            sz = Path(doc_path).stat().st_size
                            if sz > MAX_TEXT_INJECT:
                                continue
                            content = Path(doc_path).read_text(errors="replace")
                            fname = Path(doc_path).name
                            injection = f"[Content of {fname}]:\n{content}"
                            text = f"{injection}\n\n{text}" if text else injection
                        except Exception as e:
                            logger.debug("[iMessage] Failed to inject doc text: %s", e)

            # Build source
            source = self.build_source(
                chat_id=chat_guid,
                chat_name=chat_name or None,
                chat_type=chat_type,
                user_id=sender_address,
                user_name=sender_name,
            )

            # Reply context
            reply_to_guid = msg.get("threadOriginatorGuid")

            return MessageEvent(
                text=text,
                message_type=msg_type,
                source=source,
                raw_message=msg,
                message_id=msg.get("guid"),
                media_urls=media_urls,
                media_types=media_types,
                reply_to_message_id=reply_to_guid,
            )

        except Exception as e:
            logger.error("[iMessage] Error building event: %s", e, exc_info=True)
            return None

    async def _download_attachment(
        self, att_guid: str, att_meta: Dict[str, Any]
    ) -> Optional[str]:
        """Download an attachment from BlueBubbles to the local cache."""
        try:
            encoded = url_quote(att_guid, safe="")
            resp = await self._http_client.get(
                self._api_url(f"/attachment/{encoded}/download"),
                timeout=60,
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.content

            mime = att_meta.get("mimeType", "") or ""
            transfer_name = att_meta.get("transferName", "")

            if mime.startswith("image/"):
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/gif": ".gif",
                    "image/webp": ".webp",
                    "image/heic": ".jpg",  # BlueBubbles auto-converts HEIC
                }
                ext = ext_map.get(mime, ".jpg")
                cached = cache_image_from_bytes(data, ext)
                logger.info("[iMessage] Cached image: %s", cached)
                return cached

            elif mime.startswith("audio/"):
                ext_map = {
                    "audio/mp3": ".mp3",
                    "audio/mpeg": ".mp3",
                    "audio/ogg": ".ogg",
                    "audio/wav": ".wav",
                    "audio/x-caf": ".mp3",  # BlueBubbles converts CAF to MP3
                    "audio/mp4": ".m4a",
                    "audio/aac": ".m4a",
                }
                ext = ext_map.get(mime, ".mp3")
                cached = cache_audio_from_bytes(data, ext)
                logger.info("[iMessage] Cached audio: %s", cached)
                return cached

            else:
                # Documents, videos, etc.
                filename = transfer_name or f"file_{uuid.uuid4().hex[:8]}"
                cached = cache_document_from_bytes(data, filename)
                logger.info("[iMessage] Cached file: %s", cached)
                return cached

        except Exception as e:
            logger.warning("[iMessage] Failed to download attachment %s: %s", _redact(att_guid), e)
            return None
