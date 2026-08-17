"""OpenAI Realtime connector with a Gemini-Live-compatible session surface.

The rest of ADA talks to a small subset of the Google Gemini Live session API.
This adapter implements that same subset on top of OpenAI's Realtime WebSocket
protocol so the existing audio, tool, vision, dashboard, and UI paths can be
shared by both providers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from array import array
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import quote


class OpenAIRealtimeError(RuntimeError):
    """Raised when OpenAI returns a Realtime protocol error."""


@dataclass
class _Transcript:
    text: str


@dataclass
class _ServerContent:
    output_transcription: _Transcript | None = None
    input_transcription: _Transcript | None = None
    turn_complete: bool = False


@dataclass
class _FunctionCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ToolCall:
    function_calls: list[_FunctionCall]


@dataclass
class OpenAIRealtimeResponse:
    """Response shape consumed by ``JarvisLive._receive_audio``."""

    data: bytes | None = None
    server_content: _ServerContent | None = None
    tool_call: _ToolCall | None = None
    speech_started: bool = False


def _normalise_schema(value: Any) -> Any:
    """Convert Gemini's upper-case schema types to JSON Schema casing."""

    if isinstance(value, dict):
        result = {key: _normalise_schema(item) for key, item in value.items()}
        schema_type = result.get("type")
        if isinstance(schema_type, str):
            result["type"] = schema_type.lower()
        return result
    if isinstance(value, list):
        return [_normalise_schema(item) for item in value]
    return value


def _openai_tools(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": declaration["name"],
            "description": declaration.get("description", ""),
            "parameters": _normalise_schema(
                declaration.get("parameters", {"type": "object", "properties": {}})
            ),
        }
        for declaration in declarations
    ]


class OpenAIRealtimeSession:
    """Async OpenAI Realtime session used as a drop-in ADA live connector."""

    provider = "openai"

    @property
    def response_active(self) -> bool:
        return self._response_active

    def __init__(
        self,
        *,
        api_key: str,
        instructions: str,
        tool_declarations: list[dict[str, Any]],
        model: str = "gpt-realtime-2.1",
        voice: str = "marin",
        input_sample_rate: int = 16_000,
        realtime_sample_rate: int = 24_000,
        transcription_model: str = "gpt-live-transcribe",
    ) -> None:
        if not api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is required when ADA_AI_PROVIDER=openai."
            )
        self.api_key = api_key.strip()
        self.instructions = instructions
        self.tool_declarations = tool_declarations
        self.model = model.strip() or "gpt-realtime-2.1"
        self.voice = voice.strip() or "marin"
        self.input_sample_rate = input_sample_rate
        self.realtime_sample_rate = realtime_sample_rate
        self.transcription_model = transcription_model
        self._ws: Any = None
        self._send_lock = asyncio.Lock()
        self._output_transcript_started: set[str] = set()
        self._response_active = False
        self._resample_previous: int | None = None
        self._resample_position = 0.0

    async def __aenter__(self) -> "OpenAIRealtimeSession":
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Realtime requires the 'websockets' package. "
                "Run: pip install -r requirements.txt"
            ) from exc

        url = (
            "wss://api.openai.com/v1/realtime?model="
            + quote(self.model, safe="-._")
        )
        self._ws = await connect(
            url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=20,
            close_timeout=5,
            max_size=None,
        )

        # Wait for the protocol handshake before configuring the session. This
        # also surfaces authentication errors before ADA starts its task group.
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=20)
            event = json.loads(raw)
            if event.get("type") == "error":
                raise self._protocol_error(event)
            if event.get("type") == "session.created":
                break

        await self._send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "instructions": self.instructions,
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.realtime_sample_rate,
                            },
                            "transcription": {
                                "model": self.transcription_model,
                            },
                            "turn_detection": {"type": "semantic_vad"},
                        },
                        "output": {
                            "format": {"type": "audio/pcm"},
                            "voice": self.voice,
                        },
                    },
                    "tools": _openai_tools(self.tool_declarations),
                    "tool_choice": "auto",
                },
            }
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _send_json(self, event: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("OpenAI Realtime session is not connected.")
        async with self._send_lock:
            await self._ws.send(json.dumps(event, separators=(",", ":")))

    @staticmethod
    def _protocol_error(event: dict[str, Any]) -> OpenAIRealtimeError:
        error = event.get("error") or {}
        code = error.get("code") or error.get("type") or "realtime_error"
        message = error.get("message") or "Unknown OpenAI Realtime error"
        return OpenAIRealtimeError(f"OpenAI Realtime {code}: {message}")

    async def send_realtime_input(self, *, media: Any) -> None:
        data = (
            media.get("data")
            if isinstance(media, dict)
            else getattr(media, "data", None)
        )
        if not data:
            return
        if isinstance(data, str):
            data = data.encode("latin1")
        data = self._resample_pcm16(bytes(data))
        if not data:
            return
        await self._send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(data).decode("ascii"),
            }
        )

    def _resample_pcm16(self, data: bytes) -> bytes:
        """Statefully resample little-endian mono PCM16 for OpenAI Realtime."""

        if self.input_sample_rate == self.realtime_sample_rate:
            return data
        if self.input_sample_rate <= 0 or self.realtime_sample_rate <= 0:
            raise ValueError("Audio sample rates must be positive.")

        samples = array("h")
        samples.frombytes(data[: len(data) - (len(data) % 2)])
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return b""

        values = list(samples)
        if self._resample_previous is not None:
            values.insert(0, self._resample_previous)
        if len(values) < 2:
            self._resample_previous = values[-1]
            return b""

        step = self.input_sample_rate / self.realtime_sample_rate
        position = self._resample_position
        limit = len(values) - 1
        output = array("h")
        while position < limit:
            index = int(position)
            fraction = position - index
            value = values[index] + (values[index + 1] - values[index]) * fraction
            output.append(max(-32768, min(32767, round(value))))
            position += step

        self._resample_position = position - limit
        self._resample_previous = values[-1]
        if sys.byteorder != "little":
            output.byteswap()
        return output.tobytes()

    async def send_client_content(self, *, turns: Any, turn_complete: bool = True) -> None:
        """Send ADA text/image turns and explicitly request a response."""

        turn = turns[-1] if isinstance(turns, list) else turns
        parts = turn.get("parts", []) if isinstance(turn, dict) else []
        content: list[dict[str, Any]] = []

        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("text") is not None:
                content.append({"type": "input_text", "text": str(part["text"])})
                continue

            inline = part.get("inline_data") or part.get("inlineData")
            if not isinstance(inline, dict) or not inline.get("data"):
                continue
            mime_type = inline.get("mime_type") or inline.get("mimeType") or "image/jpeg"
            image_data = inline["data"]
            if isinstance(image_data, bytes):
                image_data = base64.b64encode(image_data).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image_data}",
                }
            )

        if not content:
            return

        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": content,
                },
            }
        )
        if turn_complete:
            await self._send_json({"type": "response.create"})

    async def send_tool_response(self, *, function_responses: list[Any]) -> None:
        for response in function_responses:
            call_id = getattr(response, "id", None)
            output = getattr(response, "response", None)
            if isinstance(response, dict):
                call_id = call_id or response.get("id")
                output = output if output is not None else response.get("response")
            if not call_id:
                continue
            await self._send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": str(call_id),
                        "output": json.dumps(output, default=str),
                    },
                }
            )
        await self._send_json({"type": "response.create"})

    async def cancel_response(self) -> None:
        """Cancel current generation when the user presses ADA's interrupt button."""

        if self._ws is not None and self._response_active:
            await self._send_json({"type": "response.cancel"})

    async def receive(self) -> AsyncIterator[OpenAIRealtimeResponse]:
        if self._ws is None:
            raise RuntimeError("OpenAI Realtime session is not connected.")

        while True:
            raw = await self._ws.recv()
            event = json.loads(raw)
            event_type = event.get("type", "")

            if event_type == "error":
                raise self._protocol_error(event)

            if event_type == "response.output_audio.delta":
                delta = event.get("delta")
                if delta:
                    yield OpenAIRealtimeResponse(data=base64.b64decode(delta))
                continue

            if event_type in (
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
            ):
                delta = event.get("delta", "")
                if delta:
                    response_id = str(event.get("response_id", ""))
                    if response_id:
                        self._output_transcript_started.add(response_id)
                    yield OpenAIRealtimeResponse(
                        server_content=_ServerContent(
                            output_transcription=_Transcript(delta)
                        )
                    )
                continue

            if event_type == "response.output_audio_transcript.done":
                response_id = str(event.get("response_id", ""))
                transcript = event.get("transcript", "")
                if transcript and response_id not in self._output_transcript_started:
                    yield OpenAIRealtimeResponse(
                        server_content=_ServerContent(
                            output_transcription=_Transcript(transcript)
                        )
                    )
                continue

            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                if transcript:
                    yield OpenAIRealtimeResponse(
                        server_content=_ServerContent(
                            input_transcription=_Transcript(transcript)
                        )
                    )
                continue

            if event_type == "input_audio_buffer.speech_started":
                yield OpenAIRealtimeResponse(speech_started=True)
                continue

            if event_type == "response.created":
                self._response_active = True
                continue

            if event_type != "response.done":
                continue

            response = event.get("response") or {}
            self._response_active = False
            function_calls: list[_FunctionCall] = []
            for item in response.get("output") or []:
                if item.get("type") != "function_call":
                    continue
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except (TypeError, json.JSONDecodeError):
                    arguments = {}
                function_calls.append(
                    _FunctionCall(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        args=arguments if isinstance(arguments, dict) else {},
                    )
                )

            if function_calls:
                yield OpenAIRealtimeResponse(
                    tool_call=_ToolCall(function_calls=function_calls)
                )
            else:
                response_id = str(response.get("id", ""))
                if response_id:
                    self._output_transcript_started.discard(response_id)
                yield OpenAIRealtimeResponse(
                    server_content=_ServerContent(turn_complete=True)
                )
