import base64
import json
import unittest
from array import array
from types import SimpleNamespace

from core.openai_realtime import (
    OpenAIRealtimeSession,
    _normalise_schema,
    _openai_tools,
)


class _FakeWebSocket:
    def __init__(self, events=None):
        self.events = list(events or [])
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        return json.dumps(self.events.pop(0))


def _session(**kwargs):
    return OpenAIRealtimeSession(
        api_key="test-key",
        instructions="You are ADA.",
        tool_declarations=[],
        **kwargs,
    )


class SchemaTests(unittest.TestCase):
    def test_normalises_nested_gemini_schema_types(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "items": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                }
            },
        }
        self.assertEqual(
            _normalise_schema(schema)["properties"]["items"],
            {"type": "array", "items": {"type": "string"}},
        )

    def test_converts_function_declarations(self):
        tools = _openai_tools(
            [
                {
                    "name": "weather",
                    "description": "Get weather.",
                    "parameters": {"type": "OBJECT", "properties": {}},
                }
            ]
        )
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["parameters"]["type"], "object")


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_text_and_image_as_a_conversation_item(self):
        session = _session()
        session._ws = _FakeWebSocket()

        await session.send_client_content(
            turns={
                "parts": [
                    {"text": "What is on screen?"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "aW1hZ2U=",
                        }
                    },
                ]
            }
        )

        self.assertEqual(session._ws.sent[0]["type"], "conversation.item.create")
        content = session._ws.sent[0]["item"]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "What is on screen?"})
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(session._ws.sent[1], {"type": "response.create"})

    async def test_streams_pcm_audio_as_base64(self):
        session = _session(input_sample_rate=24_000)
        session._ws = _FakeWebSocket()

        await session.send_realtime_input(media={"data": b"\x00\x01"})

        self.assertEqual(session._ws.sent[0]["type"], "input_audio_buffer.append")
        self.assertEqual(
            session._ws.sent[0]["audio"],
            base64.b64encode(b"\x00\x01").decode("ascii"),
        )

    def test_resamples_16khz_pcm_for_realtime(self):
        session = _session(input_sample_rate=16_000)
        source = array("h", range(160)).tobytes()

        converted = session._resample_pcm16(source)

        self.assertGreater(len(converted), len(source))

    async def test_returns_tool_calls_and_sends_tool_output(self):
        session = _session()
        session._ws = _FakeWebSocket(
            [
                {
                    "type": "response.done",
                    "response": {
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call-1",
                                "name": "weather",
                                "arguments": '{"city":"London"}',
                            }
                        ],
                    },
                }
            ]
        )

        event = await anext(session.receive())
        call = event.tool_call.function_calls[0]
        self.assertEqual(call.id, "call-1")
        self.assertEqual(call.args, {"city": "London"})

        await session.send_tool_response(
            function_responses=[
                SimpleNamespace(id="call-1", response={"result": "Cloudy"})
            ]
        )
        self.assertEqual(
            session._ws.sent[0]["item"]["type"], "function_call_output"
        )
        self.assertEqual(session._ws.sent[1], {"type": "response.create"})

    async def test_cancel_only_sends_during_an_active_response(self):
        session = _session()
        session._ws = _FakeWebSocket()

        await session.cancel_response()
        self.assertEqual(session._ws.sent, [])

        session._response_active = True
        await session.cancel_response()
        self.assertEqual(session._ws.sent, [{"type": "response.cancel"}])

    async def test_translates_audio_transcripts_and_turn_completion(self):
        audio = b"\x01\x02\x03\x04"
        session = _session()
        session._ws = _FakeWebSocket(
            [
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(audio).decode("ascii"),
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello ADA",
                },
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": "resp-1",
                    "delta": "Hello!",
                },
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp-1",
                        "status": "completed",
                        "output": [],
                    },
                },
            ]
        )
        stream = session.receive()

        self.assertEqual((await anext(stream)).data, audio)
        self.assertEqual(
            (await anext(stream)).server_content.input_transcription.text,
            "hello ADA",
        )
        self.assertEqual(
            (await anext(stream)).server_content.output_transcription.text,
            "Hello!",
        )
        self.assertTrue((await anext(stream)).server_content.turn_complete)


if __name__ == "__main__":
    unittest.main()
