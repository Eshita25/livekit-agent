import os 
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, AgentServer
from livekit.plugins import groq, silero
from livekit.agents import inference


# Load LIVEKIT_* and GROQ_API_KEY from .env
load_dotenv(".env")


class MyAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, concise voice assistant. "
                "Answer in short, clear sentences. "
                
            )
        )


# Create the AgentServer that LiveKit Cloud will talk to
server = AgentServer()


@server.rtc_session()
async def voice_agent(ctx: agents.JobContext):
    """
    Voice pipeline using Groq:
      - VAD: Silero (detect when user is speaking)
      - STT: Groq Whisper v3 Turbo (non-streaming)
      - LLM: Groq LLaMA 3.3 70B
      - TTS: Groq PlayAI TTS
    """

    # Load Silero VAD (required for non-streaming STT)
    vad = silero.VAD.load()

    session = AgentSession(
        # VAD + turn detection (important for Groq STT)
        turn_detection="vad",
        vad=vad,

        # Speech-to-text
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="en",
        ),

        # Language model
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",
        ),

        # Text-to-speech
       tts=inference.TTS(
    model="cartesia/sonic-3",
    voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    language="en",
)

    )

    # Start the agent in the LiveKit room
    await session.start(
        room=ctx.room,
        agent=MyAssistant(),
    )

    # Optional: greet user once at start
    await session.generate_reply(
        instructions="Talk in english.Greet the user casually."
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    agents.cli.run_app(server)


