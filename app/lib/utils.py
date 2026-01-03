from fastapi.responses import StreamingResponse
from ai_sdk.generate_text import StreamTextResult
import json

def to_data_stream_response(result: StreamTextResult) -> StreamingResponse:
    """
    Automated helper to convert an AI SDK StreamTextResult into a FastAPI StreamingResponse
    following the Vercel AI Data Stream Protocol (v1).
    """
    
    async def data_stream_generator():
        async for text_part in result.text_stream:
            # 0 indicates a text part in the Data Stream Protocol
            yield f"0:{json.dumps(text_part)}\n"

    return StreamingResponse(
        data_stream_generator(), 
        media_type="text/plain", 
        headers={"X-Vercel-AI-Data-Stream": "v1"}
    )
