import uuid
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from src.processor import download_image_from_link, execute

app = FastAPI()

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "healthy"}
    )

@app.post("/inference", status_code=status.HTTP_201_CREATED)
async def run_inference(payload: dict):
    """
    Run inference on the provided image.
    """
    
    image_reference = payload.get("link")
    process_reference = payload.get("process_id")

    if not image_reference:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Image link is required"}
        )
    
    if not process_reference:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Process ID is required"}
        )
    
    try:
        
        image_bytes = download_image_from_link(image_reference)
        _, result_json = execute(image_bytes)


        return {
                "inference_id":str(uuid.uuid4()),
                "process_id": process_reference,
                "image_link": image_reference,
                "result": result_json
            }

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )