import aiohttp
import asyncio
from io import BytesIO
import torch
from PIL import Image
from transformers import AutoImageProcessor
from .model_loader import MODEL_PATH
# Initialize the image processor - using a processor compatible with Gemma models
image_processor = None

async def init_image_processor():
    """Initialize the image processor asynchronously once"""
    global image_processor
    if image_processor is None:
        try:
            # First try to use Gemma's own image processor if available
            image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        except:
            # Fallback to a generic vision transformer processor that works well with Gemma
            try:
                image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
            except:
                # Last resort fallback
                image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

async def process_image(image_data_or_url):
    """
    Process an image for multimodal model input
    
    Args:
        image_data_or_url: Either image binary data or a URL to an image
        
    Returns:
        Processed image embedding
    """
    # Make sure the processor is initialized
    global image_processor
    if image_processor is None:
        await init_image_processor()
        
    # Handle URL
    if isinstance(image_data_or_url, str):
        async with aiohttp.ClientSession() as session:
            async with session.get(image_data_or_url) as response:
                if response.status == 200:
                    image_data = await response.read()
                else:
                    raise ValueError(f"Failed to fetch image from URL: {response.status}")
    else:
        image_data = image_data_or_url
    
    # Open the image with PIL
    image = Image.open(BytesIO(image_data))
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Process the image
    processed_image = image_processor(image, return_tensors="pt")
    
    return processed_image
