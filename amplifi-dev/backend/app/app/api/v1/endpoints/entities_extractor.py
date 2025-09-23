import gc
import json
import os
import tempfile
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from jinja2 import Template
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.be_core.config import settings
from app.be_core.logger import logger
from app.schemas.image_processing_resp_schema import IEntitiesExtractionResponse
from app.schemas.response_schema import IPostResponseBase, create_response
from app.schemas.user_schema import UserData
from app.utils.document_processing_utils import (
    process_pdf_file_to_extract_data,
)
from app.utils.image_processing_utils import (
    optimize_image_for_api,
    process_image_file_to_extract_data,
)
from app.utils.openai_utils import get_openai_client

router = APIRouter()


def _check_valid_file_type(file_extension: str, content_type: str) -> bool:
    valid_extensions = settings.ALLOWED_IMAGE_PROCESSING_EXTENSIONS
    valid_mimetypes = settings.ALLOWED_IMAGE_PROCESSING_MIME_TYPES
    if file_extension not in valid_extensions:
        return False
    if content_type not in valid_mimetypes:
        return False

    return True


def _is_image_file(mimetype: str) -> bool:
    """Check if a file is an image based on mimetype and extension"""
    # Supported MIME types
    supported_mimetypes = {"image/jpeg", "image/jpg", "image/png"}

    # Check MIME type first (exact match)
    if mimetype and mimetype in supported_mimetypes:
        return True

    return False


def _is_pdf_file(mimetype: str) -> bool:
    """Check if a file is a PDF based on mimetype and extension"""
    supported_mimetypes = {"application/pdf"}
    if mimetype and mimetype in supported_mimetypes:
        return True

    return False


@lru_cache(maxsize=10)
def _get_api_config():
    """Get API configuration with caching to avoid repeated lookups"""
    return {
        "model": getattr(settings, "OPENAI_MODEL", "gpt-4o"),
        "max_tokens": getattr(settings, "OPENAI_MAX_TOKENS", 1000),
        "temperature": getattr(settings, "OPENAI_TEMPERATURE", 0.0),
    }


# Dynamic template for analysis prompt
def create_analysis_prompt_template(
    schemas: Dict[str, Any], example_format: Optional[Dict[str, Any]] = None
) -> str:
    """Create a dynamic analysis prompt template based on provided schemas and example format"""

    # Convert examples to JSON strings for template
    single_example = (
        str(example_format.get("single_entity", {})) if example_format else "{}"
    )
    multiple_example = (
        str(example_format.get("multiple_entities", {})) if example_format else "{}"
    )

    return f"""
OCR Text: {{{{ ocr_text }}}}
Schemas: {{{{ schemas }}}}

Based on the extracted OCR text, please figure out the domain entities in json format as per the given schemas.
- If the OCR text does not contain any domain entities, return an empty json object {{}}.
- If the OCR text contains domain entities, return the json object with the domain entities.
- If the OCR text contains a domain entity that is not in the given schemas, do not include it in the json object.
- If the OCR text contains words or phrases which do not belong to any domain entity, do not include them in the json object.
- IMPORTANT: Return ONLY valid JSON format. Do not include any additional text, explanations, or markdown formatting.
- The response should be a valid JSON object that can be parsed by json.loads().

Example format for single entity:
{single_example}

Example format for multiple entities:
{multiple_example}

Finally, if any values are estimated rather than explicitly shown, note that clearly at the end.
"""


def _extract_domain_data(
    prompt: str, image_path: str, filename: str, ocr_text: str, attach_image: bool
) -> str:
    """Analyze image using AI to get a comprehensive domain data"""
    try:
        if attach_image:
            # Use optimized encoding for memory efficiency
            encoded_image = optimize_image_for_api(image_path)

        # Get OpenAI client
        client = get_openai_client()

        # Get API configuration
        api_config = _get_api_config()

        # Call GPT-4o with vision capabilities
        messages = [
            {
                "role": "system",
                "content": "You are an expert at figuring out domain entities from the given OCR text.",
            },
        ]
        if attach_image:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            )

        response = client.chat.completions.create(
            model=api_config["model"],
            messages=messages,
            max_tokens=api_config["max_tokens"],
            temperature=api_config["temperature"],
        )

        domain_data = response.choices[0].message.content.strip()
        logger.info(f"Successfully extracted domain data ({len(domain_data)} chars)")
        logger.info(f"Domain data: {domain_data}")

        # Free memory after API call
        if attach_image:
            encoded_image = None
            gc.collect()

        return domain_data

    except Exception as e:
        logger.error(f"Error analyzing image description: {str(e)}")
        return f"Error analyzing image: {str(e)}"


def _safe_parse_json(json_string: str) -> Dict[str, Any]:
    """Safely parse JSON string and return a dictionary"""
    try:
        # Try to parse the JSON string
        parsed_data = json.loads(json_string)

        # Ensure it's a dictionary
        if isinstance(parsed_data, dict):
            return parsed_data
        else:
            logger.warning(f"Parsed JSON is not a dictionary: {type(parsed_data)}")
            return {"error": "Invalid response format", "raw_data": json_string}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from OpenAI response: {e}")
        logger.error(f"Raw response: {json_string}")
        return {"error": "Failed to parse response", "raw_data": json_string}
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return {"error": "Unexpected parsing error", "raw_data": json_string}


async def extract_entities_of_type(
    file: UploadFile, prompt_template: str, schemas: Dict[str, Any]
) -> IPostResponseBase[IEntitiesExtractionResponse]:
    extension = file.filename.split(".")[-1]
    content_type = file.content_type
    if not _check_valid_file_type(file_extension=extension, content_type=content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_FILE_EXTENSIONS}",
        )

    # Write the file to a temporary location using tempfile module
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{file.filename}"
    ) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        # Process the file
        if _is_image_file(content_type):
            ocr_text = process_image_file_to_extract_data(temp_file_path)
        elif _is_pdf_file(content_type):
            ocr_text = process_pdf_file_to_extract_data(temp_file_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {settings.ALLOWED_FILE_EXTENSIONS}",
            )

        attach_image = False
        if ocr_text is None or ocr_text == "":
            attach_image = True

        # Render the template with the extracted OCR text
        prompt = Template(prompt_template).render(ocr_text=ocr_text, schemas=schemas)
        domain_data = _extract_domain_data(
            prompt, temp_file_path, file.filename, ocr_text, attach_image
        )

        # Check if domain_data contains an error
        if domain_data.startswith("Error analyzing image:"):
            logger.error(f"Domain data extraction failed: {domain_data}")
            return create_response(
                data=IEntitiesExtractionResponse(
                    extracted_data={"error": domain_data},
                ),
                message="File processed but domain extraction failed",
            )

        # Parse the domain data safely
        parsed_domain_data = _safe_parse_json(domain_data)

        return create_response(
            data=IEntitiesExtractionResponse(
                extracted_data=parsed_domain_data,
            ),
            message="File processed successfully",
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}",
        )
    finally:
        os.remove(temp_file_path)


@router.post(
    "/entities_extractor/extract",
    response_model=IPostResponseBase[IEntitiesExtractionResponse],
)
async def extract_entities(
    file: UploadFile,
    schemas: str = Form(..., description="JSON string containing the schemas"),
    example_format: Optional[str] = Form(
        None, description="JSON string containing the example format"
    ),
    current_user: UserData = Depends(
        deps.get_api_client_user(required_roles=["api_client"])
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Extract entities from file using provided schemas and example format.

    This endpoint supports API client authentication.
    """
    try:
        # Parse the JSON strings
        schemas_dict = json.loads(schemas)
        example_format_dict = json.loads(example_format) if example_format else None

        return await extract_entities_of_type(
            file,
            create_analysis_prompt_template(schemas_dict, example_format_dict),
            schemas_dict,
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON format in schemas or example_format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
