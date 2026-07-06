from enum import Enum


class SegmentationModel(Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    SAM3 = "sam3"


class VLMModel(Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_LITE = "gemini-2.5-flash-lite"
    GEMINI_ROBOTICS_PREVIEW = "gemini-robotics-er-1.5-preview"
    INTERNVL3_5_2B = "InternVL3_5-2B"
    INTERNVL3_5_4B = "InternVL3_5-4B"
    INTERNVL3_5_8B = "InternVL3_5-8B"
    INTERNVL3_5_14B = "InternVL3_5-14B"
    INTERNVL3_2B = "InternVL3-2B"
    INTERNVL3_4B = "InternVL3-4B"
    INTERNVL3_8B = "InternVL3-8B"
    INTERNVL3_14B = "InternVL3-14B"
    GEMMA_3_4B_LOCAL = "gemma-3-4b-it-local"
    GEMMA_3_12B_LOCAL = "gemma-3-12b-it-local"
    GEMMA_3_27B_LOCAL = "gemma-3-27b-it-local"
    GEMMA_3_4B_API = "gemma-3-4b-it-api"
    GEMMA_3_12B_API = "gemma-3-12b-it-api"
    GEMMA_3_27B_API = "gemma-3-27b-it-api"
    LLAVA_7B = "llava-v1.6-mistral-7b-hf"


def is_google_api(model: VLMModel):
    return model in [
        VLMModel.GEMINI_2_5_FLASH,
        VLMModel.GEMINI_2_5_LITE,
        VLMModel.GEMINI_ROBOTICS_PREVIEW,
        VLMModel.GEMMA_3_4B_API,
        VLMModel.GEMMA_3_12B_API,
        VLMModel.GEMMA_3_27B_API,
    ]


def is_gemma_api(model: VLMModel):
    return model in [
        VLMModel.GEMMA_3_4B_API,
        VLMModel.GEMMA_3_12B_API,
        VLMModel.GEMMA_3_27B_API,
    ]
