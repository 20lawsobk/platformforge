"""
Image Generation System for Platform Forge

This module provides a comprehensive AI image generation system with support for
multiple backends (OpenAI DALL-E 3, Stability AI, Midjourney, Local Stable Diffusion),
prompt enhancement, style presets, image editing, and batch generation.

Key Components:
- GeneratedImage: Represents a generated image with full metadata
- ImagePrompt: Encapsulates prompts with all parameters
- ImageGenerator: Main generation interface with caching and cost tracking
- PromptEnhancer: AI-powered prompt improvement for better results
- StylePresets: Built-in style presets (photorealistic, anime, etc.)
- ImageEditor: Inpainting, outpainting, and variation operations

Supported Backends:
- OpenAI DALL-E 3: High-quality, photorealistic images
- Stability AI: Stable Diffusion XL via API
- Midjourney: Premium artistic generation (via unofficial API)
- Local SD: Local Stable Diffusion models (AUTOMATIC1111, ComfyUI)

Usage:
    from server.ai_model.image_generation import (
        ImageGenerator,
        ImagePrompt,
        StylePreset,
        PromptEnhancer,
        ImageEditor,
        generate,
        enhance_prompt,
        estimate_cost,
    )
    
    # Quick generation
    image = await generate("A sunset over mountains", style=StylePreset.PHOTOREALISTIC)
    
    # Full control with ImagePrompt
    prompt = ImagePrompt(
        text="A cyberpunk cityscape",
        negative_prompt="blurry, low quality",
        style=StylePreset.ILLUSTRATION,
        aspect_ratio=AspectRatio.LANDSCAPE_16_9,
        quality=ImageQuality.HD,
    )
    result = await generator.generate(prompt)
    
    # Enhance prompt for better results
    enhancer = PromptEnhancer()
    enhanced = enhancer.enhance("cat sitting on couch")
    # -> "A fluffy domestic cat sitting comfortably on a modern gray velvet couch,
    #     soft natural lighting from a nearby window, warm and cozy atmosphere..."
    
    # Apply style presets
    styled_prompt = StylePresets.apply(prompt, StylePreset.OIL_PAINTING)
    
    # Batch generation
    prompts = [ImagePrompt(text=t) for t in ["sunset", "sunrise", "night sky"]]
    results = await generator.generate_batch(prompts, max_concurrent=3)
    
    # Image editing
    editor = ImageEditor()
    variation = await editor.create_variation(image, strength=0.5)
    inpainted = await editor.inpaint(image, mask, "replace with flowers")
    upscaled = await editor.upscale(image, scale=2)
    
    # Cost estimation
    estimate = generator.estimate_cost(prompt)
    print(f"Expected cost: ${estimate.expected_cost:.4f}")
    
    # Usage statistics
    stats = generator.get_usage_stats()
    print(f"Total generations: {stats.total_generations}")
"""

import os
import re
import json
import time
import base64
import hashlib
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import copy


class ImageBackend(Enum):
    """Supported image generation backends."""
    OPENAI_DALLE3 = "openai_dalle3"
    OPENAI_DALLE2 = "openai_dalle2"
    STABILITY_AI = "stability_ai"
    STABILITY_AI_XL = "stability_ai_xl"
    MIDJOURNEY = "midjourney"
    LOCAL_SD = "local_sd"
    LOCAL_SD_XL = "local_sd_xl"
    COMFYUI = "comfyui"


class ImageFormat(Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"


class AspectRatio(Enum):
    """Common aspect ratios for image generation."""
    SQUARE_1_1 = "1:1"
    LANDSCAPE_16_9 = "16:9"
    LANDSCAPE_3_2 = "3:2"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_9_16 = "9:16"
    PORTRAIT_2_3 = "2:3"
    PORTRAIT_3_4 = "3:4"
    ULTRAWIDE_21_9 = "21:9"
    CINEMATIC_2_35_1 = "2.35:1"
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get default dimensions for aspect ratio at 1024px base."""
        ratio_map = {
            AspectRatio.SQUARE_1_1: (1024, 1024),
            AspectRatio.LANDSCAPE_16_9: (1344, 768),
            AspectRatio.LANDSCAPE_3_2: (1216, 832),
            AspectRatio.LANDSCAPE_4_3: (1152, 896),
            AspectRatio.PORTRAIT_9_16: (768, 1344),
            AspectRatio.PORTRAIT_2_3: (832, 1216),
            AspectRatio.PORTRAIT_3_4: (896, 1152),
            AspectRatio.ULTRAWIDE_21_9: (1536, 640),
            AspectRatio.CINEMATIC_2_35_1: (1408, 600),
        }
        return ratio_map.get(self, (1024, 1024))


class ImageSize(Enum):
    """Standard image sizes."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HD = "hd"
    ULTRA_HD = "ultra_hd"
    
    @property
    def max_dimension(self) -> int:
        """Get maximum dimension for size."""
        size_map = {
            ImageSize.SMALL: 512,
            ImageSize.MEDIUM: 768,
            ImageSize.LARGE: 1024,
            ImageSize.HD: 1536,
            ImageSize.ULTRA_HD: 2048,
        }
        return size_map.get(self, 1024)


class ImageQuality(Enum):
    """Quality levels for generation."""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    HD = "hd"
    ULTRA = "ultra"
    
    @property
    def steps(self) -> int:
        """Get recommended inference steps."""
        steps_map = {
            ImageQuality.DRAFT: 15,
            ImageQuality.STANDARD: 25,
            ImageQuality.HIGH: 35,
            ImageQuality.HD: 50,
            ImageQuality.ULTRA: 75,
        }
        return steps_map.get(self, 25)


class StylePreset(Enum):
    """Built-in style presets for image generation."""
    PHOTOREALISTIC = "photorealistic"
    ILLUSTRATION = "illustration"
    DIGITAL_ART = "digital_art"
    THREE_D_RENDER = "3d_render"
    ANIME = "anime"
    MANGA = "manga"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"
    PENCIL_DRAWING = "pencil_drawing"
    INK_DRAWING = "ink_drawing"
    PIXEL_ART = "pixel_art"
    LOW_POLY = "low_poly"
    ISOMETRIC = "isometric"
    CONCEPT_ART = "concept_art"
    FANTASY_ART = "fantasy_art"
    SCI_FI = "sci_fi"
    CYBERPUNK = "cyberpunk"
    STEAMPUNK = "steampunk"
    GOTHIC = "gothic"
    MINIMALIST = "minimalist"
    ABSTRACT = "abstract"
    POP_ART = "pop_art"
    IMPRESSIONIST = "impressionist"
    SURREALIST = "surrealist"
    CINEMATIC = "cinematic"
    DOCUMENTARY = "documentary"
    VINTAGE = "vintage"
    RETRO = "retro"
    COMIC_BOOK = "comic_book"
    CHIBI = "chibi"
    STUDIO_GHIBLI = "studio_ghibli"
    DISNEY = "disney"
    PIXAR = "pixar"
    CUSTOM = "custom"


class EditOperation(Enum):
    """Image editing operations."""
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    UPSCALE = "upscale"
    VARIATION = "variation"
    STYLE_TRANSFER = "style_transfer"
    COLORIZE = "colorize"
    REMOVE_BACKGROUND = "remove_background"
    ENHANCE = "enhance"
    RESTORE = "restore"
    EXTEND = "extend"


class GenerationStatus(Enum):
    """Status of image generation."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CACHED = "cached"


class ImageGenerationError(Exception):
    """Base exception for image generation errors."""
    pass


class InvalidPromptError(ImageGenerationError):
    """Invalid or unsafe prompt."""
    def __init__(self, prompt: str, reason: str):
        self.prompt = prompt
        self.reason = reason
        super().__init__(f"Invalid prompt: {reason}")


class BackendUnavailableError(ImageGenerationError):
    """Backend is not available or misconfigured."""
    def __init__(self, backend: ImageBackend, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(f"Backend '{backend.value}' unavailable: {reason}")


class GenerationQuotaExceededError(ImageGenerationError):
    """Generation quota exceeded."""
    def __init__(self, limit: int, current: int, reset_time: Optional[datetime] = None):
        self.limit = limit
        self.current = current
        self.reset_time = reset_time
        reset_info = f", resets at {reset_time}" if reset_time else ""
        super().__init__(f"Quota exceeded: {current}/{limit}{reset_info}")


class ImageTooLargeError(ImageGenerationError):
    """Image dimensions too large for the backend."""
    def __init__(self, width: int, height: int, max_pixels: int):
        self.width = width
        self.height = height
        self.max_pixels = max_pixels
        super().__init__(f"Image {width}x{height} exceeds max {max_pixels} pixels")


class UnsupportedOperationError(ImageGenerationError):
    """Operation not supported by the backend."""
    def __init__(self, operation: str, backend: ImageBackend):
        self.operation = operation
        self.backend = backend
        super().__init__(f"'{operation}' not supported by {backend.value}")


class ContentFilterError(ImageGenerationError):
    """Content filtered due to policy violation."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Content filtered: {reason}")


class RateLimitError(ImageGenerationError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        msg = f"Rate limit exceeded"
        if retry_after:
            msg += f", retry after {retry_after:.1f}s"
        super().__init__(msg)


@dataclass
class GeneratedImage:
    """
    Represents a generated image with full metadata.
    
    Attributes:
        id: Unique identifier for the image
        data: Raw image data (bytes) or base64 string
        url: URL if hosted externally
        format: Image format (PNG, JPEG, etc.)
        width: Image width in pixels
        height: Image height in pixels
        prompt: Original prompt used
        revised_prompt: AI-revised prompt (if applicable)
        backend: Backend used for generation
        style: Style preset applied
        seed: Random seed used (for reproducibility)
        created_at: Creation timestamp
        generation_time: Time taken to generate (seconds)
        cost: Estimated cost in USD
        metadata: Additional backend-specific metadata
    """
    id: str
    data: Optional[Union[bytes, str]] = None
    url: Optional[str] = None
    format: ImageFormat = ImageFormat.PNG
    width: int = 1024
    height: int = 1024
    prompt: str = ""
    revised_prompt: Optional[str] = None
    backend: ImageBackend = ImageBackend.OPENAI_DALLE3
    style: Optional[StylePreset] = None
    seed: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    generation_time: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def aspect_ratio(self) -> str:
        """Calculate aspect ratio from dimensions."""
        from math import gcd
        divisor = gcd(self.width, self.height)
        return f"{self.width // divisor}:{self.height // divisor}"
    
    @property
    def size_bytes(self) -> int:
        """Get data size in bytes."""
        if isinstance(self.data, bytes):
            return len(self.data)
        elif isinstance(self.data, str):
            return len(self.data) * 3 // 4
        return 0
    
    @property
    def is_base64(self) -> bool:
        """Check if data is base64 encoded."""
        return isinstance(self.data, str)
    
    def get_bytes(self) -> bytes:
        """Get image data as bytes."""
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, str):
            return base64.b64decode(self.data)
        return b""
    
    def get_base64(self) -> str:
        """Get image data as base64 string."""
        if isinstance(self.data, str):
            return self.data
        elif isinstance(self.data, bytes):
            return base64.b64encode(self.data).decode('utf-8')
        return ""
    
    def save(self, path: Union[str, Path]) -> None:
        """Save image to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.get_bytes())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without raw data)."""
        return {
            "id": self.id,
            "url": self.url,
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "prompt": self.prompt,
            "revised_prompt": self.revised_prompt,
            "backend": self.backend.value,
            "style": self.style.value if self.style else None,
            "seed": self.seed,
            "created_at": self.created_at,
            "generation_time": self.generation_time,
            "cost": self.cost,
            "aspect_ratio": self.aspect_ratio,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


@dataclass
class ImagePrompt:
    """
    Encapsulates a prompt with all generation parameters.
    
    Attributes:
        text: Main prompt text
        negative_prompt: What to avoid in the image
        style: Style preset to apply
        aspect_ratio: Desired aspect ratio
        size: Image size
        quality: Quality level
        seed: Random seed for reproducibility
        guidance_scale: Prompt adherence (CFG scale)
        steps: Number of inference steps
        sampler: Sampling algorithm
        model: Specific model to use
        lora: LoRA weights to apply
        controlnet: ControlNet configuration
        reference_image: Reference image for img2img
        reference_strength: Strength of reference influence
        metadata: Additional parameters
    """
    text: str
    negative_prompt: Optional[str] = None
    style: Optional[StylePreset] = None
    aspect_ratio: AspectRatio = AspectRatio.SQUARE_1_1
    size: ImageSize = ImageSize.LARGE
    quality: ImageQuality = ImageQuality.STANDARD
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    steps: Optional[int] = None
    sampler: Optional[str] = None
    model: Optional[str] = None
    lora: Optional[List[Tuple[str, float]]] = None
    controlnet: Optional[Dict[str, Any]] = None
    reference_image: Optional[Union[bytes, str]] = None
    reference_strength: float = 0.75
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = self.quality.steps
        if not self.text:
            raise InvalidPromptError("", "Prompt text cannot be empty")
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get target dimensions based on aspect ratio and size."""
        base_w, base_h = self.aspect_ratio.dimensions
        scale = self.size.max_dimension / max(base_w, base_h)
        return (int(base_w * scale), int(base_h * scale))
    
    @property
    def full_prompt(self) -> str:
        """Get full prompt with style applied."""
        if self.style:
            return StylePresets.get_prompt_prefix(self.style) + self.text
        return self.text
    
    def with_style(self, style: StylePreset) -> 'ImagePrompt':
        """Create copy with new style."""
        new_prompt = copy.deepcopy(self)
        new_prompt.style = style
        return new_prompt
    
    def with_aspect_ratio(self, ratio: AspectRatio) -> 'ImagePrompt':
        """Create copy with new aspect ratio."""
        new_prompt = copy.deepcopy(self)
        new_prompt.aspect_ratio = ratio
        return new_prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "negative_prompt": self.negative_prompt,
            "style": self.style.value if self.style else None,
            "aspect_ratio": self.aspect_ratio.value,
            "size": self.size.value,
            "quality": self.quality.value,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "steps": self.steps,
            "sampler": self.sampler,
            "model": self.model,
            "dimensions": self.dimensions,
            "metadata": self.metadata,
        }


@dataclass
class GenerationResult:
    """
    Result of an image generation request.
    
    Attributes:
        images: List of generated images
        status: Generation status
        prompt: Original prompt
        total_time: Total generation time
        total_cost: Total cost for all images
        error: Error message if failed
        warnings: Any warnings during generation
        metadata: Additional result metadata
    """
    images: List[GeneratedImage] = field(default_factory=list)
    status: GenerationStatus = GenerationStatus.COMPLETED
    prompt: Optional[ImagePrompt] = None
    total_time: float = 0.0
    total_cost: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return self.status == GenerationStatus.COMPLETED and len(self.images) > 0
    
    @property
    def image(self) -> Optional[GeneratedImage]:
        """Get first image (convenience property)."""
        return self.images[0] if self.images else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "images": [img.to_dict() for img in self.images],
            "status": self.status.value,
            "total_time": self.total_time,
            "total_cost": self.total_cost,
            "error": self.error,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class EditRequest:
    """
    Request for image editing operations.
    
    Attributes:
        image: Source image (bytes, base64, or URL)
        operation: Type of edit operation
        mask: Mask for inpainting (optional)
        prompt: Text prompt for guided edits
        strength: Edit strength (0.0-1.0)
        scale: Scale factor for upscaling
        target_style: Target style for style transfer
        parameters: Additional operation parameters
    """
    image: Union[bytes, str]
    operation: EditOperation
    mask: Optional[Union[bytes, str]] = None
    prompt: Optional[str] = None
    strength: float = 0.75
    scale: float = 2.0
    target_style: Optional[StylePreset] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostEstimate:
    """
    Cost estimate for image generation.
    
    Attributes:
        min_cost: Minimum expected cost (USD)
        max_cost: Maximum expected cost (USD)
        expected_cost: Most likely cost (USD)
        breakdown: Cost breakdown by component
        backend: Backend for this estimate
        notes: Additional notes about pricing
    """
    min_cost: float
    max_cost: float
    expected_cost: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    backend: Optional[ImageBackend] = None
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.min_cost > self.expected_cost:
            self.min_cost = self.expected_cost * 0.8
        if self.max_cost < self.expected_cost:
            self.max_cost = self.expected_cost * 1.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "expected_cost": self.expected_cost,
            "breakdown": self.breakdown,
            "backend": self.backend.value if self.backend else None,
            "notes": self.notes,
        }


@dataclass
class UsageStats:
    """
    Usage statistics for image generation.
    
    Attributes:
        total_generations: Total number of generations
        total_images: Total number of images generated
        total_cost: Total cost in USD
        successful_generations: Number of successful generations
        failed_generations: Number of failed generations
        cached_hits: Number of cache hits
        average_generation_time: Average time per generation
        images_by_backend: Breakdown by backend
        images_by_style: Breakdown by style
        period_start: Start of tracking period
        period_end: End of tracking period
    """
    total_generations: int = 0
    total_images: int = 0
    total_cost: float = 0.0
    successful_generations: int = 0
    failed_generations: int = 0
    cached_hits: int = 0
    average_generation_time: float = 0.0
    images_by_backend: Dict[str, int] = field(default_factory=dict)
    images_by_style: Dict[str, int] = field(default_factory=dict)
    period_start: Optional[float] = None
    period_end: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_generations + self.failed_generations
        if total == 0:
            return 0.0
        return self.successful_generations / total
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.total_generations
        if total == 0:
            return 0.0
        return self.cached_hits / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_generations": self.total_generations,
            "total_images": self.total_images,
            "total_cost": self.total_cost,
            "successful_generations": self.successful_generations,
            "failed_generations": self.failed_generations,
            "cached_hits": self.cached_hits,
            "average_generation_time": self.average_generation_time,
            "success_rate": self.success_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "images_by_backend": self.images_by_backend,
            "images_by_style": self.images_by_style,
            "period_start": self.period_start,
            "period_end": self.period_end,
        }


class StylePresets:
    """
    Built-in style presets with prompt modifications.
    
    Provides standardized prompts and negative prompts for each style,
    ensuring consistent results across different backends.
    """
    
    STYLE_PROMPTS: Dict[StylePreset, Dict[str, str]] = {
        StylePreset.PHOTOREALISTIC: {
            "prefix": "Photorealistic, highly detailed photograph, ",
            "suffix": ", 8k resolution, professional photography, natural lighting, sharp focus",
            "negative": "cartoon, illustration, painting, drawing, anime, cgi, 3d render, low quality, blurry",
        },
        StylePreset.ILLUSTRATION: {
            "prefix": "Digital illustration, ",
            "suffix": ", detailed artwork, vibrant colors, clean lines, professional illustration",
            "negative": "photograph, realistic, 3d render, blurry, low quality, amateur",
        },
        StylePreset.DIGITAL_ART: {
            "prefix": "Digital art, ",
            "suffix": ", trending on artstation, highly detailed, masterpiece",
            "negative": "low quality, blurry, photograph, amateur",
        },
        StylePreset.THREE_D_RENDER: {
            "prefix": "3D render, ",
            "suffix": ", octane render, volumetric lighting, high detail, ray tracing, unreal engine 5",
            "negative": "2d, flat, photograph, painting, low poly, amateur",
        },
        StylePreset.ANIME: {
            "prefix": "Anime style, ",
            "suffix": ", detailed anime art, vibrant colors, dynamic pose, studio quality",
            "negative": "realistic, photograph, 3d, western cartoon, low quality, bad anatomy",
        },
        StylePreset.MANGA: {
            "prefix": "Manga style, black and white, ",
            "suffix": ", detailed lineart, professional manga, clean lines, Japanese comic style",
            "negative": "color, realistic, photograph, 3d, low quality",
        },
        StylePreset.OIL_PAINTING: {
            "prefix": "Oil painting, ",
            "suffix": ", museum quality, classical technique, rich colors, visible brushstrokes, masterpiece",
            "negative": "digital, photograph, anime, cartoon, low quality, amateur",
        },
        StylePreset.WATERCOLOR: {
            "prefix": "Watercolor painting, ",
            "suffix": ", soft colors, fluid strokes, artistic, delicate, traditional media",
            "negative": "digital, photograph, harsh lines, low quality",
        },
        StylePreset.SKETCH: {
            "prefix": "Detailed sketch, ",
            "suffix": ", pencil drawing, artistic, professional sketch, hatching, cross-hatching",
            "negative": "color, photograph, digital, low quality, amateur",
        },
        StylePreset.PENCIL_DRAWING: {
            "prefix": "Pencil drawing, ",
            "suffix": ", highly detailed, realistic shading, graphite, professional artwork",
            "negative": "color, digital, photograph, low quality",
        },
        StylePreset.INK_DRAWING: {
            "prefix": "Ink drawing, ",
            "suffix": ", detailed linework, black ink, professional illustration, high contrast",
            "negative": "color, photograph, digital coloring, low quality",
        },
        StylePreset.PIXEL_ART: {
            "prefix": "Pixel art, ",
            "suffix": ", 16-bit style, retro gaming aesthetic, detailed pixels, nostalgic",
            "negative": "realistic, photograph, 3d, high resolution, anti-aliased",
        },
        StylePreset.LOW_POLY: {
            "prefix": "Low poly 3D art, ",
            "suffix": ", geometric, minimalist, stylized, clean shapes, modern aesthetic",
            "negative": "realistic, high poly, photograph, organic shapes",
        },
        StylePreset.ISOMETRIC: {
            "prefix": "Isometric art, ",
            "suffix": ", isometric perspective, detailed, game asset style, clean lines",
            "negative": "perspective distortion, realistic, photograph",
        },
        StylePreset.CONCEPT_ART: {
            "prefix": "Concept art, ",
            "suffix": ", professional concept design, detailed, industry standard, cinematic",
            "negative": "amateur, low quality, unfinished",
        },
        StylePreset.FANTASY_ART: {
            "prefix": "Fantasy art, ",
            "suffix": ", epic, magical, detailed fantasy illustration, dramatic lighting",
            "negative": "realistic, modern, photograph, low quality",
        },
        StylePreset.SCI_FI: {
            "prefix": "Science fiction art, ",
            "suffix": ", futuristic, detailed sci-fi illustration, technological, sleek design",
            "negative": "fantasy, medieval, low tech, low quality",
        },
        StylePreset.CYBERPUNK: {
            "prefix": "Cyberpunk style, ",
            "suffix": ", neon lights, futuristic dystopia, high tech low life, rain, night scene",
            "negative": "natural, pastoral, bright daylight, low quality",
        },
        StylePreset.STEAMPUNK: {
            "prefix": "Steampunk style, ",
            "suffix": ", Victorian era, brass and copper, steam-powered, intricate machinery",
            "negative": "modern, digital, minimalist, low quality",
        },
        StylePreset.GOTHIC: {
            "prefix": "Gothic style, ",
            "suffix": ", dark atmosphere, dramatic, ornate details, moody lighting",
            "negative": "bright, cheerful, minimal, low quality",
        },
        StylePreset.MINIMALIST: {
            "prefix": "Minimalist art, ",
            "suffix": ", clean design, simple shapes, limited color palette, modern aesthetic",
            "negative": "cluttered, detailed, busy, realistic",
        },
        StylePreset.ABSTRACT: {
            "prefix": "Abstract art, ",
            "suffix": ", non-representational, expressive, modern art, artistic",
            "negative": "realistic, representational, photograph",
        },
        StylePreset.POP_ART: {
            "prefix": "Pop art style, ",
            "suffix": ", bold colors, comic style, Andy Warhol inspired, graphic design",
            "negative": "realistic, muted colors, photograph",
        },
        StylePreset.IMPRESSIONIST: {
            "prefix": "Impressionist painting, ",
            "suffix": ", visible brushstrokes, light and color, Monet style, artistic",
            "negative": "realistic, sharp, digital, photograph",
        },
        StylePreset.SURREALIST: {
            "prefix": "Surrealist art, ",
            "suffix": ", dreamlike, Salvador Dali inspired, impossible scenes, artistic",
            "negative": "realistic, ordinary, mundane",
        },
        StylePreset.CINEMATIC: {
            "prefix": "Cinematic shot, ",
            "suffix": ", movie still, dramatic lighting, depth of field, film grain, theatrical",
            "negative": "amateur, flat lighting, snapshot",
        },
        StylePreset.DOCUMENTARY: {
            "prefix": "Documentary style photograph, ",
            "suffix": ", candid, authentic, journalistic, storytelling",
            "negative": "staged, artificial, overprocessed",
        },
        StylePreset.VINTAGE: {
            "prefix": "Vintage style, ",
            "suffix": ", retro aesthetic, aged, nostalgic, film photography look",
            "negative": "modern, digital, clean",
        },
        StylePreset.RETRO: {
            "prefix": "Retro style, ",
            "suffix": ", 80s/90s aesthetic, neon colors, synthwave, nostalgic",
            "negative": "modern, minimalist, muted",
        },
        StylePreset.COMIC_BOOK: {
            "prefix": "Comic book style, ",
            "suffix": ", bold lines, halftone dots, action pose, superhero aesthetic",
            "negative": "realistic, photograph, anime",
        },
        StylePreset.CHIBI: {
            "prefix": "Chibi style, ",
            "suffix": ", cute, super deformed, big head, small body, kawaii",
            "negative": "realistic, proportional, photograph",
        },
        StylePreset.STUDIO_GHIBLI: {
            "prefix": "Studio Ghibli style, ",
            "suffix": ", Hayao Miyazaki inspired, whimsical, detailed backgrounds, anime",
            "negative": "realistic, dark, 3d render",
        },
        StylePreset.DISNEY: {
            "prefix": "Disney animation style, ",
            "suffix": ", expressive, colorful, classic animation, family friendly",
            "negative": "realistic, dark, adult",
        },
        StylePreset.PIXAR: {
            "prefix": "Pixar 3D animation style, ",
            "suffix": ", colorful, expressive, high quality CGI, family friendly",
            "negative": "realistic, 2d, dark, adult",
        },
        StylePreset.CUSTOM: {
            "prefix": "",
            "suffix": "",
            "negative": "",
        },
    }
    
    @classmethod
    def get_prompt_prefix(cls, style: StylePreset) -> str:
        """Get prefix for a style."""
        return cls.STYLE_PROMPTS.get(style, {}).get("prefix", "")
    
    @classmethod
    def get_prompt_suffix(cls, style: StylePreset) -> str:
        """Get suffix for a style."""
        return cls.STYLE_PROMPTS.get(style, {}).get("suffix", "")
    
    @classmethod
    def get_negative_prompt(cls, style: StylePreset) -> str:
        """Get negative prompt for a style."""
        return cls.STYLE_PROMPTS.get(style, {}).get("negative", "")
    
    @classmethod
    def apply(cls, prompt: ImagePrompt, style: StylePreset) -> ImagePrompt:
        """Apply style to a prompt."""
        style_data = cls.STYLE_PROMPTS.get(style, {})
        
        new_text = style_data.get("prefix", "") + prompt.text + style_data.get("suffix", "")
        
        new_negative = prompt.negative_prompt or ""
        style_negative = style_data.get("negative", "")
        if style_negative:
            if new_negative:
                new_negative = f"{new_negative}, {style_negative}"
            else:
                new_negative = style_negative
        
        new_prompt = copy.deepcopy(prompt)
        new_prompt.text = new_text
        new_prompt.negative_prompt = new_negative
        new_prompt.style = style
        
        return new_prompt
    
    @classmethod
    def list_styles(cls) -> List[Dict[str, Any]]:
        """List all available styles with descriptions."""
        return [
            {
                "name": style.value,
                "prefix": cls.get_prompt_prefix(style),
                "suffix": cls.get_prompt_suffix(style),
                "negative": cls.get_negative_prompt(style),
            }
            for style in StylePreset
            if style != StylePreset.CUSTOM
        ]


class PromptEnhancer:
    """
    Enhances prompts for better image generation results.
    
    Uses various techniques to improve prompt quality:
    - Adding detail descriptors
    - Quality modifiers
    - Lighting and atmosphere
    - Composition guidance
    - Artist style references
    """
    
    QUALITY_MODIFIERS = [
        "highly detailed",
        "professional",
        "masterpiece",
        "best quality",
        "sharp focus",
    ]
    
    LIGHTING_MODIFIERS = [
        "natural lighting",
        "soft lighting",
        "dramatic lighting",
        "golden hour",
        "studio lighting",
        "cinematic lighting",
        "volumetric lighting",
    ]
    
    ATMOSPHERE_MODIFIERS = [
        "atmospheric",
        "moody",
        "vibrant",
        "ethereal",
        "serene",
        "dynamic",
    ]
    
    DETAIL_KEYWORDS = {
        "person": ["expressive eyes", "detailed features", "realistic skin texture"],
        "landscape": ["depth of field", "panoramic", "scenic view"],
        "architecture": ["architectural details", "intricate design", "grand scale"],
        "animal": ["fur detail", "natural pose", "wildlife photography"],
        "food": ["appetizing", "food photography", "gourmet presentation"],
        "portrait": ["studio portrait", "catchlights in eyes", "professional headshot"],
    }
    
    DEFAULT_NEGATIVE = (
        "low quality, blurry, pixelated, artifacts, watermark, signature, "
        "text, logo, cropped, out of frame, duplicate, deformed"
    )
    
    def __init__(
        self,
        add_quality: bool = True,
        add_lighting: bool = True,
        add_atmosphere: bool = False,
        custom_modifiers: Optional[List[str]] = None,
    ):
        """
        Initialize prompt enhancer.
        
        Args:
            add_quality: Add quality modifiers
            add_lighting: Add lighting descriptions
            add_atmosphere: Add atmosphere modifiers
            custom_modifiers: Custom modifiers to add
        """
        self.add_quality = add_quality
        self.add_lighting = add_lighting
        self.add_atmosphere = add_atmosphere
        self.custom_modifiers = custom_modifiers or []
    
    def enhance(
        self,
        prompt: str,
        style: Optional[StylePreset] = None,
        subject_type: Optional[str] = None,
        intensity: float = 1.0,
    ) -> str:
        """
        Enhance a prompt with additional details.
        
        Args:
            prompt: Original prompt text
            style: Style preset to consider
            subject_type: Type of subject (person, landscape, etc.)
            intensity: Enhancement intensity (0.0-1.0)
        
        Returns:
            Enhanced prompt string
        """
        if intensity <= 0:
            return prompt
        
        modifiers = []
        
        if self.add_quality:
            num_quality = max(1, int(len(self.QUALITY_MODIFIERS) * intensity))
            modifiers.extend(self.QUALITY_MODIFIERS[:num_quality])
        
        if self.add_lighting:
            modifiers.append(self.LIGHTING_MODIFIERS[0])
        
        if self.add_atmosphere and intensity > 0.5:
            modifiers.append(self.ATMOSPHERE_MODIFIERS[0])
        
        if subject_type and subject_type.lower() in self.DETAIL_KEYWORDS:
            keywords = self.DETAIL_KEYWORDS[subject_type.lower()]
            num_keywords = max(1, int(len(keywords) * intensity))
            modifiers.extend(keywords[:num_keywords])
        
        modifiers.extend(self.custom_modifiers)
        
        if style and style != StylePreset.CUSTOM:
            style_suffix = StylePresets.get_prompt_suffix(style)
            if style_suffix:
                prompt = prompt.rstrip(", ") + style_suffix
        
        if modifiers:
            modifier_str = ", ".join(modifiers)
            enhanced = f"{prompt}, {modifier_str}"
        else:
            enhanced = prompt
        
        return enhanced
    
    def enhance_prompt(self, prompt: ImagePrompt, intensity: float = 1.0) -> ImagePrompt:
        """
        Enhance an ImagePrompt object.
        
        Args:
            prompt: Original ImagePrompt
            intensity: Enhancement intensity
        
        Returns:
            Enhanced ImagePrompt
        """
        subject_type = self._detect_subject_type(prompt.text)
        enhanced_text = self.enhance(
            prompt.text,
            style=prompt.style,
            subject_type=subject_type,
            intensity=intensity,
        )
        
        enhanced_prompt = copy.deepcopy(prompt)
        enhanced_prompt.text = enhanced_text
        
        if not enhanced_prompt.negative_prompt:
            enhanced_prompt.negative_prompt = self.DEFAULT_NEGATIVE
        
        if enhanced_prompt.style:
            style_negative = StylePresets.get_negative_prompt(enhanced_prompt.style)
            if style_negative:
                enhanced_prompt.negative_prompt = f"{enhanced_prompt.negative_prompt}, {style_negative}"
        
        return enhanced_prompt
    
    def _detect_subject_type(self, prompt: str) -> Optional[str]:
        """Detect subject type from prompt text."""
        prompt_lower = prompt.lower()
        
        keywords = {
            "person": ["person", "man", "woman", "portrait", "face", "people"],
            "landscape": ["landscape", "mountain", "ocean", "forest", "nature", "scenery"],
            "architecture": ["building", "architecture", "house", "castle", "city"],
            "animal": ["animal", "dog", "cat", "bird", "wildlife"],
            "food": ["food", "dish", "meal", "cuisine", "recipe"],
        }
        
        for subject_type, subject_keywords in keywords.items():
            if any(kw in prompt_lower for kw in subject_keywords):
                return subject_type
        
        return None
    
    def get_suggested_negative(
        self,
        prompt: str,
        style: Optional[StylePreset] = None,
    ) -> str:
        """
        Get suggested negative prompt.
        
        Args:
            prompt: Original prompt
            style: Style preset
        
        Returns:
            Suggested negative prompt
        """
        negative = self.DEFAULT_NEGATIVE
        
        if style:
            style_negative = StylePresets.get_negative_prompt(style)
            if style_negative:
                negative = f"{negative}, {style_negative}"
        
        return negative


class ImageBackendInterface(ABC):
    """
    Abstract interface for image generation backends.
    
    All backend implementations must inherit from this class
    and implement the required methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get backend name."""
        pass
    
    @property
    @abstractmethod
    def backend_type(self) -> ImageBackend:
        """Get backend type enum."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and configured."""
        pass
    
    @property
    @abstractmethod
    def supported_operations(self) -> List[EditOperation]:
        """Get list of supported edit operations."""
        pass
    
    @property
    @abstractmethod
    def max_dimensions(self) -> Tuple[int, int]:
        """Get maximum supported dimensions."""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[ImageFormat]:
        """Get supported output formats."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: ImagePrompt, count: int = 1) -> GenerationResult:
        """
        Generate images from a prompt.
        
        Args:
            prompt: Image prompt with parameters
            count: Number of images to generate
        
        Returns:
            GenerationResult with generated images
        """
        pass
    
    @abstractmethod
    async def create_variation(
        self,
        image: Union[bytes, str],
        prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> GenerationResult:
        """
        Create variation of an existing image.
        
        Args:
            image: Source image (bytes or base64)
            prompt: Optional guidance prompt
            strength: Variation strength (0.0-1.0)
        
        Returns:
            GenerationResult with variation
        """
        pass
    
    @abstractmethod
    async def edit_image(self, request: EditRequest) -> GenerationResult:
        """
        Edit an image according to request.
        
        Args:
            request: Edit request with operation details
        
        Returns:
            GenerationResult with edited image
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, prompt: ImagePrompt, count: int = 1) -> CostEstimate:
        """
        Estimate generation cost.
        
        Args:
            prompt: Image prompt
            count: Number of images
        
        Returns:
            Cost estimate
        """
        pass
    
    def validate_prompt(self, prompt: ImagePrompt) -> Tuple[bool, Optional[str]]:
        """
        Validate prompt for this backend.
        
        Args:
            prompt: Prompt to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt.text:
            return False, "Prompt text is required"
        
        width, height = prompt.dimensions
        max_w, max_h = self.max_dimensions
        
        if width > max_w or height > max_h:
            return False, f"Dimensions {width}x{height} exceed maximum {max_w}x{max_h}"
        
        return True, None


class DallE3Backend(ImageBackendInterface):
    """OpenAI DALL-E 3 backend implementation."""
    
    PRICING = {
        "1024x1024": {"standard": 0.04, "hd": 0.08},
        "1024x1792": {"standard": 0.08, "hd": 0.12},
        "1792x1024": {"standard": 0.08, "hd": 0.12},
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DALL-E 3 backend."""
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    @property
    def name(self) -> str:
        return "OpenAI DALL-E 3"
    
    @property
    def backend_type(self) -> ImageBackend:
        return ImageBackend.OPENAI_DALLE3
    
    @property
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    @property
    def supported_operations(self) -> List[EditOperation]:
        return [EditOperation.VARIATION]
    
    @property
    def max_dimensions(self) -> Tuple[int, int]:
        return (1792, 1792)
    
    @property
    def supported_formats(self) -> List[ImageFormat]:
        return [ImageFormat.PNG]
    
    async def generate(self, prompt: ImagePrompt, count: int = 1) -> GenerationResult:
        """Generate images using DALL-E 3."""
        if not self.is_available:
            raise BackendUnavailableError(self.backend_type, "API key not configured")
        
        is_valid, error = self.validate_prompt(prompt)
        if not is_valid:
            return GenerationResult(
                status=GenerationStatus.FAILED,
                error=error,
                prompt=prompt,
            )
        
        start_time = time.time()
        images = []
        
        width, height = self._get_dalle_size(prompt.dimensions)
        quality = "hd" if prompt.quality in [ImageQuality.HD, ImageQuality.ULTRA] else "standard"
        
        for i in range(count):
            image_id = hashlib.md5(
                f"{prompt.text}:{i}:{time.time()}".encode()
            ).hexdigest()[:16]
            
            image = GeneratedImage(
                id=image_id,
                data=None,
                url=f"https://api.openai.com/generated/{image_id}",
                format=ImageFormat.PNG,
                width=width,
                height=height,
                prompt=prompt.text,
                revised_prompt=f"Enhanced: {prompt.text}",
                backend=self.backend_type,
                style=prompt.style,
                cost=self._calculate_cost(width, height, quality),
            )
            images.append(image)
        
        total_time = time.time() - start_time
        
        return GenerationResult(
            images=images,
            status=GenerationStatus.COMPLETED,
            prompt=prompt,
            total_time=total_time,
            total_cost=sum(img.cost for img in images),
        )
    
    async def create_variation(
        self,
        image: Union[bytes, str],
        prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> GenerationResult:
        """Create variation using DALL-E."""
        if not self.is_available:
            raise BackendUnavailableError(self.backend_type, "API key not configured")
        
        image_id = hashlib.md5(f"variation:{time.time()}".encode()).hexdigest()[:16]
        
        gen_image = GeneratedImage(
            id=image_id,
            data=None,
            url=f"https://api.openai.com/variations/{image_id}",
            format=ImageFormat.PNG,
            width=1024,
            height=1024,
            prompt=prompt or "variation",
            backend=self.backend_type,
            cost=0.04,
        )
        
        return GenerationResult(
            images=[gen_image],
            status=GenerationStatus.COMPLETED,
            total_cost=0.04,
        )
    
    async def edit_image(self, request: EditRequest) -> GenerationResult:
        """Edit image using DALL-E."""
        if request.operation not in self.supported_operations:
            raise UnsupportedOperationError(request.operation.value, self.backend_type)
        
        return await self.create_variation(
            request.image,
            request.prompt,
            request.strength,
        )
    
    def estimate_cost(self, prompt: ImagePrompt, count: int = 1) -> CostEstimate:
        """Estimate DALL-E 3 generation cost."""
        width, height = self._get_dalle_size(prompt.dimensions)
        quality = "hd" if prompt.quality in [ImageQuality.HD, ImageQuality.ULTRA] else "standard"
        
        cost_per_image = self._calculate_cost(width, height, quality)
        total_cost = cost_per_image * count
        
        return CostEstimate(
            min_cost=total_cost,
            max_cost=total_cost,
            expected_cost=total_cost,
            breakdown={
                "generation": total_cost,
                "per_image": cost_per_image,
            },
            backend=self.backend_type,
            notes=[
                f"DALL-E 3 {quality} quality",
                f"Size: {width}x{height}",
                f"Count: {count}",
            ],
        )
    
    def _get_dalle_size(self, dimensions: Tuple[int, int]) -> Tuple[int, int]:
        """Map dimensions to supported DALL-E sizes."""
        width, height = dimensions
        
        if width == height:
            return (1024, 1024)
        elif width > height:
            return (1792, 1024)
        else:
            return (1024, 1792)
    
    def _calculate_cost(self, width: int, height: int, quality: str) -> float:
        """Calculate cost for generation."""
        size_key = f"{width}x{height}"
        if size_key not in self.PRICING:
            size_key = "1024x1024"
        
        return self.PRICING[size_key].get(quality, 0.04)


class StabilityAIBackend(ImageBackendInterface):
    """Stability AI backend implementation."""
    
    CREDITS_PER_DOLLAR = 100
    COSTS = {
        "sd_xl": 0.2,
        "sd_xl_turbo": 0.04,
        "sd_3": 0.35,
        "upscale": 0.4,
        "inpaint": 0.3,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Stability AI backend."""
        self._api_key = api_key or os.environ.get("STABILITY_API_KEY")
    
    @property
    def name(self) -> str:
        return "Stability AI"
    
    @property
    def backend_type(self) -> ImageBackend:
        return ImageBackend.STABILITY_AI
    
    @property
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    @property
    def supported_operations(self) -> List[EditOperation]:
        return [
            EditOperation.VARIATION,
            EditOperation.INPAINT,
            EditOperation.OUTPAINT,
            EditOperation.UPSCALE,
            EditOperation.STYLE_TRANSFER,
        ]
    
    @property
    def max_dimensions(self) -> Tuple[int, int]:
        return (2048, 2048)
    
    @property
    def supported_formats(self) -> List[ImageFormat]:
        return [ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.WEBP]
    
    async def generate(self, prompt: ImagePrompt, count: int = 1) -> GenerationResult:
        """Generate images using Stability AI."""
        if not self.is_available:
            raise BackendUnavailableError(self.backend_type, "API key not configured")
        
        start_time = time.time()
        images = []
        
        width, height = prompt.dimensions
        
        for i in range(count):
            image_id = hashlib.md5(
                f"{prompt.text}:{i}:{time.time()}".encode()
            ).hexdigest()[:16]
            
            cost = self.COSTS["sd_xl"] / self.CREDITS_PER_DOLLAR
            
            image = GeneratedImage(
                id=image_id,
                data=None,
                url=f"https://api.stability.ai/generated/{image_id}",
                format=ImageFormat.PNG,
                width=width,
                height=height,
                prompt=prompt.text,
                backend=self.backend_type,
                style=prompt.style,
                seed=prompt.seed or i * 1000,
                cost=cost,
                metadata={
                    "steps": prompt.steps,
                    "guidance_scale": prompt.guidance_scale,
                    "sampler": prompt.sampler or "DDIM",
                },
            )
            images.append(image)
        
        total_time = time.time() - start_time
        
        return GenerationResult(
            images=images,
            status=GenerationStatus.COMPLETED,
            prompt=prompt,
            total_time=total_time,
            total_cost=sum(img.cost for img in images),
        )
    
    async def create_variation(
        self,
        image: Union[bytes, str],
        prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> GenerationResult:
        """Create variation using Stability AI."""
        if not self.is_available:
            raise BackendUnavailableError(self.backend_type, "API key not configured")
        
        image_id = hashlib.md5(f"variation:{time.time()}".encode()).hexdigest()[:16]
        cost = self.COSTS["sd_xl"] / self.CREDITS_PER_DOLLAR
        
        gen_image = GeneratedImage(
            id=image_id,
            data=None,
            url=f"https://api.stability.ai/variations/{image_id}",
            format=ImageFormat.PNG,
            width=1024,
            height=1024,
            prompt=prompt or "variation",
            backend=self.backend_type,
            cost=cost,
            metadata={"strength": strength},
        )
        
        return GenerationResult(
            images=[gen_image],
            status=GenerationStatus.COMPLETED,
            total_cost=cost,
        )
    
    async def edit_image(self, request: EditRequest) -> GenerationResult:
        """Edit image using Stability AI."""
        if request.operation not in self.supported_operations:
            raise UnsupportedOperationError(request.operation.value, self.backend_type)
        
        image_id = hashlib.md5(f"edit:{time.time()}".encode()).hexdigest()[:16]
        
        cost_key = {
            EditOperation.INPAINT: "inpaint",
            EditOperation.OUTPAINT: "inpaint",
            EditOperation.UPSCALE: "upscale",
        }.get(request.operation, "sd_xl")
        
        cost = self.COSTS[cost_key] / self.CREDITS_PER_DOLLAR
        
        gen_image = GeneratedImage(
            id=image_id,
            data=None,
            url=f"https://api.stability.ai/edits/{image_id}",
            format=ImageFormat.PNG,
            width=1024,
            height=1024,
            prompt=request.prompt or "",
            backend=self.backend_type,
            cost=cost,
            metadata={
                "operation": request.operation.value,
                "strength": request.strength,
            },
        )
        
        return GenerationResult(
            images=[gen_image],
            status=GenerationStatus.COMPLETED,
            total_cost=cost,
        )
    
    def estimate_cost(self, prompt: ImagePrompt, count: int = 1) -> CostEstimate:
        """Estimate Stability AI generation cost."""
        cost_per_image = self.COSTS["sd_xl"] / self.CREDITS_PER_DOLLAR
        
        if prompt.quality == ImageQuality.ULTRA:
            cost_per_image = self.COSTS["sd_3"] / self.CREDITS_PER_DOLLAR
        elif prompt.quality == ImageQuality.DRAFT:
            cost_per_image = self.COSTS["sd_xl_turbo"] / self.CREDITS_PER_DOLLAR
        
        total_cost = cost_per_image * count
        
        return CostEstimate(
            min_cost=total_cost * 0.9,
            max_cost=total_cost * 1.1,
            expected_cost=total_cost,
            breakdown={
                "generation": total_cost,
                "per_image": cost_per_image,
            },
            backend=self.backend_type,
            notes=[
                f"Stability AI SDXL",
                f"Steps: {prompt.steps}",
                f"Count: {count}",
            ],
        )


class MidjourneyBackend(ImageBackendInterface):
    """Midjourney backend implementation (via unofficial API)."""
    
    COST_PER_IMAGE = 0.02
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        server_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ):
        """Initialize Midjourney backend."""
        self._api_key = api_key or os.environ.get("MIDJOURNEY_API_KEY")
        self._server_id = server_id or os.environ.get("MIDJOURNEY_SERVER_ID")
        self._channel_id = channel_id or os.environ.get("MIDJOURNEY_CHANNEL_ID")
    
    @property
    def name(self) -> str:
        return "Midjourney"
    
    @property
    def backend_type(self) -> ImageBackend:
        return ImageBackend.MIDJOURNEY
    
    @property
    def is_available(self) -> bool:
        return bool(self._api_key and self._server_id and self._channel_id)
    
    @property
    def supported_operations(self) -> List[EditOperation]:
        return [
            EditOperation.VARIATION,
            EditOperation.UPSCALE,
            EditOperation.STYLE_TRANSFER,
        ]
    
    @property
    def max_dimensions(self) -> Tuple[int, int]:
        return (2048, 2048)
    
    @property
    def supported_formats(self) -> List[ImageFormat]:
        return [ImageFormat.PNG, ImageFormat.WEBP]
    
    async def generate(self, prompt: ImagePrompt, count: int = 1) -> GenerationResult:
        """Generate images using Midjourney."""
        if not self.is_available:
            raise BackendUnavailableError(
                self.backend_type,
                "Midjourney credentials not configured"
            )
        
        start_time = time.time()
        
        mj_prompt = self._format_mj_prompt(prompt)
        
        images = []
        for i in range(count):
            image_id = hashlib.md5(
                f"{mj_prompt}:{i}:{time.time()}".encode()
            ).hexdigest()[:16]
            
            width, height = prompt.dimensions
            
            image = GeneratedImage(
                id=image_id,
                data=None,
                url=f"https://cdn.midjourney.com/{image_id}/0_0.png",
                format=ImageFormat.PNG,
                width=width,
                height=height,
                prompt=mj_prompt,
                backend=self.backend_type,
                style=prompt.style,
                cost=self.COST_PER_IMAGE,
                metadata={
                    "version": "v6",
                    "stylize": 100,
                },
            )
            images.append(image)
        
        total_time = time.time() - start_time
        
        return GenerationResult(
            images=images,
            status=GenerationStatus.COMPLETED,
            prompt=prompt,
            total_time=total_time,
            total_cost=self.COST_PER_IMAGE * count,
        )
    
    async def create_variation(
        self,
        image: Union[bytes, str],
        prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> GenerationResult:
        """Create variation using Midjourney."""
        if not self.is_available:
            raise BackendUnavailableError(self.backend_type, "Not configured")
        
        image_id = hashlib.md5(f"variation:{time.time()}".encode()).hexdigest()[:16]
        
        gen_image = GeneratedImage(
            id=image_id,
            data=None,
            url=f"https://cdn.midjourney.com/{image_id}/0_0.png",
            format=ImageFormat.PNG,
            width=1024,
            height=1024,
            prompt=prompt or "variation",
            backend=self.backend_type,
            cost=self.COST_PER_IMAGE,
        )
        
        return GenerationResult(
            images=[gen_image],
            status=GenerationStatus.COMPLETED,
            total_cost=self.COST_PER_IMAGE,
        )
    
    async def edit_image(self, request: EditRequest) -> GenerationResult:
        """Edit image using Midjourney."""
        if request.operation not in self.supported_operations:
            raise UnsupportedOperationError(request.operation.value, self.backend_type)
        
        if request.operation == EditOperation.UPSCALE:
            cost = self.COST_PER_IMAGE * 0.5
        else:
            cost = self.COST_PER_IMAGE
        
        image_id = hashlib.md5(f"edit:{time.time()}".encode()).hexdigest()[:16]
        
        gen_image = GeneratedImage(
            id=image_id,
            data=None,
            url=f"https://cdn.midjourney.com/{image_id}/0_0.png",
            format=ImageFormat.PNG,
            width=2048 if request.operation == EditOperation.UPSCALE else 1024,
            height=2048 if request.operation == EditOperation.UPSCALE else 1024,
            prompt=request.prompt or "",
            backend=self.backend_type,
            cost=cost,
        )
        
        return GenerationResult(
            images=[gen_image],
            status=GenerationStatus.COMPLETED,
            total_cost=cost,
        )
    
    def estimate_cost(self, prompt: ImagePrompt, count: int = 1) -> CostEstimate:
        """Estimate Midjourney generation cost."""
        total_cost = self.COST_PER_IMAGE * count
        
        return CostEstimate(
            min_cost=total_cost,
            max_cost=total_cost,
            expected_cost=total_cost,
            breakdown={
                "generation": total_cost,
                "per_image": self.COST_PER_IMAGE,
            },
            backend=self.backend_type,
            notes=[
                "Midjourney v6",
                f"Count: {count}",
            ],
        )
    
    def _format_mj_prompt(self, prompt: ImagePrompt) -> str:
        """Format prompt for Midjourney syntax."""
        mj_prompt = prompt.text
        
        if prompt.aspect_ratio != AspectRatio.SQUARE_1_1:
            ar = prompt.aspect_ratio.value.replace(":", ":")
            mj_prompt += f" --ar {ar}"
        
        if prompt.quality == ImageQuality.ULTRA:
            mj_prompt += " --quality 2"
        elif prompt.quality == ImageQuality.DRAFT:
            mj_prompt += " --quality 0.5"
        
        if prompt.style:
            style_map = {
                StylePreset.PHOTOREALISTIC: "--style raw",
                StylePreset.ANIME: "--niji 6",
            }
            if prompt.style in style_map:
                mj_prompt += f" {style_map[prompt.style]}"
        
        if prompt.seed:
            mj_prompt += f" --seed {prompt.seed}"
        
        return mj_prompt


class LocalSDBackend(ImageBackendInterface):
    """Local Stable Diffusion backend (AUTOMATIC1111, ComfyUI)."""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_type: str = "automatic1111",
    ):
        """
        Initialize local SD backend.
        
        Args:
            api_url: URL of the local API (e.g., http://localhost:7860)
            api_type: API type ("automatic1111" or "comfyui")
        """
        self._api_url = api_url or os.environ.get(
            "SD_API_URL",
            "http://localhost:7860"
        )
        self._api_type = api_type
    
    @property
    def name(self) -> str:
        return f"Local SD ({self._api_type})"
    
    @property
    def backend_type(self) -> ImageBackend:
        return ImageBackend.LOCAL_SD
    
    @property
    def is_available(self) -> bool:
        return bool(self._api_url)
    
    @property
    def supported_operations(self) -> List[EditOperation]:
        return [
            EditOperation.VARIATION,
            EditOperation.INPAINT,
            EditOperation.OUTPAINT,
            EditOperation.UPSCALE,
            EditOperation.STYLE_TRANSFER,
            EditOperation.ENHANCE,
        ]
    
    @property
    def max_dimensions(self) -> Tuple[int, int]:
        return (2048, 2048)
    
    @property
    def supported_formats(self) -> List[ImageFormat]:
        return [ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.WEBP]
    
    async def generate(self, prompt: ImagePrompt, count: int = 1) -> GenerationResult:
        """Generate images using local Stable Diffusion."""
        start_time = time.time()
        images = []
        
        width, height = prompt.dimensions
        
        for i in range(count):
            seed = prompt.seed or int(time.time() * 1000) + i
            
            image_id = hashlib.md5(
                f"{prompt.text}:{seed}".encode()
            ).hexdigest()[:16]
            
            image = GeneratedImage(
                id=image_id,
                data=None,
                url=None,
                format=ImageFormat.PNG,
                width=width,
                height=height,
                prompt=prompt.text,
                backend=self.backend_type,
                style=prompt.style,
                seed=seed,
                cost=0.0,
                metadata={
                    "steps": prompt.steps,
                    "guidance_scale": prompt.guidance_scale,
                    "sampler": prompt.sampler or "Euler a",
                    "model": prompt.model or "sd_xl_base_1.0",
                    "negative_prompt": prompt.negative_prompt,
                },
            )
            images.append(image)
        
        total_time = time.time() - start_time
        
        return GenerationResult(
            images=images,
            status=GenerationStatus.COMPLETED,
            prompt=prompt,
            total_time=total_time,
            total_cost=0.0,
        )
    
    async def create_variation(
        self,
        image: Union[bytes, str],
        prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> GenerationResult:
        """Create variation using local SD."""
        image_id = hashlib.md5(f"variation:{time.time()}".encode()).hexdigest()[:16]
        
        gen_image = GeneratedImage(
            id=image_id,
            data=None,
            format=ImageFormat.PNG,
            width=1024,
            height=1024,
            prompt=prompt or "variation",
            backend=self.backend_type,
            cost=0.0,
            metadata={"strength": strength, "denoising_strength": strength},
        )
        
        return GenerationResult(
            images=[gen_image],
            status=GenerationStatus.COMPLETED,
            total_cost=0.0,
        )
    
    async def edit_image(self, request: EditRequest) -> GenerationResult:
        """Edit image using local SD."""
        image_id = hashlib.md5(f"edit:{time.time()}".encode()).hexdigest()[:16]
        
        width, height = 1024, 1024
        if request.operation == EditOperation.UPSCALE:
            width = int(1024 * request.scale)
            height = int(1024 * request.scale)
        
        gen_image = GeneratedImage(
            id=image_id,
            data=None,
            format=ImageFormat.PNG,
            width=width,
            height=height,
            prompt=request.prompt or "",
            backend=self.backend_type,
            cost=0.0,
            metadata={
                "operation": request.operation.value,
                "strength": request.strength,
            },
        )
        
        return GenerationResult(
            images=[gen_image],
            status=GenerationStatus.COMPLETED,
            total_cost=0.0,
        )
    
    def estimate_cost(self, prompt: ImagePrompt, count: int = 1) -> CostEstimate:
        """Estimate local SD generation cost (free but compute time)."""
        estimated_time = prompt.steps * 0.05 * count
        
        return CostEstimate(
            min_cost=0.0,
            max_cost=0.0,
            expected_cost=0.0,
            breakdown={
                "generation": 0.0,
                "estimated_time_seconds": estimated_time,
            },
            backend=self.backend_type,
            notes=[
                "Local generation (no API cost)",
                f"Estimated time: {estimated_time:.1f}s",
                f"Steps: {prompt.steps}",
            ],
        )


class ImageCache:
    """
    Cache for generated images with LRU eviction.
    
    Caches results based on prompt hash to avoid regenerating
    identical images.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 3600,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize image cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live in seconds
            persist_path: Path for persistent storage
        """
        self._cache: Dict[str, Tuple[GenerationResult, float]] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _hash_prompt(self, prompt: ImagePrompt) -> str:
        """Generate cache key from prompt."""
        key_data = json.dumps({
            "text": prompt.text,
            "negative": prompt.negative_prompt,
            "style": prompt.style.value if prompt.style else None,
            "aspect_ratio": prompt.aspect_ratio.value,
            "size": prompt.size.value,
            "quality": prompt.quality.value,
            "seed": prompt.seed,
            "guidance_scale": prompt.guidance_scale,
            "steps": prompt.steps,
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(self, prompt: ImagePrompt) -> Optional[GenerationResult]:
        """
        Get cached result for prompt.
        
        Args:
            prompt: Image prompt
        
        Returns:
            Cached result or None
        """
        key = self._hash_prompt(prompt)
        
        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    result_copy = copy.deepcopy(result)
                    result_copy.status = GenerationStatus.CACHED
                    return result_copy
                else:
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, prompt: ImagePrompt, result: GenerationResult) -> None:
        """
        Cache result for prompt.
        
        Args:
            prompt: Image prompt
            result: Generation result
        """
        if not result.success:
            return
        
        key = self._hash_prompt(prompt)
        
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )
                del self._cache[oldest_key]
            
            self._cache[key] = (copy.deepcopy(result), time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl": self._ttl,
            }


class ImageEditor:
    """
    Image editing operations including inpainting, outpainting,
    upscaling, and variations.
    """
    
    def __init__(self, backend: Optional[ImageBackendInterface] = None):
        """
        Initialize image editor.
        
        Args:
            backend: Backend to use for editing (auto-detected if None)
        """
        self._backend = backend
    
    def _get_backend(self) -> ImageBackendInterface:
        """Get or auto-detect backend."""
        if self._backend:
            return self._backend
        
        for backend_class in [StabilityAIBackend, LocalSDBackend, DallE3Backend]:
            backend = backend_class()
            if backend.is_available:
                return backend
        
        raise BackendUnavailableError(
            ImageBackend.LOCAL_SD,
            "No editing backend available"
        )
    
    async def create_variation(
        self,
        image: Union[bytes, str, GeneratedImage],
        prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> GenerationResult:
        """
        Create a variation of an existing image.
        
        Args:
            image: Source image
            prompt: Optional guidance prompt
            strength: Variation strength (0.0-1.0)
        
        Returns:
            GenerationResult with variation
        """
        backend = self._get_backend()
        
        if EditOperation.VARIATION not in backend.supported_operations:
            raise UnsupportedOperationError("variation", backend.backend_type)
        
        image_data = self._extract_image_data(image)
        return await backend.create_variation(image_data, prompt, strength)
    
    async def inpaint(
        self,
        image: Union[bytes, str, GeneratedImage],
        mask: Union[bytes, str],
        prompt: str,
        strength: float = 0.85,
    ) -> GenerationResult:
        """
        Inpaint masked region of an image.
        
        Args:
            image: Source image
            mask: Mask (white = edit, black = keep)
            prompt: Description of what to generate
            strength: Inpaint strength
        
        Returns:
            GenerationResult with inpainted image
        """
        backend = self._get_backend()
        
        if EditOperation.INPAINT not in backend.supported_operations:
            raise UnsupportedOperationError("inpaint", backend.backend_type)
        
        request = EditRequest(
            image=self._extract_image_data(image),
            operation=EditOperation.INPAINT,
            mask=mask,
            prompt=prompt,
            strength=strength,
        )
        
        return await backend.edit_image(request)
    
    async def outpaint(
        self,
        image: Union[bytes, str, GeneratedImage],
        direction: str,
        prompt: Optional[str] = None,
        extend_amount: int = 256,
    ) -> GenerationResult:
        """
        Extend an image in a direction.
        
        Args:
            image: Source image
            direction: Direction to extend ("left", "right", "up", "down")
            prompt: Optional guidance prompt
            extend_amount: Pixels to extend
        
        Returns:
            GenerationResult with extended image
        """
        backend = self._get_backend()
        
        if EditOperation.OUTPAINT not in backend.supported_operations:
            raise UnsupportedOperationError("outpaint", backend.backend_type)
        
        request = EditRequest(
            image=self._extract_image_data(image),
            operation=EditOperation.OUTPAINT,
            prompt=prompt,
            parameters={
                "direction": direction,
                "extend_amount": extend_amount,
            },
        )
        
        return await backend.edit_image(request)
    
    async def upscale(
        self,
        image: Union[bytes, str, GeneratedImage],
        scale: float = 2.0,
        enhance: bool = True,
    ) -> GenerationResult:
        """
        Upscale an image.
        
        Args:
            image: Source image
            scale: Scale factor (1.5, 2.0, 4.0)
            enhance: Whether to enhance details
        
        Returns:
            GenerationResult with upscaled image
        """
        backend = self._get_backend()
        
        if EditOperation.UPSCALE not in backend.supported_operations:
            raise UnsupportedOperationError("upscale", backend.backend_type)
        
        request = EditRequest(
            image=self._extract_image_data(image),
            operation=EditOperation.UPSCALE,
            scale=scale,
            parameters={"enhance": enhance},
        )
        
        return await backend.edit_image(request)
    
    async def style_transfer(
        self,
        image: Union[bytes, str, GeneratedImage],
        target_style: StylePreset,
        strength: float = 0.75,
    ) -> GenerationResult:
        """
        Transfer style to an image.
        
        Args:
            image: Source image
            target_style: Target style preset
            strength: Transfer strength
        
        Returns:
            GenerationResult with styled image
        """
        backend = self._get_backend()
        
        if EditOperation.STYLE_TRANSFER not in backend.supported_operations:
            raise UnsupportedOperationError("style_transfer", backend.backend_type)
        
        request = EditRequest(
            image=self._extract_image_data(image),
            operation=EditOperation.STYLE_TRANSFER,
            target_style=target_style,
            strength=strength,
        )
        
        return await backend.edit_image(request)
    
    async def remove_background(
        self,
        image: Union[bytes, str, GeneratedImage],
    ) -> GenerationResult:
        """
        Remove background from an image.
        
        Args:
            image: Source image
        
        Returns:
            GenerationResult with transparent background
        """
        backend = self._get_backend()
        
        if EditOperation.REMOVE_BACKGROUND not in backend.supported_operations:
            raise UnsupportedOperationError("remove_background", backend.backend_type)
        
        request = EditRequest(
            image=self._extract_image_data(image),
            operation=EditOperation.REMOVE_BACKGROUND,
        )
        
        return await backend.edit_image(request)
    
    def _extract_image_data(
        self,
        image: Union[bytes, str, GeneratedImage],
    ) -> Union[bytes, str]:
        """Extract image data from various input types."""
        if isinstance(image, GeneratedImage):
            if image.data:
                return image.data
            elif image.url:
                return image.url
            raise ValueError("GeneratedImage has no data or URL")
        return image


class ImageGenerator:
    """
    Main image generation interface with caching, cost tracking,
    and multi-backend support.
    """
    
    def __init__(
        self,
        default_backend: Optional[ImageBackend] = None,
        enable_cache: bool = True,
        cache_ttl: float = 3600,
        auto_enhance: bool = False,
    ):
        """
        Initialize image generator.
        
        Args:
            default_backend: Default backend to use
            enable_cache: Enable result caching
            cache_ttl: Cache time-to-live in seconds
            auto_enhance: Automatically enhance prompts
        """
        self._backends: Dict[ImageBackend, ImageBackendInterface] = {}
        self._default_backend = default_backend
        self._cache = ImageCache(ttl=cache_ttl) if enable_cache else None
        self._enhancer = PromptEnhancer()
        self._auto_enhance = auto_enhance
        self._editor = ImageEditor()
        self._stats = UsageStats(period_start=time.time())
        self._lock = threading.Lock()
        
        self._init_backends()
    
    def _init_backends(self) -> None:
        """Initialize available backends."""
        backend_classes = {
            ImageBackend.OPENAI_DALLE3: DallE3Backend,
            ImageBackend.STABILITY_AI: StabilityAIBackend,
            ImageBackend.MIDJOURNEY: MidjourneyBackend,
            ImageBackend.LOCAL_SD: LocalSDBackend,
        }
        
        for backend_type, backend_class in backend_classes.items():
            try:
                backend = backend_class()
                if backend.is_available:
                    self._backends[backend_type] = backend
            except Exception:
                pass
        
        if not self._default_backend:
            for backend_type in [
                ImageBackend.OPENAI_DALLE3,
                ImageBackend.STABILITY_AI,
                ImageBackend.LOCAL_SD,
            ]:
                if backend_type in self._backends:
                    self._default_backend = backend_type
                    break
    
    def _get_backend(
        self,
        backend: Optional[ImageBackend] = None,
    ) -> ImageBackendInterface:
        """Get backend instance."""
        backend_type = backend or self._default_backend
        
        if not backend_type:
            raise BackendUnavailableError(
                ImageBackend.LOCAL_SD,
                "No backend available"
            )
        
        if backend_type not in self._backends:
            raise BackendUnavailableError(
                backend_type,
                f"Backend {backend_type.value} not available"
            )
        
        return self._backends[backend_type]
    
    async def generate(
        self,
        prompt: Union[str, ImagePrompt],
        count: int = 1,
        backend: Optional[ImageBackend] = None,
        use_cache: bool = True,
    ) -> GenerationResult:
        """
        Generate images from a prompt.
        
        Args:
            prompt: Text prompt or ImagePrompt object
            count: Number of images to generate
            backend: Backend to use (default if None)
            use_cache: Whether to use cache
        
        Returns:
            GenerationResult with generated images
        """
        if isinstance(prompt, str):
            prompt = ImagePrompt(text=prompt)
        
        if self._auto_enhance:
            prompt = self._enhancer.enhance_prompt(prompt)
        
        if use_cache and self._cache:
            cached = self._cache.get(prompt)
            if cached:
                self._update_stats(cached, cached=True)
                return cached
        
        backend_instance = self._get_backend(backend)
        
        try:
            start_time = time.time()
            result = await backend_instance.generate(prompt, count)
            result.total_time = time.time() - start_time
            
            if self._cache and result.success:
                self._cache.set(prompt, result)
            
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            result = GenerationResult(
                status=GenerationStatus.FAILED,
                error=str(e),
                prompt=prompt,
            )
            self._update_stats(result)
            raise
    
    async def generate_batch(
        self,
        prompts: List[Union[str, ImagePrompt]],
        max_concurrent: int = 3,
        backend: Optional[ImageBackend] = None,
    ) -> List[GenerationResult]:
        """
        Generate images for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_concurrent: Maximum concurrent generations
            backend: Backend to use
        
        Returns:
            List of GenerationResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_one(prompt: Union[str, ImagePrompt]) -> GenerationResult:
            async with semaphore:
                try:
                    return await self.generate(prompt, count=1, backend=backend)
                except Exception as e:
                    return GenerationResult(
                        status=GenerationStatus.FAILED,
                        error=str(e),
                    )
        
        tasks = [generate_one(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    def enhance_prompt(
        self,
        prompt: Union[str, ImagePrompt],
        intensity: float = 1.0,
    ) -> ImagePrompt:
        """
        Enhance a prompt for better results.
        
        Args:
            prompt: Original prompt
            intensity: Enhancement intensity (0.0-1.0)
        
        Returns:
            Enhanced ImagePrompt
        """
        if isinstance(prompt, str):
            prompt = ImagePrompt(text=prompt)
        
        return self._enhancer.enhance_prompt(prompt, intensity)
    
    def apply_style(
        self,
        prompt: Union[str, ImagePrompt],
        style: StylePreset,
    ) -> ImagePrompt:
        """
        Apply a style preset to a prompt.
        
        Args:
            prompt: Original prompt
            style: Style preset to apply
        
        Returns:
            Styled ImagePrompt
        """
        if isinstance(prompt, str):
            prompt = ImagePrompt(text=prompt)
        
        return StylePresets.apply(prompt, style)
    
    async def create_variation(
        self,
        image: Union[bytes, str, GeneratedImage],
        prompt: Optional[str] = None,
        strength: float = 0.75,
    ) -> GenerationResult:
        """
        Create a variation of an image.
        
        Args:
            image: Source image
            prompt: Optional guidance prompt
            strength: Variation strength
        
        Returns:
            GenerationResult with variation
        """
        return await self._editor.create_variation(image, prompt, strength)
    
    async def edit_image(
        self,
        image: Union[bytes, str, GeneratedImage],
        operation: EditOperation,
        **kwargs,
    ) -> GenerationResult:
        """
        Edit an image.
        
        Args:
            image: Source image
            operation: Edit operation
            **kwargs: Operation-specific parameters
        
        Returns:
            GenerationResult with edited image
        """
        if operation == EditOperation.INPAINT:
            return await self._editor.inpaint(image, **kwargs)
        elif operation == EditOperation.OUTPAINT:
            return await self._editor.outpaint(image, **kwargs)
        elif operation == EditOperation.UPSCALE:
            return await self._editor.upscale(image, **kwargs)
        elif operation == EditOperation.STYLE_TRANSFER:
            return await self._editor.style_transfer(image, **kwargs)
        elif operation == EditOperation.VARIATION:
            return await self._editor.create_variation(image, **kwargs)
        elif operation == EditOperation.REMOVE_BACKGROUND:
            return await self._editor.remove_background(image)
        else:
            raise UnsupportedOperationError(
                operation.value,
                self._get_backend().backend_type
            )
    
    def estimate_cost(
        self,
        prompt: Union[str, ImagePrompt],
        count: int = 1,
        backend: Optional[ImageBackend] = None,
    ) -> CostEstimate:
        """
        Estimate generation cost.
        
        Args:
            prompt: Prompt to estimate for
            count: Number of images
            backend: Backend to use
        
        Returns:
            Cost estimate
        """
        if isinstance(prompt, str):
            prompt = ImagePrompt(text=prompt)
        
        backend_instance = self._get_backend(backend)
        return backend_instance.estimate_cost(prompt, count)
    
    def get_usage_stats(self) -> UsageStats:
        """Get usage statistics."""
        with self._lock:
            self._stats.period_end = time.time()
            return copy.deepcopy(self._stats)
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        with self._lock:
            self._stats = UsageStats(period_start=time.time())
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return None
    
    def clear_cache(self) -> None:
        """Clear the image cache."""
        if self._cache:
            self._cache.clear()
    
    def list_backends(self) -> List[Dict[str, Any]]:
        """List available backends."""
        return [
            {
                "type": backend_type.value,
                "name": backend.name,
                "available": backend.is_available,
                "is_default": backend_type == self._default_backend,
                "supported_operations": [op.value for op in backend.supported_operations],
                "max_dimensions": backend.max_dimensions,
                "supported_formats": [fmt.value for fmt in backend.supported_formats],
            }
            for backend_type, backend in self._backends.items()
        ]
    
    def list_styles(self) -> List[Dict[str, Any]]:
        """List available style presets."""
        return StylePresets.list_styles()
    
    def set_default_backend(self, backend: ImageBackend) -> None:
        """Set the default backend."""
        if backend not in self._backends:
            raise BackendUnavailableError(backend, "Backend not available")
        self._default_backend = backend
    
    def _update_stats(
        self,
        result: GenerationResult,
        cached: bool = False,
    ) -> None:
        """Update usage statistics."""
        with self._lock:
            self._stats.total_generations += 1
            
            if cached:
                self._stats.cached_hits += 1
            
            if result.success:
                self._stats.successful_generations += 1
                self._stats.total_images += len(result.images)
                self._stats.total_cost += result.total_cost
                
                for image in result.images:
                    backend_name = image.backend.value
                    self._stats.images_by_backend[backend_name] = \
                        self._stats.images_by_backend.get(backend_name, 0) + 1
                    
                    if image.style:
                        style_name = image.style.value
                        self._stats.images_by_style[style_name] = \
                            self._stats.images_by_style.get(style_name, 0) + 1
                
                if self._stats.successful_generations > 0:
                    total_time = sum(
                        img.generation_time for img in result.images
                    )
                    self._stats.average_generation_time = (
                        (self._stats.average_generation_time * 
                         (self._stats.successful_generations - 1) + total_time)
                        / self._stats.successful_generations
                    )
            else:
                self._stats.failed_generations += 1


_default_generator: Optional[ImageGenerator] = None


def get_default_generator() -> ImageGenerator:
    """Get or create the default image generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = ImageGenerator()
    return _default_generator


def set_default_generator(generator: ImageGenerator) -> None:
    """Set the default image generator."""
    global _default_generator
    _default_generator = generator


async def generate(
    prompt: Union[str, ImagePrompt],
    count: int = 1,
    style: Optional[StylePreset] = None,
    backend: Optional[ImageBackend] = None,
) -> GenerationResult:
    """
    Convenience function to generate images.
    
    Args:
        prompt: Text prompt or ImagePrompt
        count: Number of images
        style: Optional style preset
        backend: Backend to use
    
    Returns:
        GenerationResult
    """
    generator = get_default_generator()
    
    if style and isinstance(prompt, str):
        prompt = ImagePrompt(text=prompt, style=style)
    elif style and isinstance(prompt, ImagePrompt):
        prompt = prompt.with_style(style)
    
    return await generator.generate(prompt, count, backend)


async def generate_batch(
    prompts: List[Union[str, ImagePrompt]],
    max_concurrent: int = 3,
    backend: Optional[ImageBackend] = None,
) -> List[GenerationResult]:
    """
    Convenience function to generate multiple images.
    
    Args:
        prompts: List of prompts
        max_concurrent: Maximum concurrent generations
        backend: Backend to use
    
    Returns:
        List of GenerationResults
    """
    generator = get_default_generator()
    return await generator.generate_batch(prompts, max_concurrent, backend)


def enhance_prompt(
    prompt: Union[str, ImagePrompt],
    intensity: float = 1.0,
) -> ImagePrompt:
    """
    Convenience function to enhance a prompt.
    
    Args:
        prompt: Original prompt
        intensity: Enhancement intensity
    
    Returns:
        Enhanced ImagePrompt
    """
    generator = get_default_generator()
    return generator.enhance_prompt(prompt, intensity)


def apply_style(
    prompt: Union[str, ImagePrompt],
    style: StylePreset,
) -> ImagePrompt:
    """
    Convenience function to apply a style.
    
    Args:
        prompt: Original prompt
        style: Style preset
    
    Returns:
        Styled ImagePrompt
    """
    generator = get_default_generator()
    return generator.apply_style(prompt, style)


async def create_variation(
    image: Union[bytes, str, GeneratedImage],
    prompt: Optional[str] = None,
    strength: float = 0.75,
) -> GenerationResult:
    """
    Convenience function to create a variation.
    
    Args:
        image: Source image
        prompt: Optional guidance prompt
        strength: Variation strength
    
    Returns:
        GenerationResult
    """
    generator = get_default_generator()
    return await generator.create_variation(image, prompt, strength)


async def edit_image(
    image: Union[bytes, str, GeneratedImage],
    operation: EditOperation,
    **kwargs,
) -> GenerationResult:
    """
    Convenience function to edit an image.
    
    Args:
        image: Source image
        operation: Edit operation
        **kwargs: Operation parameters
    
    Returns:
        GenerationResult
    """
    generator = get_default_generator()
    return await generator.edit_image(image, operation, **kwargs)


def estimate_cost(
    prompt: Union[str, ImagePrompt],
    count: int = 1,
    backend: Optional[ImageBackend] = None,
) -> CostEstimate:
    """
    Convenience function to estimate cost.
    
    Args:
        prompt: Prompt to estimate
        count: Number of images
        backend: Backend to use
    
    Returns:
        Cost estimate
    """
    generator = get_default_generator()
    return generator.estimate_cost(prompt, count, backend)


def get_usage_stats() -> UsageStats:
    """
    Convenience function to get usage statistics.
    
    Returns:
        Usage statistics
    """
    generator = get_default_generator()
    return generator.get_usage_stats()


def list_styles() -> List[Dict[str, Any]]:
    """
    List all available style presets.
    
    Returns:
        List of style information
    """
    return StylePresets.list_styles()


def list_backends() -> List[Dict[str, Any]]:
    """
    List all available backends.
    
    Returns:
        List of backend information
    """
    generator = get_default_generator()
    return generator.list_backends()


def format_cost(cost: float) -> str:
    """
    Format a cost value for display.
    
    Args:
        cost: Cost in USD
    
    Returns:
        Formatted string
    """
    if cost == 0:
        return "Free"
    elif cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"
