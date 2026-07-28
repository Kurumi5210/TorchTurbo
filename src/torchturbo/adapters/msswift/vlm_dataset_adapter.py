"""VLM dataset adapter — pad media-less samples so every FSDP rank drives the media tower.

This exists ONLY because of FSDP. Under FSDP + expert-parallel, a multimodal
encoder/decoder tower (``model.visual`` and each of its blocks) is a world-mesh
FSDP2 unit sharded with ``reshard_after_forward=True``: entering it posts a per-layer
unshard all-gather on every rank. The model forward only runs that tower when its
media input is non-empty (``pixel_values is not None``), so in a **mixed multimodal
dataset** a rank whose micro-batch lands on media-less samples skips the tower and
posts NONE of those all-gathers, while ranks whose batch carries media post them →
per-layer all-gather mismatch → forward hang. The collectives can only stay paired if
every rank drives the tower on every batch — i.e. a media-less sample must still feed
the tower something. This adapter pads media-less samples so that happens.

The mechanism: in a multimodal dataset, pad media-less samples with media so every
rank produces non-empty media inputs and the tower runs on all ranks. It is
media-agnostic: the current implementation pads with an *image* for qwen3.5-moe's
tower, but the same pattern will apply to future multimodal models whose towers
consume video/audio/etc. — adding a new medium is just a new ``_build_pad_media``
branch, not a new module.

Detection (no CLI flag): the run is multimodal iff the dataset built by swift's
``RowPreprocessor`` carries an ``images``/``videos`` column — i.e. ``'images' in
dataset.features`` after preprocessing. We observe that once when
``RowPreprocessor.__call__`` returns the constructed dataset and record it in a
module-level flag; the per-row ``Template._preprocess_inputs`` wrapper then pads
media-less rows only when that flag is set. Pure-text datasets (no media column) →
flag stays False → never pads → zero overhead, byte-identical.

Why the swift template preprocess step: it wraps ``Template._preprocess_inputs``
*before* the media encoder runs. Setting ``inputs.images = [<fake image>]`` (with
nothing else) lets swift's own ``_add_default_tags`` prepend the ``<image>``
placeholder to the user message, and its qwen ``_encode`` run the image processor +
expand ``<|image_pad|>`` tokens to exactly ``prod(grid_thw) // merge_size**2`` — all
invariants (pixel_values↔image_grid_thw↔image-token-count↔mrope position rows) are
produced consistently by swift's existing machinery, with no turbo-side collator
surgery. Adding e.g. a video pad later would set ``inputs.videos`` analogously before
the template's media-encoder runs.

Note: unlike swift's native text-only ``_get_inputs_embeds_hf`` keep-alive (which
folds a zeroed ``×0`` contribution), this injects a *real* (small) attended fake
medium into media-less samples. Its media tokens get ``labels=-100`` (the ``<image>``
is prepended to the user message, which is not supervised), so they do not contribute
to the loss — but they ARE attended. This is the same tradeoff as LlamaFactory's
collator pad, and is the price of forcing the tower onto every rank through the data
stage under FSDP.
"""

from __future__ import annotations

import functools

from ...logging import get_logger

logger = get_logger(__name__)

# Module-level multimodal flag, set once during dataset construction. The per-row
# template wrapper reads this to decide whether to pad a media-less sample. Reset on
# uninstall.
_multimodal_dataset = False

# One shared small fake image; sized above the qwen2vl image processor's
# ``min_pixels`` floor (4 * patch_size**2 = 784 for patch_size=14) so it never trips a
# too-small rejection. 64x64 = 4096 px → grid_thw small (a few merged tokens), keeping
# the added vision compute negligible. Built lazily so PIL is only imported when a
# multimodal run actually needs it.
_FAKE_IMAGE = None
_FAKE_IMAGE_SIZE = (64, 64)


def _get_fake_image():
    global _FAKE_IMAGE
    if _FAKE_IMAGE is None:
        from PIL import Image

        _FAKE_IMAGE = Image.new("RGB", _FAKE_IMAGE_SIZE, (255, 255, 255))
    return _FAKE_IMAGE


def _is_multimodal_dataset(dataset) -> bool:
    """True iff swift's built dataset carries an ``images``/``videos`` column."""
    features = getattr(dataset, "features", None)
    if features is None:
        return False
    # Extend here (e.g. ``"audios"``) when a future multimodal model needs it.
    return "images" in features or "videos" in features


def _build_pad_media(inputs) -> None:
    """Pad a media-less sample with media so its tower runs.

    Currently image-only (qwen3.5-moe). Add per-medium branches here for future
    multimodal models whose towers consume video/audio/etc. — the structure of this
    adapter itself is already media-agnostic.
    """
    # Prefer image (the qwen3.5-moe tower); a future video-only model would set
    # inputs.videos here instead.
    if not inputs.images:
        inputs.images = [_get_fake_image()]


def install_vlm_dataset_adapter() -> bool:
    """Install the VLM dataset adapter — pad media-less samples so every rank runs the tower.

    Two coordinated wraps (installed once from :meth:`MSSwiftAdapter.activate`,
    before any dataset is built):

    1. ``RowPreprocessor.__call__`` — after swift builds the dataset, inspect its
       features and record (module-level) whether the run is multimodal (carries an
       ``images``/``videos`` column). This is the only viable signal: it is a
       dataset-level fact known before the per-row template path runs, so there is no
       order dependence and no need for a CLI flag.
    2. ``Template._preprocess_inputs`` — per row, only when the multimodal flag is
       set, if the sample is media-less, pad it via :func:`_build_pad_media`. Swift
       then tags the prompt with the media placeholder and encodes a matching
       non-empty media input + tokens via its existing path, so the media tower runs
       on this rank.

    Idempotent and reversible (see :func:`uninstall_vlm_dataset_adapter`). Returns
    ``True`` if installed (or already installed), ``False`` if swift isn't importable.
    """
    try:
        from swift.dataset.preprocessor.core import RowPreprocessor
        from swift.template.base import Template
    except Exception:
        logger.debug_rank0("swift RowPreprocessor/Template not importable; skipping VLM dataset adapter")
        return False

    # --- Wrap 1: dataset-level multimodal detection -------------------------
    orig_call = RowPreprocessor.__call__
    if not getattr(orig_call, "_turbo_vlm_adapter_detect_installed", False):
        @functools.wraps(orig_call)
        def _wrapped_call(self, dataset, *args, **kwargs):
            built = orig_call(self, dataset, *args, **kwargs)
            global _multimodal_dataset
            if not _multimodal_dataset and _is_multimodal_dataset(built):
                _multimodal_dataset = True
                logger.info_rank0(
                    "torchturbo: multimodal dataset detected (media column present) — "
                    "media-less samples are padded with fake media so every rank drives "
                    "the media tower under FSDP+expert-parallel (per-layer all-gather "
                    "pairing)."
                )
            return built

        _wrapped_call._turbo_vlm_adapter_detect_installed = True
        _wrapped_call._turbo_original = orig_call
        RowPreprocessor.__call__ = _wrapped_call

    # --- Wrap 2: per-row media-less padding ---------------------------------
    orig_ppi = Template._preprocess_inputs
    if not getattr(orig_ppi, "_turbo_vlm_adapter_pad_installed", False):
        def _wrapped_ppi(self, inputs):
            is_vlm_model = (
                getattr(self, "model_meta", None) is not None
                and getattr(self.model_meta, "is_multimodal", False)
            )
            sample_is_medialess = not inputs.images and not getattr(inputs, "videos", None) \
                and not getattr(inputs, "audios", None)
            if _multimodal_dataset and sample_is_medialess and is_vlm_model:
                try:
                    _build_pad_media(inputs)
                except Exception:
                    logger.debug_rank0("VLM dataset adapter pad skipped for a sample", exc_info=True)
            return orig_ppi(self, inputs)

        _wrapped_ppi._turbo_vlm_adapter_pad_installed = True
        _wrapped_ppi._turbo_original = orig_ppi
        Template._preprocess_inputs = _wrapped_ppi

    logger.info_rank0("torchturbo VLM dataset adapter installed")
    return True


def uninstall_vlm_dataset_adapter() -> None:
    """Revert the monkeypatches and reset state (for tests / deactivate)."""
    global _multimodal_dataset, _FAKE_IMAGE

    _multimodal_dataset = False
    _FAKE_IMAGE = None

    try:
        from swift.dataset.preprocessor.core import RowPreprocessor
    except Exception:
        pass
    else:
        cur = getattr(RowPreprocessor, "__call__", None)
        orig = getattr(cur, "_turbo_original", None)
        if orig is not None:
            RowPreprocessor.__call__ = orig

    try:
        from swift.template.base import Template
    except Exception:
        pass
    else:
        cur = getattr(Template, "_preprocess_inputs", None)
        orig = getattr(cur, "_turbo_original", None)
        if orig is not None:
            Template._preprocess_inputs = orig
