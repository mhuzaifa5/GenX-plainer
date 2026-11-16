from __future__ import annotations

import gc
import inspect
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from classifier import PneumoniaClassifier
from image_processing import load_image
from knowledge_retrieval import PneumoniaKnowledgeRetriever
from overlay import create_overlay, save_overlay
from prompt_packet import build_prompt_packet, save_prompt_packet
from qwen_client import query_qwen, save_qwen_result
from segmentation import ChestXraySegmenter
from session_management import complete_case, create_case
from transformer_worker import classify_and_attribute_transformer
from xai import PneumoniaAttributor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GenXplainer",
    page_icon="🫁",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_CNN_WEIGHTS = (
    BASE_DIR / "bestepoch_fasterrcnn_resnet50_fpn.pth"
)

DEFAULT_TRANSFORMER_MODEL = (
    BASE_DIR / "my_finetuned_model"
)

DEFAULT_KB = (
    BASE_DIR / "pneumonia_kb.csv"
)

DEFAULT_OUTPUT = (
    BASE_DIR / "outputs"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .chat-user {
        background-color: #eef6ff;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    .chat-assistant {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ============================================================
# HELPERS
# ============================================================

def safe_metadata(obj):
    try:
        return obj.metadata()
    except Exception:
        return str(obj)


def safe_as_dict(obj):
    try:
        return obj.as_dict()
    except Exception:
        if isinstance(obj, dict):
            return obj
        return {"value": str(obj)}


def clear_model_gradients(model) -> None:
    """Best-effort removal of parameter gradients before releasing a model."""
    if model is None:
        return

    # Classifier wrappers commonly expose the torch module as ``model``.
    torch_model = getattr(model, "model", model)
    try:
        zero_grad = getattr(torch_model, "zero_grad", None)
        if callable(zero_grad):
            try:
                zero_grad(set_to_none=True)
            except TypeError:
                zero_grad()
    except Exception:
        pass

    try:
        parameters = getattr(torch_model, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                if getattr(parameter, "grad", None) is not None:
                    parameter.grad = None
    except Exception:
        pass


def release_model_resources(*models) -> None:
    """Release model references/gradients and clear CUDA allocator state."""
    for model in models:
        clear_model_gradients(model)
    gc.collect()
    clear_cuda_cache()


def build_prompt_packet_compat(
    image,
    prediction,
    attribution,
    segmentation,
    knowledge,
    overlay_path,
):
    """Call the installed prompt-packet API without passing too many arguments.

    Supported signatures are the two versions used by this project:
      1. (image, prediction, attribution, segmentation, knowledge)
      2. (image, prediction, attribution, segmentation, overlay_path)

    This prevents the ``takes 5 positional arguments but 6 were given`` failure
    when app.py and prompt_packet.py are temporarily out of sync.
    """
    signature = inspect.signature(build_prompt_packet)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if has_varargs:
        return build_prompt_packet(
            image, prediction, attribution, segmentation, knowledge, overlay_path
        )

    if len(positional) < 4:
        raise TypeError(
            "build_prompt_packet() must accept at least 4 positional arguments: "
            "image, prediction, attribution, segmentation"
        )

    args = [image, prediction, attribution, segmentation]

    if len(positional) == 4:
        return build_prompt_packet(*args)

    fifth_name = positional[4].name.lower()
    if any(token in fifth_name for token in ("knowledge", "retriev", "kb", "evidence")):
        args.append(knowledge)
    elif "overlay" in fifth_name or "path" in fifth_name or "image" in fifth_name:
        args.append(overlay_path)
    else:
        # The current Streamlit app is the grounded-knowledge version, so
        # prefer knowledge when the parameter has a generic name.
        args.append(knowledge)

    return build_prompt_packet(*args)


def display_array_as_image(array, caption):
    if array is None:
        return

    arr = np.asarray(array)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        st.image(
            arr,
            caption=caption,
            clamp=True,
            use_container_width=True,
        )
        return

    if arr.ndim == 3:

        if (
            arr.shape[0] in (1, 3, 4)
            and arr.shape[-1] not in (1, 3, 4)
        ):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]

        st.image(
            arr,
            caption=caption,
            clamp=True,
            use_container_width=True,
        )


def display_captum_attribution(attribution_map):
    attr = np.asarray(
        attribution_map,
        dtype=np.float32,
    )

    attr = np.squeeze(attr)

    if attr.ndim != 2:
        raise ValueError(
            "Expected a 2D attribution map, "
            f"received shape {attr.shape}"
        )

    max_abs = np.max(np.abs(attr))

    if max_abs > 0:
        normalized = attr / max_abs
    else:
        normalized = attr

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        normalized,
        cmap="seismic",
        vmin=-1,
        vmax=1,
        interpolation="bilinear",
    )

    ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Attribution",
        rotation=270,
        labelpad=15,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


def save_uploaded_image(uploaded_file) -> Path:
    temp_dir = (
        Path(tempfile.gettempdir())
        / "genxplainer"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        temp_dir
        / uploaded_file.name
    )

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return image_path


def model_display_name(model_type: str) -> str:
    if model_type == "transformer":
        return "Deformable DETR Transformer"

    return "Faster R-CNN"


def clear_cuda_cache():
    """
    Best-effort cleanup in the Streamlit process.

    This does NOT free memory held by another process such as vLLM.
    """
    try:
        import gc
        import torch

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_model(
    image,
    model_type: str,
    model_path: Path,
    detection_threshold: float,
    ig_steps: int,
):
    """
    Unified model interface.

    Both CNN and transformer return:

        prediction
        attribution

    using the same ClassificationResult / AttributionResult
    interfaces.
    """

    if model_type == "transformer":

        prediction, attribution = (
            classify_and_attribute_transformer(
                image.path,
                model_path,
                detection_threshold=detection_threshold,
                n_steps=ig_steps,
            )
        )

        return prediction, attribution

    classifier = PneumoniaClassifier(
        model_path,
        model_type="cnn",
        detection_threshold=detection_threshold,
    )
    attributor = None

    try:
        prediction = classifier.predict(image)

        attributor = PneumoniaAttributor(
            classifier,
            n_steps=ig_steps,
        )

        attribution = attributor.attribute(
            image,
            prediction,
        )

        return prediction, attribution
    finally:
        # Integrated Gradients needs gradients during attribution, but the
        # model is not needed after the result maps have been produced.
        clear_model_gradients(classifier)
        clear_model_gradients(attributor)
        del attributor
        del classifier
        gc.collect()
        clear_cuda_cache()


def initialize_case_files(
    case_dir: Path,
    stored_image_name: str,
):
    """
    Ensure files required by chat.py exist.

    create_case() normally creates these, but this keeps the
    Streamlit application robust if the session manager changes.
    """

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_path = case_dir / "case.json"

    if not case_path.exists():
        case_path.write_text(
            json.dumps(
                {
                    "stored_image": stored_image_name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    history_path = case_dir / "chat_history.json"

    if not history_path.exists():
        history_path.write_text(
            json.dumps(
                {
                    "messages": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_initial_chat_response(case_dir: Path):
    path = case_dir / "qwen_response.json"

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_segmenter():
    return ChestXraySegmenter()


@st.cache_resource(show_spinner=False)
def get_retriever(kb_path):
    return PneumoniaKnowledgeRetriever(
        Path(kb_path)
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🫁 GenXplainer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Explainable AI for Pneumonia Detection from Chest X-rays
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.subheader("Detector")

    model_type = st.selectbox(
        "Model architecture",
        options=[
            "cnn",
            "transformer",
        ],
        format_func=model_display_name,
    )

    if model_type == "cnn":

        model_path_text = st.text_input(
            "Faster R-CNN weights",
            value=str(DEFAULT_CNN_WEIGHTS),
        )

        default_threshold = 0.5

    else:

        model_path_text = st.text_input(
            "Transformer model directory",
            value=str(DEFAULT_TRANSFORMER_MODEL),
        )

        default_threshold = 0.2

    detection_threshold = st.slider(
        "Detection threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.05,
    )

    st.subheader("Explainability")

    ig_steps = st.slider(
        "Integrated Gradients steps",
        min_value=2,
        max_value=100,
        value=20,
        step=1,
    )

    st.subheader("Knowledge")

    kb_path = st.text_input(
        "Knowledge base",
        value=str(DEFAULT_KB),
    )

    st.subheader("Output")

    output_root = st.text_input(
        "Output directory",
        value=str(DEFAULT_OUTPUT),
    )

    st.subheader("Qwen")

    qwen_base_url = st.text_input(
        "Qwen/vLLM URL",
        value=os.getenv(
            "QWEN_BASE_URL",
            "http://127.0.0.1:8080/v1",
        ),
    )

    qwen_model = st.text_input(
        "Qwen model",
        value=os.getenv(
            "QWEN_MODEL",
            "Qwen/Qwen3.5-9B",
        ),
    )

    qwen_max_tokens = st.number_input(
        "Qwen max tokens",
        min_value=128,
        max_value=4096,
        value=1200,
        step=128,
    )

    skip_qwen = st.checkbox(
        "Skip initial Qwen report",
        value=False,
    )

    st.divider()

    st.caption(
        f"Selected detector: {model_display_name(model_type)}"
    )


# ============================================================
# 1. UPLOAD IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Chest X-ray</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
    ],
)


# ============================================================
# SAVE UPLOAD / PREVIEW
# ============================================================

image_path = None

if uploaded_file is not None:

    image_path = save_uploaded_image(
        uploaded_file
    )

    preview = Image.open(
        image_path
    )

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            preview,
            caption="Input Chest X-ray",
            use_container_width=True,
        )

    with c2:

        st.markdown(
            "### Image Information"
        )

        st.write(
            f"**Filename:** `{uploaded_file.name}`"
        )

        st.write(
            f"**Dimensions:** "
            f"`{preview.width} × {preview.height}`"
        )

        st.write(
            f"**Format:** `{preview.format}`"
        )

        st.write(
            f"**Detector:** "
            f"`{model_display_name(model_type)}`"
        )

        st.write(
            f"**Threshold:** `{detection_threshold:.2f}`"
        )


# ============================================================
# 2. RUN ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">2. Run Analysis</div>',
    unsafe_allow_html=True,
)

run_analysis = st.button(
    "🚀 Analyze X-ray",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PIPELINE
# ============================================================

if run_analysis:

    if uploaded_file is None:
        st.warning(
            "Please upload a chest X-ray first."
        )
        st.stop()

    model_path = Path(
        model_path_text
    ).expanduser()

    kb = Path(
        kb_path
    ).expanduser()

    output_root_path = Path(
        output_root
    ).expanduser()

    if not model_path.exists():

        st.error(
            f"Model path not found:\n\n`{model_path}`"
        )

        st.stop()

    if not kb.exists():

        st.error(
            f"Knowledge base not found:\n\n`{kb}`"
        )

        st.stop()

    output_root_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress = st.progress(0)
    status = st.empty()

    try:

        # ====================================================
        # 1. IMAGE
        # ====================================================

        status.info(
            "[1/7] Loading and preprocessing image..."
        )

        progress.progress(5)

        image = load_image(
            image_path
        )

        # ====================================================
        # CREATE CASE
        # ====================================================

        case, stored_image = create_case(
            output_root_path,
            image.path,
            None,
        )

        image = replace(
            image,
            path=stored_image,
            image_id=case.case_id,
        )

        case_dir = case.case_dir

        initialize_case_files(
            case_dir,
            Path(stored_image).name,
        )

        # ====================================================
        # 2 + 3. MODEL + XAI
        # ====================================================

        status.info(
            "[2/7] Running "
            f"{model_display_name(model_type)}..."
        )

        progress.progress(15)

        prediction, attribution = run_model(
            image=image,
            model_type=model_type,
            model_path=model_path,
            detection_threshold=detection_threshold,
            ig_steps=ig_steps,
        )

        progress.progress(35)

        # XAI model resources are released inside run_model().
        # This extra cleanup is intentionally lightweight and does not clear
        # Streamlit-cached segmentation resources.
        gc.collect()
        clear_cuda_cache()

        # ====================================================
        # 4. SEGMENTATION
        # ====================================================

        status.info(
            "[4/7] Running anatomical segmentation..."
        )

        progress.progress(50)

        segmenter = get_segmenter()

        segmentation = (
            segmenter.segment_image_data(
                image
            )
        )

        # ====================================================
        # 5. KNOWLEDGE
        # ====================================================

        status.info(
            "[5/7] Retrieving grounded knowledge..."
        )

        progress.progress(62)

        retriever = get_retriever(
            str(kb)
        )

        knowledge = retriever.retrieve(
            prediction,
            attribution,
            segmentation,
        )

        # ====================================================
        # 6. OVERLAY
        # ====================================================

        status.info(
            "[6/7] Creating evidence overlay..."
        )

        progress.progress(72)

        overlay = create_overlay(
            image,
            prediction,
            segmentation,
            attribution,
        )

        overlay_path = save_overlay(
            overlay,
            case_dir
            / "attribution_prediction_segmentation_overlay.png",
        )

        # ====================================================
        # 7. PROMPT PACKET
        # ====================================================

        status.info(
            "[7/7] Building frozen Qwen prompt packet..."
        )

        progress.progress(80)

        prompt_packet = build_prompt_packet_compat(
            image=image,
            prediction=prediction,
            attribution=attribution,
            segmentation=segmentation,
            knowledge=knowledge,
            overlay_path=overlay_path,
        )

        prompt_artifacts = save_prompt_packet(
            prompt_packet,
            case_dir,
        )

        # ====================================================
        # INITIAL QWEN REPORT
        # ====================================================

        qwen_result = None
        qwen_artifacts = {}

        if not skip_qwen:

            status.info(
                "Generating grounded Qwen report..."
            )

            progress.progress(88)

            qwen_result = query_qwen(
                prompt_packet,
                base_url=qwen_base_url,
                model=qwen_model,
                max_tokens=int(
                    qwen_max_tokens
                ),
            )

            qwen_artifacts = (
                save_qwen_result(
                    qwen_result,
                    case_dir,
                )
            )

        # ====================================================
        # SEGMENTATION ARRAYS
        # ====================================================

        arrays = {

            "class_map":
                segmentation.class_map,

            "lung_mask":
                segmentation.lung_mask,

            "heart_mask":
                segmentation.heart_mask,

            "image_left_lung":
                segmentation.image_left_lung,

            "image_right_lung":
                segmentation.image_right_lung,

            "anatomical_right_lung":
                segmentation.anatomical_right_lung,

            "anatomical_left_lung":
                segmentation.anatomical_left_lung,
        }

        arrays.update(
            {
                f"region_{name}": mask
                for name, mask
                in segmentation.regions.items()
            }
        )

        arrays_path = (
            case_dir
            / "segmentation_arrays.npz"
        )

        np.savez_compressed(
            arrays_path,
            **arrays,
        )

        # ====================================================
        # ATTRIBUTION
        # ====================================================

        attribution_path = (
            case_dir
            / "integrated_gradients_attribution.npz"
        )

        np.savez_compressed(
            attribution_path,

            signed_map=
                attribution.signed_map,

            absolute_map=
                attribution.absolute_map,

            channel_attribution=
                attribution.channel_attribution,

            convergence_delta=
                attribution.convergence_delta,
        )

        # ====================================================
        # PIPELINE RECORD
        # ====================================================

        record = {

            "image":
                safe_metadata(image),

            "model": {
                "type": model_type,
                "name": model_display_name(
                    model_type
                ),
                "path": str(model_path),
                "detection_threshold":
                    detection_threshold,
            },

            "prediction":
                safe_as_dict(prediction),

            "segmentation":
                safe_metadata(segmentation),

            "attribution":
                safe_metadata(attribution),

            "knowledge":
                safe_as_dict(knowledge),

            "prompt_packet":
                safe_metadata(prompt_packet),

            "qwen":
                (
                    qwen_result.metadata()
                    if qwen_result is not None
                    else None
                ),

            "overlay": {

                "box_count":
                    overlay.box_count,

                "region_count":
                    overlay.region_count,

                "segmentation_status":
                    overlay.segmentation_status,

                "coordinate_space":
                    overlay.coordinate_space,

                "attribution_method":
                    overlay.attribution_method,
            },

            "artifacts": {

                "segmentation_arrays":
                    str(arrays_path),

                "integrated_gradients_attribution":
                    str(attribution_path),

                "attribution_prediction_segmentation_overlay":
                    str(overlay_path),

                "prompt_packet":
                    str(
                        prompt_artifacts[
                            "packet"
                        ]
                    ),

                **{
                    f"qwen_{name}":
                        str(path)
                    for name, path
                    in qwen_artifacts.items()
                },
            },
        }

        record_path = (
            case_dir
            / "pipeline_results.json"
        )

        record_path.write_text(
            json.dumps(
                record,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # ====================================================
        # SESSION STATE
        # ====================================================

        st.session_state.results = {

            "record":
                record,

            "image":
                image,

            "prediction":
                prediction,

            "attribution":
                attribution,

            "segmentation":
                segmentation,

            "knowledge":
                knowledge,

            "overlay":
                overlay,

            "overlay_path":
                overlay_path,

            "record_path":
                record_path,

            "arrays_path":
                arrays_path,

            "attribution_path":
                attribution_path,

            "prompt_artifacts":
                prompt_artifacts,

            "qwen_result":
                qwen_result,

            "qwen_artifacts":
                qwen_artifacts,

            "case":
                case,

            "case_dir":
                case_dir,

            "model_type":
                model_type,

            "model_path":
                model_path,

            "qwen_base_url":
                qwen_base_url,

            "qwen_model":
                qwen_model,
        }

        st.session_state.chat_messages = []

        progress.progress(100)

        status.success(
            "✅ Analysis completed successfully."
        )

        try:
            complete_case(case)
        except Exception:
            pass

    except Exception as e:

        status.error(
            "❌ Pipeline failed."
        )

        st.exception(e)

        st.stop()


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results


if results is not None:

    prediction = results[
        "prediction"
    ]

    attribution = results[
        "attribution"
    ]

    segmentation = results[
        "segmentation"
    ]

    knowledge = results[
        "knowledge"
    ]

    overlay = results[
        "overlay"
    ]

    model_type = results[
        "model_type"
    ]


    # ========================================================
    # 3. PREDICTION SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">3. Prediction Summary</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Prediction",
            str(
                prediction.predicted_class
            ),
        )

    with c2:

        st.metric(
            "Detections",
            len(
                prediction.detections
            ),
        )

    with c3:

        st.metric(
            "Model",
            model_display_name(
                model_type
            ),
        )

    with c4:

        st.metric(
            "XAI",
            "Integrated Gradients",
        )


    # ========================================================
    # 4. DETECTIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">4. Detection Results</div>',
        unsafe_allow_html=True,
    )

    if len(
        prediction.detections
    ) == 0:

        st.info(
            "No pneumonia detection boxes were produced."
        )

    else:

        rows = []

        for i, detection in enumerate(
            prediction.detections
        ):

            rows.append(
                {
                    "Detection":
                        i + 1,

                    **safe_as_dict(
                        detection
                    ),
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
        )


    # ========================================================
    # 5. EVIDENCE OVERLAY
    # ========================================================

    st.markdown(
        '<div class="section-title">5. Evidence Overlay</div>',
        unsafe_allow_html=True,
    )

    overlay_path = Path(
        results["overlay_path"]
    )

    if overlay_path.exists():

        st.image(
            str(overlay_path),
            caption=(
                "Prediction + Segmentation + Attribution"
            ),
            use_container_width=True,
        )


    # ========================================================
    # 6. INTEGRATED GRADIENTS
    # ========================================================

    st.markdown(
        '<div class="section-title">6. Integrated Gradients</div>',
        unsafe_allow_html=True,
    )

    captum_map = np.asarray(
        attribution.signed_map,
        dtype=np.float32,
    )

    display_captum_attribution(
        captum_map
    )


    # ========================================================
    # 7. SEGMENTATION
    # ========================================================

    st.markdown(
        '<div class="section-title">7. Segmentation Map</div>',
        unsafe_allow_html=True,
    )

    display_array_as_image(
        segmentation.class_map,
        "Anatomical Segmentation",
    )


    # ========================================================
    # 8. GROUNDED KNOWLEDGE
    # ========================================================

    st.markdown(
        '<div class="section-title">8. Grounded Knowledge</div>',
        unsafe_allow_html=True,
    )

    st.json(
        safe_as_dict(
            knowledge
        )
    )


    # ========================================================
    # 9. QWEN REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">9. Initial Qwen Report</div>',
        unsafe_allow_html=True,
    )

    qwen_result = results.get(
        "qwen_result"
    )

    if qwen_result is None:

        st.info(
            "Initial Qwen report was skipped."
        )

    else:

        try:
            st.json(
                qwen_result.as_dict()
            )
        except Exception:

            try:
                st.json(
                    qwen_result.metadata()
                )
            except Exception:

                st.write(
                    qwen_result
                )


    # ========================================================
    # 10. PROMPT PACKET
    # ========================================================

    st.markdown(
        '<div class="section-title">10. Qwen Prompt Packet</div>',
        unsafe_allow_html=True,
    )

    prompt_artifacts = (
        results[
            "prompt_artifacts"
        ]
    )

    packet_path = (
        prompt_artifacts.get(
            "packet"
        )
    )

    if packet_path:

        packet_path = Path(
            packet_path
        )

        if packet_path.exists():

            packet_text = (
                packet_path.read_text(
                    encoding="utf-8"
                )
            )

            with st.expander(
                "View frozen prompt packet"
            ):

                st.code(
                    packet_text,
                    language="json",
                )


    # ========================================================
    # 11. GROUNDED CHAT
    # ========================================================

    st.markdown(
        '<div class="section-title">11. Case-Grounded Chat</div>',
        unsafe_allow_html=True,
    )

    case_dir = Path(
        results["case_dir"]
    )

    st.caption(
        f"Chat is restricted to case: `{case_dir.name}`"
    )

    initial_chat_response = (
        load_initial_chat_response(
            case_dir
        )
    )

    if initial_chat_response is not None:

        with st.expander(
            "View stored initial Qwen response"
        ):

            st.json(
                initial_chat_response
            )

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    history_path = (
        case_dir
        / "chat_history.json"
    )

    if history_path.exists():

        try:

            history = json.loads(
                history_path.read_text(
                    encoding="utf-8"
                )
            )

            stored_messages = history.get(
                "messages",
                [],
            )

            for message in stored_messages:

                if not isinstance(
                    message,
                    dict,
                ):
                    continue

                role = message.get(
                    "role"
                )

                content = message.get(
                    "content",
                    "",
                )

                if role == "user":

                    st.markdown(
                        f"""
                        <div class="chat-user">
                        <strong>You</strong><br>
                        {content}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif role == "assistant":

                    try:

                        parsed = json.loads(
                            content
                        )

                        st.markdown(
                            '<div class="chat-assistant">'
                            '<strong>GenXplainer</strong>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        st.json(parsed)

                    except Exception:

                        st.markdown(
                            f"""
                            <div class="chat-assistant">
                            <strong>GenXplainer</strong><br>
                            {content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        except Exception as e:

            st.warning(
                f"Could not load chat history: {e}"
            )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about this case..."
    )

    if question:

        from chat import chat_once

        try:

            with st.spinner(
                "Qwen is answering from the frozen case evidence..."
            ):

                response = chat_once(
                    case_dir=case_dir,
                    question=question,
                    base_url=results[
                        "qwen_base_url"
                    ],
                    model=results[
                        "qwen_model"
                    ],
                    max_tokens=800,
                    history_messages=6,
                    include_images=True,
                )

            st.rerun()

        except Exception as e:

            st.error(
                "Chat request failed."
            )

            st.exception(e)


    # ========================================================
    # 12. PIPELINE METADATA
    # ========================================================

    st.markdown(
        '<div class="section-title">12. Pipeline Metadata</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "View Complete Pipeline JSON"
    ):

        st.json(
            results["record"]
        )


    # ========================================================
    # 13. DOWNLOADS
    # ========================================================

    st.markdown(
        '<div class="section-title">13. Download Results</div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2)


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    record_path = Path(
        results["record_path"]
    )

    if record_path.exists():

        with open(
            record_path,
            "rb",
        ) as f:

            d1.download_button(
                "⬇️ Pipeline Results JSON",
                data=f.read(),
                file_name="pipeline_results.json",
                mime="application/json",
                use_container_width=True,
            )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    overlay_path = Path(
        results["overlay_path"]
    )

    if overlay_path.exists():

        with open(
            overlay_path,
            "rb",
        ) as f:

            d2.download_button(
                "⬇️ Evidence Overlay PNG",
                data=f.read(),
                file_name=(
                    "attribution_prediction_"
                    "segmentation_overlay.png"
                ),
                mime="image/png",
                use_container_width=True,
            )


    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    arrays_path = Path(
        results["arrays_path"]
    )

    if arrays_path.exists():

        with open(
            arrays_path,
            "rb",
        ) as f:

            st.download_button(
                "⬇️ Segmentation Arrays (.npz)",
                data=f.read(),
                file_name=(
                    "segmentation_arrays.npz"
                ),
                mime="application/octet-stream",
                use_container_width=True,
            )


    # --------------------------------------------------------
    # ATTRIBUTION
    # --------------------------------------------------------

    attribution_path = Path(
        results["attribution_path"]
    )

    if attribution_path.exists():

        with open(
            attribution_path,
            "rb",
        ) as f:

            st.download_button(
                "⬇️ Integrated Gradients (.npz)",
                data=f.read(),
                file_name=(
                    "integrated_gradients_"
                    "attribution.npz"
                ),
                mime="application/octet-stream",
                use_container_width=True,
            )


    # --------------------------------------------------------
    # PROMPT PACKET
    # --------------------------------------------------------

    if packet_path:

        packet_path = Path(
            packet_path
        )

        if packet_path.exists():

            with open(
                packet_path,
                "rb",
            ) as f:

                st.download_button(
                    "⬇️ Qwen Prompt Packet",
                    data=f.read(),
                    file_name="prompt_packet.json",
                    mime="application/json",
                    use_container_width=True,
                )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    if history_path.exists():

        with open(
            history_path,
            "rb",
        ) as f:

            st.download_button(
                "⬇️ Chat History",
                data=f.read(),
                file_name="chat_history.json",
                mime="application/json",
                use_container_width=True,
            )
