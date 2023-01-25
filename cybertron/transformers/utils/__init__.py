from .. import __version__

from .doc import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    copy_func,
    replace_return_docstrings,
)

from .generic import (
    ContextManagers,
    ExplicitEnum,
    ModelOutput,
    PaddingStrategy,
    TensorType,
    cached_property,
    can_return_loss,
    expand_dims,
    find_labels,
    flatten_dict,
    is_numpy_array,
    is_tensor,
    is_ms_tensor,
    reshape,
    squeeze,
    tensor_size,
    to_numpy,
    to_py_obj,
    transpose,
    working_or_temp_dir,
)


from .hub import (
    CLOUDFRONT_DISTRIB_PREFIX,
    DISABLE_TELEMETRY,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_PREFIX,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    S3_BUCKET_PREFIX,
    TRANSFORMERS_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    EntryNotFoundError,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_file,
    default_cache_path,
    define_sagemaker_information,
    download_url,
    extract_commit_hash,
    get_cached_models,
    get_file_from_repo,
    get_full_repo_name,
    has_file,
    http_user_agent,
    is_offline_mode,
    is_remote_url,
    move_cache,
    send_example_telemetry,
)


from .import_utils import is_ms_available

WEIGHTS_NAME = "mindspore_model.ckpt"
WEIGHTS_INDEX_NAME = "mindspore_model.ckpt.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
