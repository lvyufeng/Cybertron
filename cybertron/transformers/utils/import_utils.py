# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.util
import json
import os
import shutil
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any

from packaging import version

from .versions import importlib_metadata

from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_MS = os.environ.get("USE_MS", "AUTO").upper()


if USE_MS in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _ms_available = importlib.util.find_spec("mindspore") is not None
    if _ms_available:
        candidates = (
            "mindspore",
            "mindspore-gpu",
            "mindspore-ascend",
            "mindspore-dev",
        )
        _ms_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _ms_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _ms_available = _ms_version is not None
    if _ms_available:
        if version.parse(_ms_version) < version.parse("2"):
            logger.info(
                f"MindSpore found but with version {_ms_version}. Transformers requires version 2 minimum."
            )
            _ms_available = False
        else:
            logger.info(f"TensorFlow version {_ms_version} available.")

_datasets_available = importlib.util.find_spec("datasets") is not None
try:
    # Check we're not importing a "datasets" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    _ = importlib_metadata.version("datasets")
    _datasets_metadata = importlib_metadata.metadata("datasets")
    if _datasets_metadata.get("author", "") != "HuggingFace Inc.":
        _datasets_available = False
except importlib_metadata.PackageNotFoundError:
    _datasets_available = False


_detectron2_available = importlib.util.find_spec("detectron2") is not None
try:
    _detectron2_version = importlib_metadata.version("detectron2")
    logger.debug(f"Successfully imported detectron2 version {_detectron2_version}")
except importlib_metadata.PackageNotFoundError:
    _detectron2_available = False


_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib_metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib_metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib_metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib_metadata.PackageNotFoundError:
        _faiss_available = False


_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
except importlib_metadata.PackageNotFoundError:
    _ftfy_available = False


coloredlogs = importlib.util.find_spec("coloredlogs") is not None
try:
    _coloredlogs_available = importlib_metadata.version("coloredlogs")
    logger.debug(f"Successfully imported sympy version {_coloredlogs_available}")
except importlib_metadata.PackageNotFoundError:
    _coloredlogs_available = False


sympy_available = importlib.util.find_spec("sympy") is not None
try:
    _sympy_available = importlib_metadata.version("sympy")
    logger.debug(f"Successfully imported sympy version {_sympy_available}")
except importlib_metadata.PackageNotFoundError:
    _sympy_available = False


_tf2onnx_available = importlib.util.find_spec("tf2onnx") is not None
try:
    _tf2onnx_version = importlib_metadata.version("tf2onnx")
    logger.debug(f"Successfully imported tf2onnx version {_tf2onnx_version}")
except importlib_metadata.PackageNotFoundError:
    _tf2onnx_available = False


_onnx_available = importlib.util.find_spec("onnxruntime") is not None
try:
    _onxx_version = importlib_metadata.version("onnx")
    logger.debug(f"Successfully imported onnx version {_onxx_version}")
except importlib_metadata.PackageNotFoundError:
    _onnx_available = False


_pytorch_quantization_available = importlib.util.find_spec("pytorch_quantization") is not None
try:
    _pytorch_quantization_version = importlib_metadata.version("pytorch_quantization")
    logger.debug(f"Successfully imported pytorch-quantization version {_pytorch_quantization_version}")
except importlib_metadata.PackageNotFoundError:
    _pytorch_quantization_available = False


_soundfile_available = importlib.util.find_spec("soundfile") is not None
try:
    _soundfile_version = importlib_metadata.version("soundfile")
    logger.debug(f"Successfully imported soundfile version {_soundfile_version}")
except importlib_metadata.PackageNotFoundError:
    _soundfile_available = False


_tensorflow_probability_available = importlib.util.find_spec("tensorflow_probability") is not None
try:
    _tensorflow_probability_version = importlib_metadata.version("tensorflow_probability")
    logger.debug(f"Successfully imported tensorflow-probability version {_tensorflow_probability_version}")
except importlib_metadata.PackageNotFoundError:
    _tensorflow_probability_available = False


_timm_available = importlib.util.find_spec("timm") is not None
try:
    _timm_version = importlib_metadata.version("timm")
    logger.debug(f"Successfully imported timm version {_timm_version}")
except importlib_metadata.PackageNotFoundError:
    _timm_available = False


_natten_available = importlib.util.find_spec("natten") is not None
try:
    _natten_version = importlib_metadata.version("natten")
    logger.debug(f"Successfully imported natten version {_natten_version}")
except importlib_metadata.PackageNotFoundError:
    _natten_available = False


_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
try:
    _torchaudio_version = importlib_metadata.version("torchaudio")
    logger.debug(f"Successfully imported torchaudio version {_torchaudio_version}")
except importlib_metadata.PackageNotFoundError:
    _torchaudio_available = False


_phonemizer_available = importlib.util.find_spec("phonemizer") is not None
try:
    _phonemizer_version = importlib_metadata.version("phonemizer")
    logger.debug(f"Successfully imported phonemizer version {_phonemizer_version}")
except importlib_metadata.PackageNotFoundError:
    _phonemizer_available = False


_pyctcdecode_available = importlib.util.find_spec("pyctcdecode") is not None
try:
    _pyctcdecode_version = importlib_metadata.version("pyctcdecode")
    logger.debug(f"Successfully imported pyctcdecode version {_pyctcdecode_version}")
except importlib_metadata.PackageNotFoundError:
    _pyctcdecode_available = False


_librosa_available = importlib.util.find_spec("librosa") is not None
try:
    _librosa_version = importlib_metadata.version("librosa")
    logger.debug(f"Successfully imported librosa version {_librosa_version}")
except importlib_metadata.PackageNotFoundError:
    _librosa_available = False

ccl_version = "N/A"
_is_ccl_available = (
    importlib.util.find_spec("torch_ccl") is not None
    or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
)
try:
    ccl_version = importlib_metadata.version("oneccl_bind_pt")
    logger.debug(f"Successfully imported oneccl_bind_pt version {ccl_version}")
except importlib_metadata.PackageNotFoundError:
    _is_ccl_available = False

_decord_availale = importlib.util.find_spec("decord") is not None
try:
    _decord_version = importlib_metadata.version("decord")
    logger.debug(f"Successfully imported decord version {_decord_version}")
except importlib_metadata.PackageNotFoundError:
    _decord_availale = False

# This is the version of torch required to run torch.fx features and torch.onnx with dictionary inputs.
TORCH_FX_REQUIRED_VERSION = version.parse("1.10")
TORCH_ONNX_DICT_INPUTS_MINIMUM_VERSION = version.parse("1.8")

def is_ms_available():
    return _ms_available

def is_kenlm_available():
    return importlib.util.find_spec("kenlm") is not None


def is_pyctcdecode_available():
    return _pyctcdecode_available


def is_librosa_available():
    return _librosa_available


def is_bs4_available():
    return importlib.util.find_spec("bs4") is not None


def is_coloredlogs_available():
    return _coloredlogs_available


def is_tf2onnx_available():
    return _tf2onnx_available


def is_onnx_available():
    return _onnx_available



def is_torch_tensorrt_fx_available():
    if importlib.util.find_spec("torch_tensorrt") is None:
        return False
    return importlib.util.find_spec("torch_tensorrt.fx") is not None


def is_datasets_available():
    return _datasets_available


def is_detectron2_available():
    return _detectron2_available


def is_more_itertools_available():
    return importlib.util.find_spec("more_itertools") is not None


def is_rjieba_available():
    return importlib.util.find_spec("rjieba") is not None


def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None


def is_py3nvml_available():
    return importlib.util.find_spec("py3nvml") is not None


def is_sacremoses_available():
    return importlib.util.find_spec("sacremoses") is not None


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_ninja_available():
    return importlib.util.find_spec("ninja") is not None


def is_bitsandbytes_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_torchdistx_available():
    return importlib.util.find_spec("torchdistx") is not None


def is_faiss_available():
    return _faiss_available


def is_scipy_available():
    return importlib.util.find_spec("scipy") is not None


def is_sklearn_available():
    if importlib.util.find_spec("sklearn") is None:
        return False
    return is_scipy_available() and importlib.util.find_spec("sklearn.metrics")


def is_sentencepiece_available():
    return importlib.util.find_spec("sentencepiece") is not None


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_accelerate_available():
    return importlib.util.find_spec("accelerate") is not None


def is_optimum_available():
    return importlib.util.find_spec("optimum") is not None


def is_safetensors_available():
    return importlib.util.find_spec("safetensors") is not None


def is_tokenizers_available():
    return importlib.util.find_spec("tokenizers") is not None


def is_vision_available():
    return importlib.util.find_spec("PIL") is not None


def is_pytesseract_available():
    return importlib.util.find_spec("pytesseract") is not None


def is_spacy_available():
    return importlib.util.find_spec("spacy") is not None


def is_in_notebook():
    try:
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")
        if "DATABRICKS_RUNTIME_VERSION" in os.environ and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0":
            # Databricks Runtime 11.0 and above uses IPython kernel by default so it should be compatible with Jupyter notebook
            # https://docs.microsoft.com/en-us/azure/databricks/notebooks/ipython-kernel
            raise ImportError("databricks")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


def is_pytorch_quantization_available():
    return _pytorch_quantization_available


def is_tensorflow_probability_available():
    return _tensorflow_probability_available


def is_pandas_available():
    return importlib.util.find_spec("pandas") is not None


def is_sagemaker_dp_enabled():
    # Get the sagemaker specific env variable.
    sagemaker_params = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        sagemaker_params = json.loads(sagemaker_params)
        if not sagemaker_params.get("sagemaker_distributed_dataparallel_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return importlib.util.find_spec("smdistributed") is not None


def is_sagemaker_mp_enabled():
    # Get the sagemaker specific mp parameters from smp_options variable.
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # Parse it and check the field "partitions" is included, it is required for model parallel.
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # Get the sagemaker specific framework parameters from mpi_options variable.
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return importlib.util.find_spec("smdistributed") is not None


def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


def is_soundfile_availble():
    return _soundfile_available


def is_timm_available():
    return _timm_available


def is_natten_available():
    return _natten_available


def is_torchaudio_available():
    return _torchaudio_available


def is_speech_available():
    # For now this depends on torchaudio but the exact dependency might evolve in the future.
    return _torchaudio_available


def is_phonemizer_available():
    return _phonemizer_available


def is_ccl_available():
    return _is_ccl_available


def is_decord_available():
    return _decord_availale


def is_sudachi_available():
    return importlib.util.find_spec("sudachipy") is not None


def is_jumanpp_available():
    return (importlib.util.find_spec("rhoknp") is not None) and (shutil.which("jumanpp") is not None)


def is_cython_available():
    return importlib.util.find_spec("pyximport") is not None


# docstyle-ignore
DATASETS_IMPORT_ERROR = """
{0} requires the 🤗 Datasets library but it was not found in your environment. You can install it with:
```
pip install datasets
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install datasets
```
then restarting your kernel.

Note that if you have a local folder named `datasets` or a local python file named `datasets.py` in your current
working directory, python may try to import this instead of the 🤗 Datasets library. You should rename this folder or
that python file if that's the case. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
TOKENIZERS_IMPORT_ERROR = """
{0} requires the 🤗 Tokenizers library but it was not found in your environment. You can install it with:
```
pip install tokenizers
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tokenizers
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYTORCH_IMPORT_ERROR_WITH_TF = """
{0} requires the PyTorch library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to our PyTorch classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!

If you really do want to use PyTorch please go to
https://pytorch.org/get-started/locally/ and follow the instructions that
match your environment.
"""

# docstyle-ignore
TF_IMPORT_ERROR_WITH_PYTORCH = """
{0} requires the TensorFlow library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
with "TF", but are otherwise identically named to our TF classes.
If you want to use PyTorch, please use those classes instead!

If you really do want to use TensorFlow, please follow the instructions on the
installation page https://www.tensorflow.org/install that match your environment.
"""

# docstyle-ignore
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SKLEARN_IMPORT_ERROR = """
{0} requires the scikit-learn library but it was not found in your environment. You can install it with:
```
pip install -U scikit-learn
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install -U scikit-learn
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYTORCH_QUANTIZATION_IMPORT_ERROR = """
{0} requires the pytorch-quantization library but it was not found in your environment. You can install it with pip:
`pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TENSORFLOW_PROBABILITY_IMPORT_ERROR = """
{0} requires the tensorflow_probability library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/tensorflow/probability. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TENSORFLOW_TEXT_IMPORT_ERROR = """
{0} requires the tensorflow_text library but it was not found in your environment. You can install it with pip as
explained here: https://www.tensorflow.org/text/guide/tf_text_intro.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PHONEMIZER_IMPORT_ERROR = """
{0} requires the phonemizer library but it was not found in your environment. You can install it with pip:
`pip install phonemizer`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SACREMOSES_IMPORT_ERROR = """
{0} requires the sacremoses library but it was not found in your environment. You can install it with pip:
`pip install sacremoses`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SPEECH_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
NATTEN_IMPORT_ERROR = """
{0} requires the natten library but it was not found in your environment. You can install it by referring to:
shi-labs.com/natten . You can also install it with pip (may take longer to build):
`pip install natten`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library but it was not found in your environment. You can install it with pip:
`pip install accelerate`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
CCL_IMPORT_ERROR = """
{0} requires the torch ccl library but it was not found in your environment. You can install it with pip:
`pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable`
Please note that you may need to restart your runtime after installation.
"""

DECORD_IMPORT_ERROR = """
{0} requires the decord library but it was not found in your environment. You can install it with pip: `pip install
decord`. Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("sacremoses", (is_sacremoses_available, SACREMOSES_IMPORT_ERROR)),
        ("pytorch_quantization", (is_pytorch_quantization_available, PYTORCH_QUANTIZATION_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        ("tensorflow_probability", (is_tensorflow_probability_available, TENSORFLOW_PROBABILITY_IMPORT_ERROR)),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("natten", (is_natten_available, NATTEN_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        ("oneccl_bind_pt", (is_ccl_available, CCL_IMPORT_ERROR)),
        ("decord", (is_decord_available, DECORD_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    # Raise an error for users who might not realize that classes without "TF" are torch-only
    if "mindspore" in backends and not is_ms_available():
        raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))


    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattribute__(cls, key):
        if key.startswith("_") and key != "_from_config":
            return super().__getattribute__(key)
        requires_backends(cls, cls._backends)


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""