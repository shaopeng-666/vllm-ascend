# Integrating New Multi-Modal Models into vLLM-Ascend

This guide demonstrates how to integrate novel or customized Multi-Modal models into vLLM-Ascend. For foundational concepts, it is highly recommended to refer to:
[Adding a New Model - vLLM Documentation](https://docs.vllm.ai/en/stable/contributing/model/)

This guide distinguishes between two scenarios for multimodal model transfer tasks: one involves existing multimodal models with custom visual processing modules, and the other employs novel visual processing modules combined with language models.

## 1.Implementing New Multi-Modal Models Only With Custom Vit Modules
### 1.1 Prepare Your Pytorch Code
(1) prepare Vit modules code
**Critical Implementation Details**
- __init__() func should include a `prefix` argument
- The load_weights func requires implementation.

Here is a reference implementation of code based on the Qwen2-VL baseline model, but replacing only its ViT module.
```python
from transformer import CustomQwen2VisionConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
class CustomQwen2VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: CustomQwen2VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        ...

    def load_weights(self, weights: Iterable[Tuple[str,
                                                torch.Tensor]]) -> Set[str]:
        ...

@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class CustomQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config)
        self.visual = CustomQwen2VisionTransformer(
            self.config.vision_config,
            quant_config=self._maybe_ignore_quant_config(
                vllm_config.quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )
```
(2) Register Custom Models as Out-of-Tree Plugins in vLLM 

Firstly, Save the PyTorch code prepared in Section 1.1 (1)as <model_name>.py and place it under the vllm_ascend/models/ directory.
Secondly, Add the registration code to the vllm_ascend/models/__init__.py file as shown in the example below:
```python
from vllm import ModelRegistry

def register_model():
    from .custom_model import CustomQwen2VLForConditionalGeneration        # New custom model

    # For NEW architectures: Register with unique name
    ModelRegistry.register_model(
        "CustomQwen2VLForConditionalGeneration",
        "vllm_ascend.models.custom_model:CustomQwen2VLForConditionalGeneration"
    )
```
## 2.Implementing New Multi-Modal Models from Scratch
### 2.1 Prepare LanguageModel Code
For language models, prepare and register the code by following the guidelines in [vllm-ascend/docs/source/developer_guide/model_registration.md](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/model_registration.md)
### 2.2 Prepare additional code to handle multimodal model inputs
#### 2.2.1 Prepare input processing handler classes
(1) Define a subclass of [BaseProcessingInfo](https://docs.vllm.ai/en/latest/api/vllm/multimodal/processing.html#vllm.multimodal.processing.BaseProcessingInfo).

The subclass should contain a "get_supported_mm_limits" func to to specify the maximum input count per modality.

(2) Define a subclass of [BaseDummyInputsBuilder](https://docs.vllm.ai/en/latest/api/vllm/multimodal/profiling.html#vllm.multimodal.profiling.BaseDummyInputsBuilder)

The subclass should contain "get_dummy_text" and "get_dummy_mm_data" func to construct inputs with worst-case memory usage.

(3) Define a subclass of [BaseMultiModalProcessor](https://docs.vllm.ai/en/latest/api/vllm/multimodal/processing.html#vllm.multimodal.processing.BaseMultiModalProcessor)

The subclass contains a func named "_get_mm_fields_config" to return the tensor schema of HF processor outputs corresponding to input multimodal items and a func named "_get_prompt_updates" to return the update operations performed by Hugging Face processors.

#### 2.2.2 Adjust *ModelForCausalLM Code To Satisfy Multi-Modal Input
(1) Register processor-related classes.

Qwen2-VL model register example
```python
from vllm.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class Qwen2VLForConditionalGeneration(...):
    ...
```
(2) Add get_language_model func to retrieve the language model object.
```python
def get_language_model(self) -> torch.nn.Module:
        return self.language_model
```
(3) Add Multi-Modal embedding related func.
```python
#func1: Multi-Modal embedding func
def get_multimodal_embeddings(
        self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
#func2: Merge Multi-Modal embedding and Input-ids embedding
def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
```
In v1 mode, [model_runner_v1](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/model_runner_v1.py) will invoke these two functions by default to obtain the actual input embedding outputs.

### 2.3 Register Custom Models as Out-of-Tree Plugins in vLLM
Refer to Section 1.1(2) to register your model.