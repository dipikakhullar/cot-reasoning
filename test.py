from datasets import Dataset
import torch
from tyche import VolumeConfig, VolumeEstimator


class BasinVolumeExtractor:
    """
    Extractor that computes the basin volume for each text input using the VolumeEstimator.
    """
    def __init__(
        self,
        n_samples=10,
        cutoff=1e-2,
        max_seq_len=1024,
        **kwargs
    ):
        super().__init__(feature_names=["basin_volume"], **kwargs)
        self.n_samples = n_samples
        self.cutoff = cutoff
        self.max_seq_len = max_seq_len

    def compute_features(self, inputs) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        dataset = Dataset.from_dict({"text": list(inputs)})
        cfg = VolumeConfig(
            model=self.model,
            tokenizer=self.model.tokenizer,
            dataset=dataset,
            text_key="text",
            n_samples=self.n_samples,
            cutoff=self.cutoff,
            max_seq_len=self.max_seq_len,
            val_size=len(inputs),
            cache_mode=None,
            chunking=False,
            model_type="causal",
            implicit_vectors=True,
            reduction=None
        )
        estimator = VolumeEstimator.from_config(cfg)
        with torch.no_grad():
            result = estimator.run()
        
        # Transpose the result to have shape [batch_size, n_samples]
        volume = result.estimates.transpose(0, 1)
        
        return {"basin_volume": volume}


from datasets import Dataset
import torch
from tyche import VolumeConfig, VolumeEstimator
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Custom class
class FeatureExtractor:
    def __init__(self, feature_names, **kwargs):
        self.feature_names = feature_names

class BasinVolumeExtractor(FeatureExtractor):
    """
    Extractor that computes the basin volume for each text input using the VolumeEstimator.
    """
    def __init__(
        self,
        model,
        tokenizer,
        n_samples=10,
        cutoff=1e-2,
        max_seq_len=1024,
        **kwargs
    ):
        super().__init__(feature_names=["basin_volume"], **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.cutoff = cutoff
        self.max_seq_len = max_seq_len

    def compute_features(self, inputs) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        dataset = Dataset.from_dict({"text": list(inputs)})
        
        cfg = VolumeConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            text_key="text",
            n_samples=self.n_samples,
            cutoff=self.cutoff,
            max_seq_len=self.max_seq_len,
            val_size=len(inputs),
            cache_mode=None,
            chunking=False,
            model_type="causal",
            implicit_vectors=True,
            reduction=None
        )

        estimator = VolumeEstimator.from_config(cfg)

        with torch.no_grad():
            result = estimator.run()
        
        # Transpose the result to have shape [batch_size, n_samples]
        volume = result.estimates.transpose(0, 1)
        
        return {"basin_volume": volume}



# Use only available GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load Model and Tokenizer
model_path = "POSER/models/genie-0"  # Adjust to your model path

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create Extractor
extractor = BasinVolumeExtractor(model=model, tokenizer=tokenizer)

# Sample inputs
inputs = ["This is a test sentence.", "Another example of text."]

# Run extraction
results = extractor.compute_features(inputs)

# Print results
print(f"Basin Volume Results: {results}")