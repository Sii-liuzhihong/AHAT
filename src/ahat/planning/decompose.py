import json
import os
from abc import ABC, abstractmethod
from openai import OpenAI

from typing import Optional, Literal

from ahat.prompts.task_decompose import system_prompt_task_decompose

class LLMClient(ABC):
    """Abstract base class for LLM client implementations."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_content: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: System prompt to guide the LLM.
            user_content: User's input content.
            
        Returns:
            LLM-generated response string.
        """
        pass


class APIClient(LLMClient):
    """LLM client using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 model_name: Optional[str] = None):
        """
        Initialize APIClient with OpenAI credentials.
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model_name = model_name or os.getenv("MODEL_NAME")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate(self, system_prompt: str, user_content: str) -> str:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
        )
        return response.choices[0].message.content


class LocalClient(LLMClient):
    """LLM client using local models with transformers."""
    
    def __init__(self, model_name: Optional[str] = None, device: str = "cuda"):
        """
        Initialize LocalClient with a local model.
        
        Args:
            model_name: HuggingFace model name or path.
            device: Device to run the model on ("cpu" or "cuda").
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Initialized LocalClient with model:", model_name)
        print("Device:", device)
        
        self.model_name = model_name
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            fix_mistral_regex=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device
        )
        
        
    
    def generate(self, system_prompt: str, user_content: str) -> str:
        """Generate response using local model."""
        # Combine system prompt and user content
        messages = [
            {"role": "system", "content": system_prompt_task_decompose},
            {"role": "user", "content": user_content}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks using LLM-based planning.
    
    Supports two modes:
    1. API mode: Uses OpenAI or compatible API
    2. Local mode: Uses local transformer models
    """
    
    def __init__(self, mode: Literal["api", "local"] = "api", **kwargs):
        """
        Initialize TaskDecomposer with specified mode.
        
        Args:
            mode: "api" for OpenAI API or "local" for local models.
            **kwargs: Additional arguments passed to the respective client.
                For API mode: api_key, base_url, model_name
                For local mode: model_name, device
        """
        if mode == "api":
            self.client = APIClient(
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                model_name=kwargs.get("model_name")
            )
        elif mode == "local":
            self.client = LocalClient(
                model_name=kwargs.get("model_name"),
                device=kwargs.get("device", "cuda")
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'api' or 'local'.")
    
    def generate(self, instruction: str, scene_graph: dict) -> str:
        """
        Generate subtask decomposition using the initialized LLM client.
        
        Args:
            instruction: Natural language task description.
            scene_graph: Current environment state as a dict.
            
        Returns:
            LLM-generated subtask decomposition string.
        """
        scene_graph_str = json.dumps(scene_graph, indent=2)
        user_content = f"<Instruction>{instruction}</Instruction>\n<SceneGraph>{scene_graph_str}</SceneGraph>"
        
        return self.client.generate(system_prompt_task_decompose, user_content)
