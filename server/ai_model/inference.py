"""
Inference Engine for Custom Code Generation Model
Handles model loading, text generation, and API integration
"""

import json
import torch
from pathlib import Path
from typing import Optional, List, Dict, Union

from .tokenizer import CodeTokenizer
from .transformer import CodeTransformer


class CodeGenerator:
    """
    High-level interface for code generation using the trained model.
    
    Features:
    - Load models from checkpoints
    - Generate code completions
    - Multi-language support
    - Streaming generation
    - Temperature and sampling controls
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.model: Optional[CodeTransformer] = None
        self.tokenizer: Optional[CodeTokenizer] = None
        self.config: Dict = {}
        
        if checkpoint_path:
            self.load(checkpoint_path)
    
    def load(self, checkpoint_path: str):
        """
        Load model and tokenizer from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        config_path = checkpoint_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.tokenizer = CodeTokenizer(vocab_size=self.config['vocab_size'])
        tokenizer_path = checkpoint_dir / 'tokenizer.json'
        if tokenizer_path.exists():
            self.tokenizer.load(str(tokenizer_path))
        
        self.model = CodeTransformer(
            vocab_size=self.config['vocab_size'],
            embed_dim=self.config['embed_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            max_seq_len=self.config.get('max_seq_len', 2048),
            pad_token_id=self.config.get('pad_token_id', 0)
        )
        
        model_path = checkpoint_dir / 'model.pt'
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")
    
    def _ensure_loaded(self):
        """Ensure model is loaded before inference."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        language: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate code completion from a prompt.
        
        Args:
            prompt: Input text/code prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            language: Programming language hint
            stop_sequences: Sequences that stop generation
            
        Returns:
            Generated text
        """
        self._ensure_loaded()
        
        if language:
            input_ids = self.tokenizer.encode_code(prompt, language=language)
        else:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        if language:
            output_text = self.tokenizer.decode_code(
                generated[0].tolist(),
                skip_special_tokens=True
            )
        else:
            output_text = self.tokenizer.decode(
                generated[0].tolist(),
                skip_special_tokens=True
            )
        
        if stop_sequences:
            for seq in stop_sequences:
                if seq in output_text:
                    output_text = output_text[:output_text.index(seq)]
        
        return output_text
    
    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        Stream tokens one at a time as they're generated.
        
        Args:
            prompt: Input text/code prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            
        Yields:
            Generated tokens one at a time
        """
        self._ensure_loaded()
        
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        self.model.eval()
        
        for _ in range(max_tokens):
            if input_tensor.size(1) >= self.model.max_seq_len:
                break
            
            outputs = self.model(input_tensor)
            logits = outputs['logits'][:, -1, :]
            
            logits = logits / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            yield token_text
    
    def complete_code(
        self,
        code: str,
        language: str = 'python',
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        Complete code snippet.
        
        Args:
            code: Partial code to complete
            language: Programming language
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Code completion
        """
        return self.generate(
            prompt=code,
            max_tokens=max_tokens,
            temperature=temperature,
            language=language,
            stop_sequences=['\n\n\n', '# End', '// End']
        )
    
    def explain_code(
        self,
        code: str,
        language: str = 'python'
    ) -> str:
        """
        Generate an explanation of code.
        
        Args:
            code: Code to explain
            language: Programming language
            
        Returns:
            Explanation text
        """
        prompt = f"# Explanation of the following {language} code:\n{code}\n\n# This code"
        return self.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            stop_sequences=['\n\n', '```']
        )
    
    def fix_code(
        self,
        code: str,
        error: str,
        language: str = 'python'
    ) -> str:
        """
        Suggest a fix for code with an error.
        
        Args:
            code: Buggy code
            error: Error message
            language: Programming language
            
        Returns:
            Fixed code suggestion
        """
        prompt = f"""# Original code with error:
{code}

# Error: {error}

# Fixed code:
"""
        return self.generate(
            prompt=prompt,
            max_tokens=250,
            temperature=0.5,
            language=language
        )
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'parameters': self.model.count_parameters(),
            'config': self.config,
            'device': str(self.device),
            'vocab_size': len(self.tokenizer) if self.tokenizer else 0
        }


class AIAssistant:
    """
    High-level AI assistant that wraps the code generator with conversation support.
    """
    
    def __init__(self, generator: CodeGenerator):
        self.generator = generator
        self.conversation_history: List[Dict] = []
    
    def chat(self, message: str, max_tokens: int = 200) -> str:
        """
        Chat with the AI assistant.
        
        Args:
            message: User message
            max_tokens: Maximum response length
            
        Returns:
            Assistant response
        """
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        context = self._build_context()
        
        response = self.generator.generate(
            prompt=context,
            max_tokens=max_tokens,
            temperature=0.8
        )
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def _build_context(self) -> str:
        """Build conversation context from history."""
        context_parts = []
        
        recent_history = self.conversation_history[-6:]
        
        for msg in recent_history:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                context_parts.append(f"User: {content}")
            else:
                context_parts.append(f"Assistant: {content}")
        
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def analyze_code(self, code: str, language: str = 'python') -> Dict:
        """
        Analyze code and provide insights.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Analysis results
        """
        return {
            'explanation': self.generator.explain_code(code, language),
            'completion': self.generator.complete_code(code, language, max_tokens=50),
            'language': language
        }


def load_generator(checkpoint_path: str) -> CodeGenerator:
    """
    Convenience function to load a code generator.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded CodeGenerator instance
    """
    generator = CodeGenerator(checkpoint_path)
    return generator


if __name__ == '__main__':
    print("Inference Engine Demo")
    print("="*60)
    
    generator = CodeGenerator()
    
    from .tokenizer import CodeTokenizer
    from .transformer import create_model
    
    tokenizer = CodeTokenizer(vocab_size=5000)
    sample_data = [
        "def hello(): return 'world'",
        "class Foo: pass",
        "for i in range(10): print(i)"
    ]
    tokenizer.train(sample_data, min_frequency=1, verbose=False)
    
    model = create_model(len(tokenizer), 'tiny')
    model.eval()
    
    generator.model = model
    generator.tokenizer = tokenizer
    generator.config = model.get_config()
    
    print("\nTest generation:")
    prompt = "def add(a, b):"
    result = generator.generate(prompt, max_tokens=20, temperature=1.0)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    
    print("\nModel Info:")
    print(generator.get_model_info())
