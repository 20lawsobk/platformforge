"""
HTTP API Server for Custom AI Model
Provides REST endpoints for model inference and training
"""

import os
import json
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional
import threading

from .tokenizer import CodeTokenizer
from .transformer import create_model, CodeTransformer
from .inference import CodeGenerator, AIAssistant

MODEL_DIR = Path(__file__).parent / "checkpoints"
TRAINED_MODEL_V2 = MODEL_DIR / "code_model_v2" / "best_model"
TRAINED_MODEL = MODEL_DIR / "code_model" / "best_model"
QUICK_MODEL = MODEL_DIR / "quick_model" / "best_model"
DEFAULT_CHECKPOINT = MODEL_DIR / "demo_model"

def get_best_checkpoint() -> Optional[Path]:
    """Get the best available checkpoint (prioritizes latest trained model)."""
    if TRAINED_MODEL_V2.exists():
        return TRAINED_MODEL_V2
    if TRAINED_MODEL.exists():
        return TRAINED_MODEL
    if QUICK_MODEL.exists():
        return QUICK_MODEL
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    return None


class ModelState:
    """Global model state manager."""
    
    def __init__(self):
        self._generator: Optional[CodeGenerator] = None
        self._assistant: Optional[AIAssistant] = None
        self.is_initialized = False
        self.is_training = False
        self.training_progress = 0.0
        self._lock = threading.Lock()
    
    @property
    def generator(self) -> CodeGenerator:
        """Get generator, raising if not initialized."""
        if self._generator is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self._generator
    
    @property
    def assistant(self) -> AIAssistant:
        """Get assistant, raising if not initialized."""
        if self._assistant is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self._assistant
        
    def initialize(self, checkpoint_path: Optional[str] = None, force: bool = False):
        """Initialize the model with thread safety."""
        with self._lock:
            if self.is_initialized and not force:
                return
            
            gen = CodeGenerator()
            loaded_from = None
            
            if checkpoint_path and Path(checkpoint_path).exists():
                if self._validate_checkpoint(checkpoint_path):
                    print(f"Loading model from: {checkpoint_path}")
                    gen.load(checkpoint_path)
                    loaded_from = checkpoint_path
                else:
                    raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
            else:
                best_checkpoint = get_best_checkpoint()
                if best_checkpoint and self._validate_checkpoint(str(best_checkpoint)):
                    print(f"Loading trained model from: {best_checkpoint}")
                    gen.load(str(best_checkpoint))
                    loaded_from = str(best_checkpoint)
                else:
                    print("No valid trained model found, creating demo model...")
                    self._create_demo_model(gen)
                    loaded_from = "demo"
            
            self._generator = gen
            self._assistant = AIAssistant(gen)
            self.is_initialized = True
            self.loaded_checkpoint = loaded_from
            print(f"Model initialization complete (source: {loaded_from})")
    
    def reload(self, checkpoint_path: Optional[str] = None):
        """Reload model from checkpoint (for hot-reloading after training)."""
        with self._lock:
            self.is_initialized = False
            self._generator = None
            self._assistant = None
        self.initialize(checkpoint_path, force=True)
    
    def _validate_checkpoint(self, path: str) -> bool:
        """Validate checkpoint directory has required files."""
        checkpoint_dir = Path(path)
        required_files = ['model.pt', 'tokenizer.json', 'config.json']
        return all((checkpoint_dir / f).exists() for f in required_files)
    
    def _create_demo_model(self, gen: CodeGenerator):
        """Create a demo model for testing."""
        tokenizer = CodeTokenizer(vocab_size=5000)
        
        demo_data = [
            "def hello_world():\n    print('Hello, World!')\n    return True",
            "class Calculator:\n    def add(self, a, b):\n        return a + b",
            "async function fetchData(url) {\n    const response = await fetch(url);\n    return response.json();\n}",
            "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    pass",
            "const express = require('express');\nconst app = express();\napp.listen(3000);",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "SELECT * FROM users WHERE id = 1;",
            "CREATE TABLE products (id SERIAL PRIMARY KEY, name VARCHAR(255));",
            "provider \"aws\" {\n  region = \"us-east-1\"\n}",
            "resource \"aws_instance\" \"web\" {\n  ami = \"ami-12345\"\n  instance_type = \"t2.micro\"\n}",
        ]
        
        tokenizer.train(demo_data, min_frequency=1, verbose=False)
        
        model = create_model(len(tokenizer), 'tiny')
        model.eval()
        
        gen.model = model
        gen.tokenizer = tokenizer
        gen.config = model.get_config()
        
        print("Demo model initialized (untrained - for testing only)")


model_state = ModelState()


class AIModelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for AI model endpoints."""
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _read_body(self) -> dict:
        """Read JSON body from request."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        return {}
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/health':
            self._send_json({
                'status': 'healthy',
                'model_initialized': model_state.is_initialized,
                'is_training': model_state.is_training
            })
        
        elif path == '/model/info':
            if not model_state.is_initialized:
                self._send_json({'error': 'Model not initialized'}, 503)
                return
            
            info = model_state.generator.get_model_info()
            self._send_json(info)
        
        elif path == '/training/status':
            self._send_json({
                'is_training': model_state.is_training,
                'progress': model_state.training_progress
            })
        
        else:
            self._send_json({'error': 'Not found'}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        try:
            body = self._read_body()
        except json.JSONDecodeError:
            self._send_json({'error': 'Invalid JSON'}, 400)
            return
        
        if path == '/initialize':
            try:
                checkpoint = body.get('checkpoint_path')
                model_state.initialize(checkpoint)
                self._send_json({'status': 'initialized'})
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/reload':
            try:
                checkpoint = body.get('checkpoint_path')
                model_state.reload(checkpoint)
                self._send_json({
                    'status': 'reloaded',
                    'checkpoint': getattr(model_state, 'loaded_checkpoint', None)
                })
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/generate':
            if not model_state.is_initialized:
                self._send_json({'error': 'Model not initialized'}, 503)
                return
            
            prompt = body.get('prompt', '')
            max_tokens = body.get('max_tokens', 100)
            temperature = body.get('temperature', 0.8)
            language = body.get('language')
            
            try:
                result = model_state.generator.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    language=language
                )
                self._send_json({
                    'generated_text': result,
                    'prompt': prompt
                })
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/chat':
            if not model_state.is_initialized:
                self._send_json({'error': 'Model not initialized'}, 503)
                return
            
            message = body.get('message', '')
            max_tokens = body.get('max_tokens', 200)
            
            try:
                response = model_state.assistant.chat(message, max_tokens)
                self._send_json({
                    'response': response,
                    'message': message
                })
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/complete':
            if not model_state.is_initialized:
                self._send_json({'error': 'Model not initialized'}, 503)
                return
            
            code = body.get('code', '')
            language = body.get('language', 'python')
            max_tokens = body.get('max_tokens', 150)
            
            try:
                result = model_state.generator.complete_code(
                    code=code,
                    language=language,
                    max_tokens=max_tokens
                )
                self._send_json({
                    'completion': result,
                    'original_code': code
                })
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/explain':
            if not model_state.is_initialized:
                self._send_json({'error': 'Model not initialized'}, 503)
                return
            
            code = body.get('code', '')
            language = body.get('language', 'python')
            
            try:
                explanation = model_state.generator.explain_code(code, language)
                self._send_json({
                    'explanation': explanation,
                    'code': code
                })
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/fix':
            if not model_state.is_initialized:
                self._send_json({'error': 'Model not initialized'}, 503)
                return
            
            code = body.get('code', '')
            error = body.get('error', '')
            language = body.get('language', 'python')
            
            try:
                fixed = model_state.generator.fix_code(code, error, language)
                self._send_json({
                    'fixed_code': fixed,
                    'original_code': code,
                    'error': error
                })
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/analyze':
            if not model_state.is_initialized:
                self._send_json({'error': 'Model not initialized'}, 503)
                return
            
            code = body.get('code', '')
            language = body.get('language', 'python')
            
            try:
                analysis = model_state.assistant.analyze_code(code, language)
                self._send_json(analysis)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        
        elif path == '/clear_history':
            if model_state.assistant:
                model_state.assistant.clear_history()
            self._send_json({'status': 'cleared'})
        
        else:
            self._send_json({'error': 'Not found'}, 404)
    
    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[AI Model API] {args[0]}")


def run_server(port: int = 8001):
    """Run the AI model API server."""
    model_state.initialize()
    
    server = HTTPServer(('0.0.0.0', port), AIModelHandler)
    print(f"\n{'='*60}")
    print(f"Custom AI Model API Server")
    print(f"{'='*60}")
    print(f"  Running on: http://0.0.0.0:{port}")
    print(f"  Model initialized: {model_state.is_initialized}")
    print(f"{'='*60}\n")
    print("Available endpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /model/info      - Get model information")
    print("  POST /initialize      - Initialize model")
    print("  POST /generate        - Generate text from prompt")
    print("  POST /chat            - Chat with AI assistant")
    print("  POST /complete        - Complete code")
    print("  POST /explain         - Explain code")
    print("  POST /fix             - Fix buggy code")
    print("  POST /analyze         - Analyze code")
    print("  POST /clear_history   - Clear chat history")
    print(f"{'='*60}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AI Model API Server...")
        server.shutdown()


if __name__ == '__main__':
    port = int(os.environ.get('AI_MODEL_PORT', 8001))
    run_server(port)
