"""
Custom Tokenizer for Code and Text Processing
Supports BPE (Byte-Pair Encoding) and character-level tokenization
"""

import re
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class BytePairTokenizer:
    """
    Byte-Pair Encoding tokenizer for code and natural language.
    Can be trained on custom data or loaded from saved vocabulary.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
            '<MASK>': 5,
            '<CODE>': 6,
            '<TEXT>': 7,
        }
        
        self._initialize_vocab()
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens and basic characters."""
        self.vocab = dict(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        base_chars = (
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
            ' \t\n\r'
            '!@#$%^&*()_+-=[]{}|;:\'",.<>?/\\`~'
        )
        
        idx = len(self.special_tokens)
        for char in base_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1
    
    def _get_stats(self, tokens_list: List[List[str]]) -> Counter:
        """Count frequency of adjacent pairs."""
        pairs = Counter()
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs
    
    def _merge_pair(self, tokens_list: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge all occurrences of a pair in the token lists."""
        new_tokens_list = []
        merged = pair[0] + pair[1]
        
        for tokens in tokens_list:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens_list.append(new_tokens)
        
        return new_tokens_list
    
    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = True):
        """
        Train the tokenizer on a corpus of texts using BPE algorithm.
        
        Args:
            texts: List of text strings to train on
            min_frequency: Minimum frequency for a merge to be applied
            verbose: Whether to print training progress
        """
        tokens_list = [[char for char in text] for text in texts]
        
        current_vocab_size = len(self.vocab)
        target_vocab_size = self.vocab_size
        
        if verbose:
            print(f"Training BPE tokenizer...")
            print(f"Starting vocab size: {current_vocab_size}")
            print(f"Target vocab size: {target_vocab_size}")
        
        iteration = 0
        while current_vocab_size < target_vocab_size:
            pairs = self._get_stats(tokens_list)
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < min_frequency:
                break
            
            tokens_list = self._merge_pair(tokens_list, best_pair)
            
            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            
            if merged_token not in self.vocab:
                self.vocab[merged_token] = current_vocab_size
                self.inverse_vocab[current_vocab_size] = merged_token
                current_vocab_size += 1
            
            iteration += 1
            if verbose and iteration % 100 == 0:
                print(f"  Iteration {iteration}: vocab size = {current_vocab_size}")
        
        if verbose:
            print(f"Training complete. Final vocab size: {len(self.vocab)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = list(text)
        
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [pair[0] + pair[1]] + tokens[i + 2:]
                else:
                    i += 1
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<BOS>'])
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                for char in token:
                    if char in self.vocab:
                        token_ids.append(self.vocab[char])
                    else:
                        token_ids.append(self.special_tokens['<UNK>'])
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<EOS>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        special_ids = set(self.special_tokens.values())
        
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.inverse_vocab:
                tokens.append(self.inverse_vocab[tid])
            else:
                tokens.append('<UNK>')
        
        return ''.join(tokens)
    
    def save(self, path: str):
        """Save tokenizer vocabulary and merges to disk."""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load tokenizer vocabulary and merges from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = [tuple(m) for m in data['merges']]
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens['<PAD>']
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens['<BOS>']
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens['<EOS>']
    
    def __len__(self) -> int:
        return len(self.vocab)


class CodeTokenizer(BytePairTokenizer):
    """
    Specialized tokenizer for source code with programming language awareness.
    """
    
    def __init__(self, vocab_size: int = 32000):
        super().__init__(vocab_size)
        
        self.code_special_tokens = {
            '<INDENT>': len(self.vocab),
            '<DEDENT>': len(self.vocab) + 1,
            '<NEWLINE>': len(self.vocab) + 2,
            '<PYTHON>': len(self.vocab) + 3,
            '<JAVASCRIPT>': len(self.vocab) + 4,
            '<TYPESCRIPT>': len(self.vocab) + 5,
            '<RUST>': len(self.vocab) + 6,
            '<GO>': len(self.vocab) + 7,
        }
        
        self.vocab.update(self.code_special_tokens)
        self.inverse_vocab.update({v: k for k, v in self.code_special_tokens.items()})
        self.special_tokens.update(self.code_special_tokens)
    
    def _preprocess_code(self, code: str, language: Optional[str] = None) -> str:
        """Preprocess code for tokenization."""
        lines = code.split('\n')
        processed_lines = []
        
        for line in lines:
            stripped = line.lstrip()
            indent_level = (len(line) - len(stripped)) // 4
            
            indent_tokens = '<INDENT>' * indent_level
            processed_lines.append(indent_tokens + stripped)
        
        result = '<NEWLINE>'.join(processed_lines)
        
        if language:
            lang_token = f'<{language.upper()}>'
            if lang_token in self.special_tokens:
                result = lang_token + result
        
        return result
    
    def encode_code(self, code: str, language: Optional[str] = None, add_special_tokens: bool = True) -> List[int]:
        """
        Encode source code into token IDs with language-specific preprocessing.
        
        Args:
            code: Source code to encode
            language: Programming language (python, javascript, etc.)
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        processed = self._preprocess_code(code, language)
        return self.encode(processed, add_special_tokens)
    
    def decode_code(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back into source code.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded source code string
        """
        text = self.decode(token_ids, skip_special_tokens=False)
        
        lines = text.split('<NEWLINE>')
        processed_lines = []
        
        for line in lines:
            indent_count = line.count('<INDENT>')
            line = line.replace('<INDENT>', '').replace('<DEDENT>', '')
            processed_lines.append('    ' * indent_count + line)
        
        result = '\n'.join(processed_lines)
        
        for token in ['<PYTHON>', '<JAVASCRIPT>', '<TYPESCRIPT>', '<RUST>', '<GO>']:
            result = result.replace(token, '')
        
        if skip_special_tokens:
            for token in ['<BOS>', '<EOS>', '<PAD>', '<UNK>', '<SEP>', '<MASK>']:
                result = result.replace(token, '')
        
        return result


if __name__ == '__main__':
    tokenizer = CodeTokenizer(vocab_size=5000)
    
    training_data = [
        "def hello_world():\n    print('Hello, World!')\n    return True",
        "function greet(name) {\n    console.log('Hello, ' + name);\n    return name;\n}",
        "class Calculator:\n    def __init__(self):\n        self.result = 0\n    def add(self, x):\n        self.result += x",
        "const fetchData = async (url) => {\n    const response = await fetch(url);\n    return response.json();\n}",
        "import os\nimport sys\n\ndef main():\n    args = sys.argv[1:]\n    for arg in args:\n        print(arg)",
    ]
    
    tokenizer.train(training_data, min_frequency=1, verbose=True)
    
    test_code = "def add(a, b):\n    return a + b"
    encoded = tokenizer.encode_code(test_code, language='python')
    decoded = tokenizer.decode_code(encoded)
    
    print(f"\nOriginal:\n{test_code}")
    print(f"\nEncoded: {encoded[:20]}... (length: {len(encoded)})")
    print(f"\nDecoded:\n{decoded}")
