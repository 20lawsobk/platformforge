"""
Universal File Processor for AI Model Training
Handles all file types including directories and zip archives
"""

import os
import io
import zipfile
import tarfile
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Iterator, Union
from dataclasses import dataclass, field


@dataclass
class ProcessedFile:
    """Represents a processed file with its content and metadata."""
    path: str
    content: str
    file_type: str
    language: str
    size: int
    encoding: str = 'utf-8'
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result of processing files from a source."""
    files: List[ProcessedFile]
    total_files: int
    processed_files: int
    skipped_files: int
    errors: List[str]
    stats: Dict


FILE_TYPE_MAPPING = {
    '.py': ('python', 'Python'),
    '.pyw': ('python', 'Python'),
    '.pyi': ('python', 'Python Interface'),
    '.pyx': ('python', 'Cython'),
    
    '.js': ('javascript', 'JavaScript'),
    '.mjs': ('javascript', 'JavaScript Module'),
    '.cjs': ('javascript', 'CommonJS'),
    '.jsx': ('javascript', 'React JSX'),
    
    '.ts': ('typescript', 'TypeScript'),
    '.tsx': ('typescript', 'React TSX'),
    '.mts': ('typescript', 'TypeScript Module'),
    '.cts': ('typescript', 'CommonJS TypeScript'),
    
    '.java': ('java', 'Java'),
    '.kt': ('kotlin', 'Kotlin'),
    '.kts': ('kotlin', 'Kotlin Script'),
    '.scala': ('scala', 'Scala'),
    '.groovy': ('groovy', 'Groovy'),
    
    '.c': ('c', 'C'),
    '.h': ('c', 'C Header'),
    '.cpp': ('cpp', 'C++'),
    '.cc': ('cpp', 'C++'),
    '.cxx': ('cpp', 'C++'),
    '.hpp': ('cpp', 'C++ Header'),
    '.hxx': ('cpp', 'C++ Header'),
    
    '.cs': ('csharp', 'C#'),
    '.fs': ('fsharp', 'F#'),
    '.vb': ('vb', 'Visual Basic'),
    
    '.go': ('go', 'Go'),
    '.rs': ('rust', 'Rust'),
    '.swift': ('swift', 'Swift'),
    '.m': ('objective-c', 'Objective-C'),
    '.mm': ('objective-cpp', 'Objective-C++'),
    
    '.rb': ('ruby', 'Ruby'),
    '.erb': ('ruby', 'ERB Template'),
    '.rake': ('ruby', 'Rake'),
    
    '.php': ('php', 'PHP'),
    '.phtml': ('php', 'PHP HTML'),
    
    '.pl': ('perl', 'Perl'),
    '.pm': ('perl', 'Perl Module'),
    
    '.lua': ('lua', 'Lua'),
    '.r': ('r', 'R'),
    '.R': ('r', 'R'),
    '.jl': ('julia', 'Julia'),
    '.ex': ('elixir', 'Elixir'),
    '.exs': ('elixir', 'Elixir Script'),
    '.erl': ('erlang', 'Erlang'),
    '.hrl': ('erlang', 'Erlang Header'),
    '.clj': ('clojure', 'Clojure'),
    '.cljs': ('clojure', 'ClojureScript'),
    '.hs': ('haskell', 'Haskell'),
    '.lhs': ('haskell', 'Literate Haskell'),
    '.ml': ('ocaml', 'OCaml'),
    '.mli': ('ocaml', 'OCaml Interface'),
    
    '.sh': ('shell', 'Shell Script'),
    '.bash': ('shell', 'Bash Script'),
    '.zsh': ('shell', 'Zsh Script'),
    '.fish': ('shell', 'Fish Script'),
    '.ps1': ('powershell', 'PowerShell'),
    '.psm1': ('powershell', 'PowerShell Module'),
    '.bat': ('batch', 'Batch Script'),
    '.cmd': ('batch', 'Command Script'),
    
    '.sql': ('sql', 'SQL'),
    '.psql': ('sql', 'PostgreSQL'),
    '.mysql': ('sql', 'MySQL'),
    
    '.html': ('html', 'HTML'),
    '.htm': ('html', 'HTML'),
    '.xhtml': ('html', 'XHTML'),
    '.vue': ('vue', 'Vue Component'),
    '.svelte': ('svelte', 'Svelte Component'),
    
    '.css': ('css', 'CSS'),
    '.scss': ('scss', 'SCSS'),
    '.sass': ('sass', 'Sass'),
    '.less': ('less', 'Less'),
    '.styl': ('stylus', 'Stylus'),
    
    '.json': ('json', 'JSON'),
    '.jsonc': ('json', 'JSON with Comments'),
    '.json5': ('json', 'JSON5'),
    
    '.yaml': ('yaml', 'YAML'),
    '.yml': ('yaml', 'YAML'),
    
    '.xml': ('xml', 'XML'),
    '.xsd': ('xml', 'XML Schema'),
    '.xsl': ('xml', 'XSL'),
    '.xslt': ('xml', 'XSLT'),
    '.svg': ('xml', 'SVG'),
    
    '.md': ('markdown', 'Markdown'),
    '.mdx': ('markdown', 'MDX'),
    '.rst': ('restructuredtext', 'reStructuredText'),
    '.txt': ('text', 'Plain Text'),
    
    '.tf': ('terraform', 'Terraform'),
    '.tfvars': ('terraform', 'Terraform Variables'),
    '.tf.json': ('terraform', 'Terraform JSON'),
    '.hcl': ('hcl', 'HCL'),
    
    '.dockerfile': ('docker', 'Dockerfile'),
    '.containerfile': ('docker', 'Containerfile'),
    
    '.toml': ('toml', 'TOML'),
    '.ini': ('ini', 'INI'),
    '.cfg': ('ini', 'Config'),
    '.conf': ('config', 'Config'),
    '.properties': ('properties', 'Properties'),
    '.env': ('env', 'Environment'),
    
    '.graphql': ('graphql', 'GraphQL'),
    '.gql': ('graphql', 'GraphQL'),
    '.proto': ('protobuf', 'Protocol Buffers'),
    
    '.asm': ('assembly', 'Assembly'),
    '.s': ('assembly', 'Assembly'),
    
    '.cmake': ('cmake', 'CMake'),
    '.make': ('make', 'Makefile'),
    '.mk': ('make', 'Makefile'),
    
    '.gradle': ('gradle', 'Gradle'),
    '.sbt': ('sbt', 'SBT'),
    '.pom': ('maven', 'Maven POM'),
    
    '.sol': ('solidity', 'Solidity'),
    '.vy': ('vyper', 'Vyper'),
    
    '.ipynb': ('jupyter', 'Jupyter Notebook'),
    
}

SPECIAL_FILENAMES = {
    'Dockerfile': ('docker', 'Dockerfile'),
    'Containerfile': ('docker', 'Containerfile'),
    'Makefile': ('make', 'Makefile'),
    'CMakeLists.txt': ('cmake', 'CMake'),
    'Vagrantfile': ('ruby', 'Vagrantfile'),
    'Gemfile': ('ruby', 'Gemfile'),
    'Rakefile': ('ruby', 'Rakefile'),
    'Podfile': ('ruby', 'Podfile'),
    'Brewfile': ('ruby', 'Brewfile'),
    'Jenkinsfile': ('groovy', 'Jenkinsfile'),
    'BUILD': ('starlark', 'Bazel BUILD'),
    'WORKSPACE': ('starlark', 'Bazel WORKSPACE'),
    '.gitignore': ('gitignore', 'Git Ignore'),
    '.dockerignore': ('dockerignore', 'Docker Ignore'),
    '.prettierrc': ('json', 'Prettier Config'),
    '.eslintrc': ('json', 'ESLint Config'),
    '.babelrc': ('json', 'Babel Config'),
    'package.json': ('json', 'NPM Package'),
    'tsconfig.json': ('json', 'TypeScript Config'),
    'composer.json': ('json', 'Composer'),
    'Cargo.toml': ('toml', 'Cargo'),
    'pyproject.toml': ('toml', 'Python Project'),
    'go.mod': ('go', 'Go Module'),
    'go.sum': ('go', 'Go Sum'),
    'requirements.txt': ('pip', 'Requirements'),
    'setup.py': ('python', 'Setup Script'),
    'setup.cfg': ('ini', 'Setup Config'),
}

BINARY_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.a', '.lib', '.o', '.obj',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.svg',
    '.mp3', '.mp4', '.wav', '.ogg', '.flac', '.avi', '.mkv', '.mov',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.pyc', '.pyo', '.class', '.jar', '.war', '.ear',
    '.woff', '.woff2', '.ttf', '.otf', '.eot',
    '.db', '.sqlite', '.sqlite3', '.mdb',
    '.lock', '.bin', '.dat',
}

SKIP_DIRECTORIES = {
    'node_modules', '__pycache__', '.git', '.svn', '.hg', '.bzr',
    'venv', '.venv', 'env', '.env', 'virtualenv',
    'dist', 'build', 'target', 'out', 'bin', 'obj',
    '.next', '.nuxt', '.cache', '.parcel-cache',
    'vendor', 'bower_components',
    '.idea', '.vscode', '.vs',
    'coverage', '.nyc_output', 'htmlcov',
    '.tox', '.pytest_cache', '.mypy_cache',
    'eggs', '*.egg-info', '.eggs',
    '.terraform', '.serverless',
}

MAX_FILE_SIZE = 1024 * 1024  # 1MB
MIN_CONTENT_LENGTH = 10


class UniversalFileProcessor:
    """
    Universal file processor that handles all file types,
    directories, zip archives, and tar archives.
    """
    
    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        min_content_length: int = MIN_CONTENT_LENGTH,
        include_binary: bool = False,
        skip_dirs: Optional[Set[str]] = None,
        allowed_extensions: Optional[Set[str]] = None,
        excluded_extensions: Optional[Set[str]] = None,
    ):
        self.max_file_size = max_file_size
        self.min_content_length = min_content_length
        self.include_binary = include_binary
        self.skip_dirs = skip_dirs or SKIP_DIRECTORIES
        self.allowed_extensions = allowed_extensions
        self.excluded_extensions = excluded_extensions or BINARY_EXTENSIONS
        self.errors: List[str] = []
    
    def process(self, source: Union[str, Path, bytes, io.BytesIO]) -> ProcessingResult:
        """
        Process files from any source: directory, file, zip, or bytes.
        
        Args:
            source: Path to directory/file, zip bytes, or BytesIO object
            
        Returns:
            ProcessingResult with all processed files
        """
        self.errors = []
        files: List[ProcessedFile] = []
        total = 0
        processed = 0
        skipped = 0
        
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.is_dir():
                files, total, processed, skipped = self._process_directory(source_path)
            elif source_path.suffix.lower() in ('.zip', '.tar', '.gz', '.tgz', '.tar.gz', '.tar.bz2', '.tar.xz'):
                files, total, processed, skipped = self._process_archive(source_path)
            elif source_path.is_file():
                result = self._process_single_file(source_path)
                if result:
                    files.append(result)
                    processed = 1
                else:
                    skipped = 1
                total = 1
        elif isinstance(source, (bytes, io.BytesIO)):
            files, total, processed, skipped = self._process_bytes(source)
        
        stats = self._calculate_stats(files)
        
        return ProcessingResult(
            files=files,
            total_files=total,
            processed_files=processed,
            skipped_files=skipped,
            errors=self.errors,
            stats=stats,
        )
    
    def _process_directory(self, directory: Path) -> Tuple[List[ProcessedFile], int, int, int]:
        """Process all files in a directory recursively."""
        files = []
        total = 0
        processed = 0
        skipped = 0
        
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in self.skip_dirs and not d.startswith('.')]
            
            for filename in filenames:
                total += 1
                file_path = Path(root) / filename
                
                try:
                    result = self._process_single_file(file_path)
                    if result:
                        files.append(result)
                        processed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    self.errors.append(f"Error processing {file_path}: {str(e)}")
                    skipped += 1
        
        return files, total, processed, skipped
    
    def _get_archive_type(self, path: Path) -> str:
        """Determine archive type handling multi-extension files like .tar.gz"""
        name = path.name.lower()
        
        if name.endswith('.tar.gz') or name.endswith('.tgz'):
            return 'tar'
        if name.endswith('.tar.bz2') or name.endswith('.tbz2'):
            return 'tar'
        if name.endswith('.tar.xz') or name.endswith('.txz'):
            return 'tar'
        if name.endswith('.tar'):
            return 'tar'
        if name.endswith('.zip'):
            return 'zip'
        if name.endswith('.gz') and not name.endswith('.tar.gz'):
            return 'tar'  # Could be gzipped single file
        
        return 'unknown'
    
    def _process_archive(self, archive_path: Path) -> Tuple[List[ProcessedFile], int, int, int]:
        """Process files from a zip or tar archive."""
        archive_type = self._get_archive_type(archive_path)
        
        if archive_type == 'zip':
            return self._process_zip(archive_path)
        elif archive_type == 'tar':
            return self._process_tar(archive_path)
        else:
            self.errors.append(f"Unknown archive format: {archive_path.suffix}")
            return [], 0, 0, 0
    
    def _process_zip(self, zip_path: Path) -> Tuple[List[ProcessedFile], int, int, int]:
        """Process files from a zip archive."""
        files = []
        total = 0
        processed = 0
        skipped = 0
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    
                    total += 1
                    
                    if self._should_skip_archive_path(info.filename):
                        skipped += 1
                        continue
                    
                    try:
                        result = self._process_archive_file(
                            info.filename,
                            zf.read(info.filename),
                            info.file_size
                        )
                        if result:
                            files.append(result)
                            processed += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        self.errors.append(f"Error in zip {info.filename}: {str(e)}")
                        skipped += 1
        except zipfile.BadZipFile as e:
            self.errors.append(f"Invalid zip file: {str(e)}")
        
        return files, total, processed, skipped
    
    def _process_tar(self, tar_path: Path) -> Tuple[List[ProcessedFile], int, int, int]:
        """Process files from a tar archive."""
        files = []
        total = 0
        processed = 0
        skipped = 0
        
        mode = 'r:*'  # Auto-detect compression
        
        try:
            with tarfile.open(tar_path, mode) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    
                    total += 1
                    
                    if self._should_skip_archive_path(member.name):
                        skipped += 1
                        continue
                    
                    try:
                        f = tf.extractfile(member)
                        if f:
                            content = f.read()
                            result = self._process_archive_file(
                                member.name,
                                content,
                                member.size
                            )
                            if result:
                                files.append(result)
                                processed += 1
                            else:
                                skipped += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        self.errors.append(f"Error in tar {member.name}: {str(e)}")
                        skipped += 1
        except tarfile.TarError as e:
            self.errors.append(f"Invalid tar file: {str(e)}")
        
        return files, total, processed, skipped
    
    def _process_bytes(self, data: Union[bytes, io.BytesIO]) -> Tuple[List[ProcessedFile], int, int, int]:
        """Process files from bytes (assumes zip format)."""
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        
        files = []
        total = 0
        processed = 0
        skipped = 0
        
        try:
            with zipfile.ZipFile(data, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    
                    total += 1
                    
                    if self._should_skip_archive_path(info.filename):
                        skipped += 1
                        continue
                    
                    try:
                        result = self._process_archive_file(
                            info.filename,
                            zf.read(info.filename),
                            info.file_size
                        )
                        if result:
                            files.append(result)
                            processed += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        self.errors.append(f"Error processing {info.filename}: {str(e)}")
                        skipped += 1
        except zipfile.BadZipFile:
            try:
                data.seek(0)
                with tarfile.open(fileobj=data, mode='r:*') as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        
                        total += 1
                        
                        if self._should_skip_archive_path(member.name):
                            skipped += 1
                            continue
                        
                        try:
                            f = tf.extractfile(member)
                            if f:
                                content = f.read()
                                result = self._process_archive_file(
                                    member.name,
                                    content,
                                    member.size
                                )
                                if result:
                                    files.append(result)
                                    processed += 1
                                else:
                                    skipped += 1
                            else:
                                skipped += 1
                        except Exception as e:
                            self.errors.append(f"Error: {str(e)}")
                            skipped += 1
            except Exception as e:
                self.errors.append(f"Could not parse bytes as archive: {str(e)}")
        
        return files, total, processed, skipped
    
    def _should_skip_archive_path(self, path: str) -> bool:
        """Check if an archive path should be skipped (includes security checks)."""
        if os.path.isabs(path):
            return True
        
        if '..' in path:
            return True
        
        normalized = os.path.normpath(path)
        if normalized.startswith('..') or normalized.startswith('/'):
            return True
        
        parts = Path(path).parts
        for part in parts:
            if part in self.skip_dirs or part.startswith('.'):
                return True
            if part == '..':
                return True
        
        return False
    
    def _process_single_file(self, file_path: Path) -> Optional[ProcessedFile]:
        """Process a single file."""
        if not file_path.exists():
            return None
        
        ext = file_path.suffix.lower()
        filename = file_path.name
        
        if not self.include_binary and ext in self.excluded_extensions:
            return None
        
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return None
        
        try:
            size = file_path.stat().st_size
            if size > self.max_file_size:
                return None
            
            lang, file_type = self._detect_language(filename, ext)
            
            content = self._read_file(file_path)
            if not content or len(content.strip()) < self.min_content_length:
                return None
            
            return ProcessedFile(
                path=str(file_path),
                content=content,
                file_type=file_type,
                language=lang,
                size=size,
            )
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {str(e)}")
            return None
    
    def _process_archive_file(self, path: str, data: bytes, size: int) -> Optional[ProcessedFile]:
        """Process a file from an archive."""
        filename = Path(path).name
        ext = Path(path).suffix.lower()
        
        if not self.include_binary and ext in self.excluded_extensions:
            return None
        
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return None
        
        if size > self.max_file_size:
            return None
        
        lang, file_type = self._detect_language(filename, ext)
        
        content = self._decode_content(data)
        if not content or len(content.strip()) < self.min_content_length:
            return None
        
        return ProcessedFile(
            path=path,
            content=content,
            file_type=file_type,
            language=lang,
            size=size,
        )
    
    def _detect_language(self, filename: str, ext: str) -> Tuple[str, str]:
        """Detect the programming language and file type."""
        if filename in SPECIAL_FILENAMES:
            return SPECIAL_FILENAMES[filename]
        
        lower_name = filename.lower()
        sorted_exts = sorted(FILE_TYPE_MAPPING.keys(), key=len, reverse=True)
        for multi_ext in sorted_exts:
            if lower_name.endswith(multi_ext):
                return FILE_TYPE_MAPPING[multi_ext]
        
        return ('unknown', 'Unknown')
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding detection."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception:
                return None
        
        return None
    
    def _decode_content(self, data: bytes) -> Optional[str]:
        """Decode bytes content with encoding detection."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        return None
    
    def _calculate_stats(self, files: List[ProcessedFile]) -> Dict:
        """Calculate statistics for processed files."""
        if not files:
            return {
                'total_size': 0,
                'total_content_length': 0,
                'languages': {},
                'file_types': {},
            }
        
        languages: Dict[str, int] = {}
        file_types: Dict[str, int] = {}
        total_size = 0
        total_content = 0
        
        for f in files:
            total_size += f.size
            total_content += len(f.content)
            languages[f.language] = languages.get(f.language, 0) + 1
            file_types[f.file_type] = file_types.get(f.file_type, 0) + 1
        
        return {
            'total_size': total_size,
            'total_content_length': total_content,
            'avg_file_size': total_size // len(files),
            'avg_content_length': total_content // len(files),
            'languages': dict(sorted(languages.items(), key=lambda x: -x[1])),
            'file_types': dict(sorted(file_types.items(), key=lambda x: -x[1])),
        }


def process_for_training(source: Union[str, Path, bytes], 
                         languages: Optional[List[str]] = None) -> List[str]:
    """
    Process files for AI model training.
    
    Args:
        source: Directory path, file path, or zip bytes
        languages: Optional list of languages to include
        
    Returns:
        List of code content strings for training
    """
    processor = UniversalFileProcessor()
    result = processor.process(source)
    
    training_samples = []
    for f in result.files:
        if languages and f.language not in languages:
            continue
        
        content = f.content.strip()
        if len(content) >= 50:  # Minimum useful content
            training_samples.append(content)
    
    return training_samples


def extract_code_blocks(content: str, language: str) -> List[str]:
    """
    Extract meaningful code blocks from content.
    Useful for splitting large files into training samples.
    """
    blocks = []
    
    if language == 'python':
        pattern = r'(?:^|\n)((?:def |class |async def )[^\n]+(?:\n(?:    [^\n]*|\n))*)'
        matches = re.findall(pattern, content, re.MULTILINE)
        blocks.extend(m.strip() for m in matches if len(m.strip()) > 50)
    
    elif language in ('javascript', 'typescript'):
        pattern = r'(?:^|\n)((?:function |class |const \w+ = (?:async )?\([^)]*\) =>|async function )[^\n]+(?:\n(?:  [^\n]*|\n))*)'
        matches = re.findall(pattern, content, re.MULTILINE)
        blocks.extend(m.strip() for m in matches if len(m.strip()) > 50)
    
    elif language == 'sql':
        pattern = r'(?:^|\n)((?:CREATE |SELECT |INSERT |UPDATE |DELETE |ALTER |DROP |WITH )[^;]+;)'
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        blocks.extend(m.strip() for m in matches if len(m.strip()) > 30)
    
    elif language == 'terraform':
        pattern = r'(?:^|\n)((?:resource |data |module |variable |output |provider |locals )"[^"]+"\s*\{[^}]+\})'
        matches = re.findall(pattern, content, re.MULTILINE)
        blocks.extend(m.strip() for m in matches if len(m.strip()) > 50)
    
    if not blocks:
        chunks = content.split('\n\n')
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) >= 50:
                blocks.append(chunk)
    
    return blocks


def get_supported_extensions() -> Dict[str, str]:
    """Get all supported file extensions and their language mappings."""
    result = {}
    for ext, (lang, file_type) in FILE_TYPE_MAPPING.items():
        result[ext] = f"{lang} ({file_type})"
    return result


def print_processing_summary(result: ProcessingResult):
    """Print a summary of the processing result."""
    print(f"\n{'='*60}")
    print("File Processing Summary")
    print(f"{'='*60}")
    print(f"Total files found: {result.total_files}")
    print(f"Successfully processed: {result.processed_files}")
    print(f"Skipped: {result.skipped_files}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.stats.get('languages'):
        print("Languages detected:")
        for lang, count in result.stats['languages'].items():
            print(f"  - {lang}: {count} files")
    
    if result.stats.get('total_content_length'):
        print(f"\nTotal content: {result.stats['total_content_length']:,} characters")
        print(f"Average file size: {result.stats.get('avg_file_size', 0):,} bytes")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"  - {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python file_processor.py <directory_or_zip>")
        print("\nSupported extensions:")
        for ext, desc in sorted(get_supported_extensions().items()):
            print(f"  {ext}: {desc}")
        sys.exit(1)
    
    source = sys.argv[1]
    processor = UniversalFileProcessor()
    result = processor.process(source)
    print_processing_summary(result)
    
    samples = process_for_training(source)
    print(f"Training samples extracted: {len(samples)}")
