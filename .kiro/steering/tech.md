# Technology Stack

## Core Framework
- **Language**: Rust (2021 edition)
- **ML Library**: Candle (Hugging Face's deep learning framework)
  - `candle-core`: Core tensor operations and device management
  - `candle-nn`: Neural network building blocks

## Dependencies
- **Error Handling**: `anyhow` for ergonomic error management
- **CLI**: `clap` with derive features for command-line interfaces
- **System Info**: `num_cpus` for CPU thread detection

## Backend Support
- **Metal**: Apple Silicon GPU acceleration (default feature)
- **CUDA**: NVIDIA GPU support (optional feature)
- **CPU**: Fallback CPU backend with multi-threading

## Build System

### Common Commands
```bash
# Build with default Metal backend
cargo build

# Build with CUDA support
cargo build --features cuda

# Build with Intel MKL optimization
cargo build --features mkl

# Run the application
cargo run

# Run tests
cargo test

# Run with release optimizations
cargo run --release
```

### Feature Flags
- `default = ["metal"]`: Metal backend enabled by default
- `cuda`: Enable NVIDIA CUDA support
- `metal`: Enable Apple Metal support  
- `mkl`: Enable Intel Math Kernel Library optimizations

## Architecture Patterns
- **Trait-based Design**: Exercise system uses traits for extensibility
- **Backend Abstraction**: Device management abstracted through BackendManager
- **Performance Monitoring**: Built-in timing and metrics collection
- **Modular Structure**: Clear separation between exercises, backend, and performance modules