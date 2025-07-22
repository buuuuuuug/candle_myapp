# Project Structure

## Root Level
- `Cargo.toml`: Project configuration with Candle dependencies and feature flags
- `Cargo.lock`: Dependency lock file
- `src/main.rs`: Application entry point with demo and framework initialization

## Source Organization (`src/`)

### Core Modules
- `backend.rs`: Device management and backend detection
  - `BackendManager`: Handles CUDA/Metal/CPU device selection
  - `BackendType`: Enum for backend identification
  - Automatic fallback hierarchy: CUDA → Metal → CPU

- `exercise.rs`: Exercise framework foundation
  - `Exercise` trait: Interface for all learning exercises
  - `ExerciseCategory`: Groups related exercises
  - `ExerciseFramework`: Main orchestration system
  - `ExerciseResult`: Standardized output format

- `performance.rs`: Performance monitoring system
  - `PerformanceMonitor`: Metrics collection and analysis
  - `OperationStats`: Statistical analysis of operations
  - `time_operation!` macro: Convenient timing wrapper

### Exercise Implementations (`src/exercises/`)
- `mod.rs`: Module declarations
- `tensor_basics.rs`: Fundamental tensor operations
  - `TensorCreationExercise`: Basic tensor creation patterns
  - `TensorFromDataExercise`: Data source integration
  - `TensorIndexingExercise`: Slicing and indexing operations
  - `TensorSlicingPatternsExercise`: Advanced slicing techniques
  - `TensorShapeExercise`: Shape manipulation operations

### Test Organization
- `src/backend/tests.rs`: Backend functionality tests
- `src/exercise/tests.rs`: Exercise framework tests
- `src/performance/tests.rs`: Performance monitoring tests

## Conventions
- **Module Structure**: Each major component has its own module with optional test submodule
- **Error Handling**: Consistent use of `anyhow::Result` throughout
- **Naming**: Clear, descriptive names following Rust conventions
- **Documentation**: Educational focus with comprehensive examples and explanatory notes
- **Testing**: Parallel test structure mirroring main source organization