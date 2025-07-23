mod backend;
mod exercise;
mod exercises;
mod performance;

use anyhow::Result;
use backend::BackendManager;
use exercise::{ExerciseFramework, ExerciseCategory};
use performance::PerformanceMonitor;
use candle_core::Tensor;

fn main() -> Result<()> {
    println!("🚀 Candle Practice Application");
    println!("===============================");
    
    // Initialize backend manager with automatic device detection
    let backend_manager = BackendManager::new()?;
    backend_manager.display_status();
    
    // Initialize performance monitor
    let mut performance_monitor = PerformanceMonitor::new(
        backend_manager.get_backend_type().clone()
    );
    
    // Get the device for tensor operations
    let device = backend_manager.get_device();
    
    // Demo: Create and manipulate tensors with performance monitoring
    println!("📊 Basic Tensor Operations Demo (with Performance Monitoring):");
    
    // Create a simple tensor with timing
    let (tensor, duration) = time_operation!(performance_monitor, "tensor_creation", {
        let vec = vec![1u32, 2, 3, 4, 5, 6];
        Tensor::from_vec(vec, (2, 3), device)?
    });
    println!("   Original tensor (2x3): {:?} (created in {:.2}ns)", tensor, duration.as_nanos());
    
    // Convert to 2D vector with timing
    let (vec_2d, duration) = time_operation!(performance_monitor, "tensor_to_vec2", {
        tensor.to_vec2::<u32>()?
    });
    println!("   As 2D vector: {:?} (converted in {:.2}ns)", vec_2d, duration.as_nanos());
    
    // Create a float tensor for more operations
    let (float_tensor, duration) = time_operation!(performance_monitor, "float_tensor_creation", {
        let float_data = vec![1.0f32, 2.0, 3.0, 4.0];
        Tensor::from_vec(float_data, (2, 2), device)?
    });
    println!("   Float tensor (2x2): {:?} (created in {:.2}ns)", float_tensor, duration.as_nanos());
    
    // Demonstrate transpose with timing
    let (transposed, duration) = time_operation!(performance_monitor, "tensor_transpose", {
        float_tensor.transpose(0, 1)?
    });
    println!("   Transposed: {:?} (transposed in {:.2}ns)", transposed, duration.as_nanos());
    
    // Demonstrate matrix multiplication with timing
    let (matrix_result, duration) = time_operation!(performance_monitor, "matrix_multiplication", {
        float_tensor.matmul(&transposed)?
    });
    println!("   Matrix multiplication result: {:?} (computed in {:.2}ns)", matrix_result, duration.as_nanos());
    
    println!("\n✅ Backend manager working correctly!");
    
    // Initialize exercise framework
    println!("🏗️  Initializing Exercise Framework...");
    let mut framework = ExerciseFramework::new();
    
    // Create categories and add exercises
    let mut basic_tensors = ExerciseCategory::new(
        "Basic Tensors", 
        "Fundamental tensor creation and manipulation"
    );
    
    // Add tensor creation exercises
    use exercises::tensor_basics::{TensorCreationExercise, TensorFromDataExercise, TensorIndexingExercise, TensorSlicingPatternsExercise, TensorShapeExercise};
    basic_tensors.add_exercise(Box::new(TensorCreationExercise));
    basic_tensors.add_exercise(Box::new(TensorFromDataExercise));
    basic_tensors.add_exercise(Box::new(TensorIndexingExercise));
    basic_tensors.add_exercise(Box::new(TensorSlicingPatternsExercise));
    basic_tensors.add_exercise(Box::new(TensorShapeExercise));
    
    let mut matrix_ops = ExerciseCategory::new(
        "Matrix Operations", 
        "Linear algebra operations and transformations"
    );
    
    // Add matrix operation exercises
    use exercises::matrix_operations::{MatrixArithmeticExercise, BroadcastingExercise};
    matrix_ops.add_exercise(Box::new(MatrixArithmeticExercise));
    matrix_ops.add_exercise(Box::new(BroadcastingExercise));
    
    let neural_networks = ExerciseCategory::new(
        "Neural Networks", 
        "Neural network components and training"
    );
    
    framework.add_category(basic_tensors);
    framework.add_category(matrix_ops);
    framework.add_category(neural_networks);
    
    // Display the framework menu
    framework.display_menu();
    
    println!("✅ Exercise framework initialized successfully!");
    println!("📚 {} categories available", framework.list_categories().len());
    
    // Display performance metrics
    println!("\n🔍 Performance Analysis:");
    performance_monitor.display_metrics();
    
    // Show operation comparison
    let operations = ["tensor_creation", "tensor_transpose", "matrix_multiplication"];
    performance_monitor.display_comparison(&operations);
    
    println!("✅ Performance monitoring system working correctly!");
    
    // Run a matrix arithmetic exercise as a demo
    println!("\n🧪 Running Matrix Arithmetic Exercise Demo:");
    let category_name = "Matrix Operations";
    let exercise_name = "Matrix Arithmetic Operations";
    
    match framework.run_exercise(category_name, exercise_name, device) {
        Ok(result) => {
            println!("✅ Exercise completed successfully!");
            println!("📝 Output preview (truncated):");
            
            // Display just the first few lines of output for preview
            let preview_lines: Vec<&str> = result.output.lines().take(10).collect();
            for line in preview_lines {
                println!("   {}", line);
            }
            
            if result.output.lines().count() > 10 {
                println!("   ... (output truncated)");
            }
            
            println!("\n📊 Created {} tensors", result.tensors.len());
            println!("📚 {} educational notes", result.educational_notes.len());
        },
        Err(e) => {
            println!("❌ Failed to run exercise: {}", e);
        }
    }
    
    println!("\nReady to explore more exercises...");
    
    Ok(())
}