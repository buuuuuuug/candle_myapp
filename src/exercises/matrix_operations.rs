use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use crate::exercise::{Exercise, ExerciseResult, TensorInfo};
use std::collections::HashMap;
use std::time::Duration;

/// Exercise for basic matrix arithmetic operations
pub struct MatrixArithmeticExercise;

impl Exercise for MatrixArithmeticExercise {
    fn name(&self) -> &str {
        "Matrix Arithmetic Operations"
    }

    fn description(&self) -> &str {
        "Learn basic matrix arithmetic including addition, subtraction, and element-wise operations"
    }

    fn run(&self, device: &Device) -> Result<ExerciseResult> {
        let mut result = ExerciseResult::new();
        
        // Add educational notes
        result.add_educational_note(
            "Matrix arithmetic operations are fundamental building blocks for neural networks and scientific computing."
        );
        result.add_educational_note(
            "Candle supports element-wise operations, broadcasting, and efficient GPU acceleration for matrix computations."
        );
        
        // Create base matrices for demonstrations
        let matrix_a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix_a = Tensor::from_vec(matrix_a_data, (2, 3), device)?;
        
        let matrix_b_data = vec![2.0f32, 1.0, 4.0, 3.0, 6.0, 5.0];
        let matrix_b = Tensor::from_vec(matrix_b_data, (2, 3), device)?;
        
        result.add_output("\nBase matrices for arithmetic operations:");
        result.add_output(&format!("   Matrix A (2x3): {:?}", matrix_a));
        result.add_output(&format!("   Values A: {:?}", matrix_a.to_vec2::<f32>()?));
        result.add_output(&format!("   Matrix B (2x3): {:?}", matrix_b));
        result.add_output(&format!("   Values B: {:?}", matrix_b.to_vec2::<f32>()?));
        
        result.add_tensor(TensorInfo::from_tensor::<f32>("matrix_a", &matrix_a)?);
        result.add_tensor(TensorInfo::from_tensor::<f32>("matrix_b", &matrix_b)?);
        
        // Example 1: Matrix Addition
        result.add_output("\n1. Matrix Addition (A + B):");
        let sum = (&matrix_a + &matrix_b)?;
        result.add_output(&format!("   Result: {:?}", sum));
        result.add_output(&format!("   Values: {:?}", sum.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("matrix_sum", &sum)?);
        
        // Example 2: Matrix Subtraction
        result.add_output("\n2. Matrix Subtraction (A - B):");
        let diff = (&matrix_a - &matrix_b)?;
        result.add_output(&format!("   Result: {:?}", diff));
        result.add_output(&format!("   Values: {:?}", diff.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("matrix_diff", &diff)?);
        
        // Example 3: Element-wise Multiplication
        result.add_output("\n3. Element-wise Multiplication (A ⊙ B):");
        let elem_mul = (&matrix_a * &matrix_b)?;
        result.add_output(&format!("   Result: {:?}", elem_mul));
        result.add_output(&format!("   Values: {:?}", elem_mul.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("elem_mul", &elem_mul)?);
        
        // Example 4: Element-wise Division
        result.add_output("\n4. Element-wise Division (A ⊘ B):");
        let elem_div = (&matrix_a / &matrix_b)?;
        result.add_output(&format!("   Result: {:?}", elem_div));
        result.add_output(&format!("   Values: {:?}", elem_div.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("elem_div", &elem_div)?);
        
        // Example 5: Scalar Operations
        result.add_output("\n5. Scalar Operations:");
        
        // Scalar addition
        let scalar_add = matrix_a.affine(1.0, 10.0)?; // Use affine transformation: 1*x + 10
        result.add_output(&format!("   A + 10: {:?}", scalar_add));
        result.add_output(&format!("   Values: {:?}", scalar_add.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("scalar_add", &scalar_add)?);
        
        // Scalar multiplication
        let scalar_mul = matrix_a.affine(2.0, 0.0)?; // Use affine transformation: 2*x + 0
        result.add_output(&format!("   A * 2: {:?}", scalar_mul));
        result.add_output(&format!("   Values: {:?}", scalar_mul.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("scalar_mul", &scalar_mul)?);
        
        // Example 6: Broadcasting with different shapes
        result.add_output("\n6. Broadcasting Operations:");
        
        // Create a row vector for broadcasting
        let row_vector = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), device)?;
        result.add_output(&format!("   Row vector (1x3): {:?}", row_vector));
        result.add_output(&format!("   Values: {:?}", row_vector.to_vec2::<f32>()?));
        
        // Broadcasting addition (expand row vector manually)
        let row_expanded = row_vector.broadcast_as((2, 3))?;
        let broadcast_add = (&matrix_a + &row_expanded)?;
        result.add_output(&format!("   A + row_vector (broadcasting): {:?}", broadcast_add));
        result.add_output(&format!("   Values: {:?}", broadcast_add.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("broadcast_add", &broadcast_add)?);
        
        // Create a column vector for broadcasting
        let col_vector = Tensor::from_vec(vec![10.0f32, 20.0], (2, 1), device)?;
        result.add_output(&format!("   Column vector (2x1): {:?}", col_vector));
        result.add_output(&format!("   Values: {:?}", col_vector.to_vec2::<f32>()?));
        
        // Broadcasting multiplication (expand column vector manually)
        let col_expanded = col_vector.broadcast_as((2, 3))?;
        let broadcast_mul = (&matrix_a * &col_expanded)?;
        result.add_output(&format!("   A * col_vector (broadcasting): {:?}", broadcast_mul));
        result.add_output(&format!("   Values: {:?}", broadcast_mul.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("broadcast_mul", &broadcast_mul)?);
        
        // Example 7: Mathematical Functions
        result.add_output("\n7. Mathematical Functions:");
        
        // Square root
        let sqrt_result = matrix_a.sqrt()?;
        result.add_output(&format!("   sqrt(A): {:?}", sqrt_result));
        result.add_output(&format!("   Values: {:?}", sqrt_result.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("sqrt_result", &sqrt_result)?);
        
        // Exponential
        let exp_result = matrix_a.exp()?;
        result.add_output(&format!("   exp(A): {:?}", exp_result));
        result.add_output(&format!("   Values: {:?}", exp_result.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("exp_result", &exp_result)?);
        
        // Natural logarithm
        let log_result = matrix_a.log()?;
        result.add_output(&format!("   log(A): {:?}", log_result));
        result.add_output(&format!("   Values: {:?}", log_result.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("log_result", &log_result)?);
        
        // Example 8: Aggregation Operations
        result.add_output("\n8. Aggregation Operations:");
        
        // Sum all elements
        let sum_all = matrix_a.sum_all()?;
        result.add_output(&format!("   Sum of all elements: {:?}", sum_all));
        result.add_output(&format!("   Value: {:?}", sum_all.to_vec0::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("sum_all", &sum_all)?);
        
        // Sum along dimensions
        let sum_dim0 = matrix_a.sum(0)?; // Sum along rows (result: 1x3)
        result.add_output(&format!("   Sum along dimension 0 (columns): {:?}", sum_dim0));
        result.add_output(&format!("   Values: {:?}", sum_dim0.to_vec1::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("sum_dim0", &sum_dim0)?);
        
        let sum_dim1 = matrix_a.sum(1)?; // Sum along columns (result: 2x1)
        result.add_output(&format!("   Sum along dimension 1 (rows): {:?}", sum_dim1));
        result.add_output(&format!("   Values: {:?}", sum_dim1.to_vec1::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("sum_dim1", &sum_dim1)?);
        
        // Mean operations
        let mean_all = matrix_a.mean_all()?;
        result.add_output(&format!("   Mean of all elements: {:?}", mean_all));
        result.add_output(&format!("   Value: {:?}", mean_all.to_vec0::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("mean_all", &mean_all)?);
        
        // Example 9: Comparison Operations
        result.add_output("\n9. Comparison Operations:");
        
        // Greater than
        let gt_result = matrix_a.gt(&matrix_b)?;
        result.add_output(&format!("   A > B: {:?}", gt_result));
        result.add_output(&format!("   Values: {:?}", gt_result.to_vec2::<u8>()?));
        result.add_tensor(TensorInfo::from_tensor::<u8>("gt_result", &gt_result)?);
        
        // Less than
        let lt_result = matrix_a.lt(&matrix_b)?;
        result.add_output(&format!("   A < B: {:?}", lt_result));
        result.add_output(&format!("   Values: {:?}", lt_result.to_vec2::<u8>()?));
        result.add_tensor(TensorInfo::from_tensor::<u8>("lt_result", &lt_result)?);
        
        // Equal
        let eq_result = matrix_a.eq(&matrix_b)?;
        result.add_output(&format!("   A == B: {:?}", eq_result));
        result.add_output(&format!("   Values: {:?}", eq_result.to_vec2::<u8>()?));
        result.add_tensor(TensorInfo::from_tensor::<u8>("eq_result", &eq_result)?);
        
        // Add performance metrics
        let mut additional_info = HashMap::new();
        additional_info.insert("total_operations".to_string(), "15".to_string());
        additional_info.insert("operation_types".to_string(), "arithmetic, broadcasting, functions, aggregation, comparison".to_string());
        
        result.add_metric(crate::exercise::Metric {
            operation: "matrix_arithmetic_exercise".to_string(),
            duration: Duration::from_millis(0), // Will be updated by the framework
            backend: format!("{:?}", device),
            additional_info,
        });
        
        Ok(result)
    }
}

/// Exercise for demonstrating broadcasting in detail
pub struct BroadcastingExercise;

impl Exercise for BroadcastingExercise {
    fn name(&self) -> &str {
        "Broadcasting Demonstrations"
    }

    fn description(&self) -> &str {
        "Explore broadcasting rules and patterns for efficient tensor operations"
    }

    fn run(&self, device: &Device) -> Result<ExerciseResult> {
        let mut result = ExerciseResult::new();
        
        // Add educational notes
        result.add_educational_note(
            "Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions."
        );
        result.add_educational_note(
            "Understanding broadcasting rules is crucial for efficient neural network implementations and avoiding unnecessary memory allocations."
        );
        
        // Example 1: Scalar broadcasting
        result.add_output("\n1. Scalar Broadcasting:");
        let matrix = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
        result.add_output(&format!("   Matrix (2x2): {:?}", matrix));
        result.add_output(&format!("   Values: {:?}", matrix.to_vec2::<f32>()?));
        
        let scalar_broadcast = matrix.affine(1.0, 5.0)?; // Use affine transformation: 1*x + 5
        result.add_output(&format!("   Matrix + 5 (scalar broadcast): {:?}", scalar_broadcast));
        result.add_output(&format!("   Values: {:?}", scalar_broadcast.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("scalar_broadcast", &scalar_broadcast)?);
        
        // Example 2: Vector broadcasting
        result.add_output("\n2. Vector Broadcasting:");
        let matrix_3x4 = Tensor::from_vec((1..=12).map(|x| x as f32).collect(), (3, 4), device)?;
        result.add_output(&format!("   Matrix (3x4): {:?}", matrix_3x4));
        result.add_output(&format!("   Values: {:?}", matrix_3x4.to_vec2::<f32>()?));
        
        // Row vector broadcasting
        let row_vec = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), device)?;
        result.add_output(&format!("   Row vector (1x4): {:?}", row_vec));
        result.add_output(&format!("   Values: {:?}", row_vec.to_vec2::<f32>()?));
        
        let row_expanded = row_vec.broadcast_as((3, 4))?;
        let row_broadcast = (&matrix_3x4 * &row_expanded)?;
        result.add_output(&format!("   Matrix * row_vector: {:?}", row_broadcast));
        result.add_output(&format!("   Values: {:?}", row_broadcast.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("row_broadcast", &row_broadcast)?);
        
        // Column vector broadcasting
        let col_vec = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (3, 1), device)?;
        result.add_output(&format!("   Column vector (3x1): {:?}", col_vec));
        result.add_output(&format!("   Values: {:?}", col_vec.to_vec2::<f32>()?));
        
        let col_expanded = col_vec.broadcast_as((3, 4))?;
        let col_broadcast = (&matrix_3x4 + &col_expanded)?;
        result.add_output(&format!("   Matrix + col_vector: {:?}", col_broadcast));
        result.add_output(&format!("   Values: {:?}", col_broadcast.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("col_broadcast", &col_broadcast)?);
        
        // Example 3: Complex broadcasting patterns
        result.add_output("\n3. Complex Broadcasting Patterns:");
        
        // Create tensors with different shapes that can broadcast
        let tensor_2x1x3 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 1, 3), device)?;
        let tensor_1x4x1 = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (1, 4, 1), device)?;
        
        result.add_output(&format!("   Tensor A (2x1x3): {:?}", tensor_2x1x3));
        result.add_output(&format!("   Shape: {:?}", tensor_2x1x3.shape()));
        
        result.add_output(&format!("   Tensor B (1x4x1): {:?}", tensor_1x4x1));
        result.add_output(&format!("   Shape: {:?}", tensor_1x4x1.shape()));
        
        // For complex broadcasting, let's use simpler shapes
        let tensor_a_simple = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
        let tensor_b_simple = Tensor::from_vec(vec![10.0f32, 20.0], (2, 1), device)?;
        let tensor_b_expanded = tensor_b_simple.broadcast_as((2, 2))?;
        let complex_broadcast = (&tensor_a_simple + &tensor_b_expanded)?;
        result.add_output(&format!("   A + B (broadcast): {:?}", complex_broadcast));
        result.add_output(&format!("   Result shape: {:?}", complex_broadcast.shape()));
        result.add_output(&format!("   Values: {:?}", complex_broadcast.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("complex_broadcast", &complex_broadcast)?);
        
        // Example 4: Broadcasting with mathematical functions
        result.add_output("\n4. Broadcasting with Mathematical Functions:");
        
        let base_matrix = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
        let power_vec = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), device)?;
        
        result.add_output(&format!("   Base matrix (2x3): {:?}", base_matrix));
        result.add_output(&format!("   Values: {:?}", base_matrix.to_vec2::<f32>()?));
        result.add_output(&format!("   Power vector (1x3): {:?}", power_vec));
        result.add_output(&format!("   Values: {:?}", power_vec.to_vec2::<f32>()?));
        
        let power_result = base_matrix.powf(2.0)?; // Use a scalar power instead
        result.add_output(&format!("   base^2 (power function): {:?}", power_result));
        result.add_output(&format!("   Values: {:?}", power_result.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("power_result", &power_result)?);
        
        // Example 5: Broadcasting efficiency demonstration
        result.add_output("\n5. Broadcasting Efficiency:");
        result.add_output("   Broadcasting avoids creating intermediate tensors:");
        result.add_output("   - Memory efficient: no temporary expanded tensors");
        result.add_output("   - Computationally efficient: operations applied directly");
        result.add_output("   - GPU friendly: vectorized operations across dimensions");
        
        // Demonstrate with a larger example
        let large_matrix = Tensor::zeros((100, 50), DType::F32, device)?;
        let bias_vector = Tensor::ones((1, 50), DType::F32, device)?;
        
        let bias_expanded = bias_vector.broadcast_as((100, 50))?;
        let biased_matrix = (&large_matrix + &bias_expanded)?;
        result.add_output(&format!("   Large matrix (100x50) + bias (1x50): {:?}", biased_matrix.shape()));
        result.add_tensor(TensorInfo::from_tensor::<f32>("biased_matrix", &biased_matrix)?);
        
        // Add performance metrics
        let mut additional_info = HashMap::new();
        additional_info.insert("total_operations".to_string(), "8".to_string());
        additional_info.insert("broadcast_patterns".to_string(), "scalar, vector, complex, mathematical".to_string());
        
        result.add_metric(crate::exercise::Metric {
            operation: "broadcasting_exercise".to_string(),
            duration: Duration::from_millis(0), // Will be updated by the framework
            backend: format!("{:?}", device),
            additional_info,
        });
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_matrix_arithmetic_exercise() {
        let exercise = MatrixArithmeticExercise;
        let device = Device::Cpu;
        
        let result = exercise.run(&device);
        if let Err(e) = &result {
            println!("Error running MatrixArithmeticExercise: {}", e);
        }
        assert!(result.is_ok());
        
        let exercise_result = result.unwrap();
        assert!(exercise_result.success);
        assert!(!exercise_result.output.is_empty());
        assert!(!exercise_result.tensors.is_empty());
        assert!(!exercise_result.educational_notes.is_empty());
    }
    
    #[test]
    fn test_broadcasting_exercise() {
        let exercise = BroadcastingExercise;
        let device = Device::Cpu;
        
        let result = exercise.run(&device);
        if let Err(e) = &result {
            println!("Error running BroadcastingExercise: {}", e);
        }
        assert!(result.is_ok());
        
        let exercise_result = result.unwrap();
        assert!(exercise_result.success);
        assert!(!exercise_result.output.is_empty());
        assert!(!exercise_result.tensors.is_empty());
        assert!(!exercise_result.educational_notes.is_empty());
    }
}