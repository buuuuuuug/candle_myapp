use anyhow::Result;
use candle_core::{Device, DType, Tensor, Shape};
use crate::exercise::{Exercise, ExerciseResult, TensorInfo};
use std::collections::HashMap;
use std::time::Duration;

/// Exercise for creating tensors with different data types
pub struct TensorCreationExercise;

impl Exercise for TensorCreationExercise {
    fn name(&self) -> &str {
        "Tensor Creation"
    }

    fn description(&self) -> &str {
        "Create tensors with different data types and shapes"
    }

    fn run(&self, device: &Device) -> Result<ExerciseResult> {
        let mut result = ExerciseResult::new();
        
        // Add educational note
        result.add_educational_note(
            "Tensors are the fundamental data structure in Candle, similar to arrays or matrices but with n-dimensions."
        );
        result.add_educational_note(
            "Candle supports various data types: f32, f64, u8, u32, i64, etc."
        );
        
        // Example 1: Create a tensor from a scalar
        result.add_output("\n1. Creating a tensor from a scalar:");
        let scalar_tensor = Tensor::new(42.0f32, device)?;
        result.add_output(&format!("   Scalar tensor: {:?}", scalar_tensor));
        result.add_tensor(TensorInfo::from_tensor::<f32>("scalar_tensor", &scalar_tensor)?);
        
        // Example 2: Create a tensor from a 1D vector
        result.add_output("\n2. Creating a tensor from a 1D vector:");
        let vec_tensor = Tensor::new(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32], device)?;
        result.add_output(&format!("   Vector tensor: {:?}", vec_tensor));
        result.add_output(&format!("   Shape: {:?}", vec_tensor.shape()));
        result.add_tensor(TensorInfo::from_tensor::<f32>("vec_tensor", &vec_tensor)?);
        
        // Example 3: Create a tensor from a 2D array
        result.add_output("\n3. Creating a tensor from a 2D array:");
        let array_2d = &[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]];
        let matrix_tensor = Tensor::new(array_2d, device)?;
        result.add_output(&format!("   Matrix tensor: {:?}", matrix_tensor));
        result.add_output(&format!("   Shape: {:?}", matrix_tensor.shape()));
        result.add_tensor(TensorInfo::from_tensor::<f32>("matrix_tensor", &matrix_tensor)?);
        
        // Example 4: Create a tensor with explicit shape
        result.add_output("\n4. Creating a tensor with explicit shape:");
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
        let shaped_tensor = Tensor::from_vec(data, (2, 3), device)?;
        result.add_output(&format!("   Shaped tensor (2x3): {:?}", shaped_tensor));
        result.add_output(&format!("   As 2D vector: {:?}", shaped_tensor.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("shaped_tensor", &shaped_tensor)?);
        
        // Example 5: Create tensors with different data types
        result.add_output("\n5. Creating tensors with different data types:");
        
        // Float 32
        let f32_tensor = Tensor::new(&[1.0f32, 2.0, 3.0], device)?;
        result.add_output(&format!("   f32 tensor: {:?}", f32_tensor));
        result.add_output(&format!("   dtype: {:?}", f32_tensor.dtype()));
        result.add_tensor(TensorInfo::from_tensor::<f32>("f32_tensor", &f32_tensor)?);
        
        // Float 64
        let f64_tensor = Tensor::new(&[1.0f64, 2.0f64, 3.0f64], device)?;
        result.add_output(&format!("   f64 tensor: {:?}", f64_tensor));
        result.add_output(&format!("   dtype: {:?}", f64_tensor.dtype()));
        result.add_tensor(TensorInfo::from_tensor::<f64>("f64_tensor", &f64_tensor)?);
        
        // Integer 64 (i32 is not supported by WithDType)
        let i64_tensor = Tensor::new(&[1i64, 2, 3], device)?;
        result.add_output(&format!("   i64 tensor: {:?}", i64_tensor));
        result.add_output(&format!("   dtype: {:?}", i64_tensor.dtype()));
        result.add_tensor(TensorInfo::from_tensor::<i64>("i64_tensor", &i64_tensor)?);
        
        // Unsigned 32
        let u32_tensor = Tensor::new(&[1u32, 2, 3], device)?;
        result.add_output(&format!("   u32 tensor: {:?}", u32_tensor));
        result.add_output(&format!("   dtype: {:?}", u32_tensor.dtype()));
        result.add_tensor(TensorInfo::from_tensor::<u32>("u32_tensor", &u32_tensor)?);
        
        // Example 6: Special tensor creation methods
        result.add_output("\n6. Special tensor creation methods:");
        
        // Zeros
        let zeros = Tensor::zeros((2, 3), DType::F32, device)?;
        result.add_output(&format!("   Zeros (2x3): {:?}", zeros));
        result.add_output(&format!("   Values: {:?}", zeros.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("zeros", &zeros)?);
        
        // Ones
        let ones = Tensor::ones((2, 2), DType::F32, device)?;
        result.add_output(&format!("   Ones (2x2): {:?}", ones));
        result.add_output(&format!("   Values: {:?}", ones.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("ones", &ones)?);
        
        // Random (uniform) - explicitly use f32
        let rand_uniform = Tensor::rand(0.0f32, 1.0f32, (2, 3), device)?;
        result.add_output(&format!("   Random uniform (2x3): {:?}", rand_uniform));
        result.add_output(&format!("   Values: {:?}", rand_uniform.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("rand_uniform", &rand_uniform)?);
        
        // Random (normal) - explicitly use f32
        let rand_normal = Tensor::randn(0.0f32, 1.0f32, (2, 3), device)?;
        result.add_output(&format!("   Random normal (2x3): {:?}", rand_normal));
        result.add_output(&format!("   Values: {:?}", rand_normal.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("rand_normal", &rand_normal)?);
        
        // Example 7: Creating tensors with arange
        result.add_output("\n7. Creating tensors with arange:");
        let arange = Tensor::arange(0.0f32, 10.0f32, device)?;
        result.add_output(&format!("   Arange [0, 10): {:?}", arange));
        result.add_output(&format!("   Values: {:?}", arange.to_vec1::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("arange", &arange)?);
        
        // Example 8: Eye (identity matrix)
        result.add_output("\n8. Creating an identity matrix:");
        let eye = Tensor::eye(3, DType::F32, device)?;
        result.add_output(&format!("   Identity (3x3): {:?}", eye));
        result.add_output(&format!("   Values: {:?}", eye.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("eye", &eye)?);
        
        // Add performance metrics
        let mut additional_info = HashMap::new();
        additional_info.insert("total_tensors".to_string(), "15".to_string());
        
        result.add_metric(crate::exercise::Metric {
            operation: "tensor_creation_exercise".to_string(),
            duration: Duration::from_millis(0), // Will be updated by the framework
            backend: format!("{:?}", device),
            additional_info,
        });
        
        Ok(result)
    }
}

/// Exercise for creating tensors from different data sources
pub struct TensorFromDataExercise;

impl Exercise for TensorFromDataExercise {
    fn name(&self) -> &str {
        "Tensor From Data Sources"
    }

    fn description(&self) -> &str {
        "Create tensors from different data sources like Vec, arrays, and files"
    }

    fn run(&self, device: &Device) -> Result<ExerciseResult> {
        let mut result = ExerciseResult::new();
        
        // Add educational notes
        result.add_educational_note(
            "Tensors can be created from various data sources including Rust vectors, arrays, and even files."
        );
        result.add_educational_note(
            "When creating tensors, you need to specify the shape and ensure the data size matches."
        );
        
        // Example 1: From Vec with shape
        result.add_output("\n1. Creating a tensor from Vec with explicit shape:");
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, (2, 3), device)?;
        result.add_output(&format!("   Tensor: {:?}", tensor));
        result.add_output(&format!("   Shape: {:?}", tensor.shape()));
        result.add_output(&format!("   Values: {:?}", tensor.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("vec_tensor", &tensor)?);
        
        // Example 2: From slice with shape
        result.add_output("\n2. Creating a tensor from slice with explicit shape:");
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, (2, 2), device)?;
        result.add_output(&format!("   Tensor: {:?}", tensor));
        result.add_output(&format!("   Values: {:?}", tensor.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("slice_tensor", &tensor)?);
        
        // Example 3: From nested Vec
        result.add_output("\n3. Creating a tensor from nested Vec:");
        let nested_data = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tensor = Tensor::new(nested_data, device)?;
        result.add_output(&format!("   Tensor: {:?}", tensor));
        result.add_output(&format!("   Values: {:?}", tensor.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("nested_vec_tensor", &tensor)?);
        
        // Example 4: From iterator (collect to vec first since from_iter doesn't take shape)
        result.add_output("\n4. Creating a tensor from iterator:");
        let iter_data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(iter_data, (2, 3), device)?;
        result.add_output(&format!("   Tensor: {:?}", tensor));
        result.add_output(&format!("   Values: {:?}", tensor.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("iter_tensor", &tensor)?);
        
        // Example 5: Creating a tensor with a specific shape
        result.add_output("\n5. Creating a tensor with a specific shape:");
        let shape = Shape::from_dims(&[2, 3, 4]);
        let zeros = Tensor::zeros(shape, DType::F32, device)?;
        result.add_output(&format!("   Tensor shape: {:?}", zeros.shape()));
        result.add_output(&format!("   Dimensions: {}", zeros.dims().len()));
        result.add_output(&format!("   Total elements: {}", zeros.elem_count()));
        result.add_tensor(TensorInfo::from_tensor::<f32>("shaped_zeros", &zeros)?);
        
        // Example 6: Creating tensors with broadcasting in mind
        result.add_output("\n6. Creating tensors for broadcasting:");
        let row_vector = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), device)?;
        let col_vector = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), device)?;
        
        result.add_output(&format!("   Row vector: {:?}", row_vector));
        result.add_output(&format!("   Row vector shape: {:?}", row_vector.shape()));
        result.add_output(&format!("   Row vector values: {:?}", row_vector.to_vec2::<f32>()?));
        
        result.add_output(&format!("   Column vector: {:?}", col_vector));
        result.add_output(&format!("   Column vector shape: {:?}", col_vector.shape()));
        result.add_output(&format!("   Column vector values: {:?}", col_vector.to_vec2::<f32>()?));
        
        result.add_tensor(TensorInfo::from_tensor::<f32>("row_vector", &row_vector)?);
        result.add_tensor(TensorInfo::from_tensor::<f32>("col_vector", &col_vector)?);
        
        // Add performance metrics
        let mut additional_info = HashMap::new();
        additional_info.insert("total_tensors".to_string(), "7".to_string());
        
        result.add_metric(crate::exercise::Metric {
            operation: "tensor_from_data_exercise".to_string(),
            duration: Duration::from_millis(0), // Will be updated by the framework
            backend: format!("{:?}", device),
            additional_info,
        });
        
        Ok(result)
    }
}

/// Exercise for tensor shape manipulation
pub struct TensorShapeExercise;

impl Exercise for TensorShapeExercise {
    fn name(&self) -> &str {
        "Tensor Shape Manipulation"
    }

    fn description(&self) -> &str {
        "Manipulate tensor shapes through reshaping, squeezing, and unsqueezing"
    }

    fn run(&self, device: &Device) -> Result<ExerciseResult> {
        let mut result = ExerciseResult::new();
        
        // Add educational notes
        result.add_educational_note(
            "Tensor shape manipulation is crucial for preparing data for various operations."
        );
        result.add_educational_note(
            "Common operations include reshape, squeeze (remove dimensions of size 1), and unsqueeze (add dimensions of size 1)."
        );
        
        // Create a base tensor for demonstrations
        let base_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let base_tensor = Tensor::from_vec(base_data.clone(), (2, 3), device)?;
        
        result.add_output("\nBase tensor for shape manipulations:");
        result.add_output(&format!("   Tensor: {:?}", base_tensor));
        result.add_output(&format!("   Shape: {:?}", base_tensor.shape()));
        result.add_output(&format!("   Values: {:?}", base_tensor.to_vec2::<f32>()?));
        
        // Example 1: Reshape
        result.add_output("\n1. Reshaping a tensor:");
        let reshaped = base_tensor.reshape((3, 2))?;
        result.add_output(&format!("   Reshaped (3x2): {:?}", reshaped));
        result.add_output(&format!("   New shape: {:?}", reshaped.shape()));
        result.add_output(&format!("   Values: {:?}", reshaped.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("reshaped", &reshaped)?);
        
        // Example 2: Flatten
        result.add_output("\n2. Flattening a tensor:");
        let flattened = base_tensor.flatten_all()?;
        result.add_output(&format!("   Flattened: {:?}", flattened));
        result.add_output(&format!("   New shape: {:?}", flattened.shape()));
        result.add_output(&format!("   Values: {:?}", flattened.to_vec1::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("flattened", &flattened)?);
        
        // Example 3: Squeeze
        result.add_output("\n3. Squeezing a tensor (removing dimensions of size 1):");
        let tensor_with_ones = Tensor::from_vec(base_data.clone(), (1, 2, 3), device)?;
        result.add_output(&format!("   Original: {:?} with shape {:?}", tensor_with_ones, tensor_with_ones.shape()));
        
        let squeezed = tensor_with_ones.squeeze(0)?;
        result.add_output(&format!("   After squeeze(0): {:?} with shape {:?}", squeezed, squeezed.shape()));
        
        // Create another tensor with ones at the end
        let tensor_with_end_one = Tensor::from_vec(base_data.clone(), (2, 3, 1), device)?;
        let squeezed_end = tensor_with_end_one.squeeze(2)?;
        result.add_output(&format!("   Tensor with shape (2,3,1) after squeeze(2): {:?} with shape {:?}", 
                                  squeezed_end, squeezed_end.shape()));
        result.add_tensor(TensorInfo::from_tensor::<f32>("squeezed", &squeezed_end)?);
        
        // Example 4: Unsqueeze
        result.add_output("\n4. Unsqueezing a tensor (adding dimensions of size 1):");
        let unsqueezed_0 = base_tensor.unsqueeze(0)?;
        result.add_output(&format!("   After unsqueeze(0): {:?} with shape {:?}", unsqueezed_0, unsqueezed_0.shape()));
        
        let unsqueezed_2 = base_tensor.unsqueeze(2)?;
        result.add_output(&format!("   After unsqueeze(2): {:?} with shape {:?}", unsqueezed_2, unsqueezed_2.shape()));
        result.add_tensor(TensorInfo::from_tensor::<f32>("unsqueezed", &unsqueezed_2)?);
        
        // Example 5: Broadcast_as (expand dimensions)
        result.add_output("\n5. Broadcasting a tensor to a new shape:");
        let vector = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), device)?;
        result.add_output(&format!("   Original vector: {:?} with shape {:?}", vector, vector.shape()));
        
        let broadcasted = vector.broadcast_as((2, 3))?;
        result.add_output(&format!("   Broadcasted to (2,3): {:?}", broadcasted));
        result.add_output(&format!("   Values: {:?}", broadcasted.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("broadcasted", &broadcasted)?);
        
        // Example 6: View (reshape without copy)
        result.add_output("\n6. Using view for efficient reshaping:");
        let view_reshaped = base_tensor.reshape((6, 1))?;
        result.add_output(&format!("   View as (6,1): {:?} with shape {:?}", view_reshaped, view_reshaped.shape()));
        result.add_output(&format!("   Values: {:?}", view_reshaped.to_vec2::<f32>()?));
        result.add_tensor(TensorInfo::from_tensor::<f32>("view_reshaped", &view_reshaped)?);
        
        // Add performance metrics
        let mut additional_info = HashMap::new();
        additional_info.insert("total_operations".to_string(), "6".to_string());
        
        result.add_metric(crate::exercise::Metric {
            operation: "tensor_shape_exercise".to_string(),
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
    fn test_tensor_creation_exercise() {
        let exercise = TensorCreationExercise;
        let device = Device::Cpu;
        
        let result = exercise.run(&device);
        assert!(result.is_ok());
        
        let exercise_result = result.unwrap();
        assert!(exercise_result.success);
        assert!(!exercise_result.output.is_empty());
        assert!(!exercise_result.tensors.is_empty());
        assert!(!exercise_result.educational_notes.is_empty());
    }
    
    #[test]
    fn test_tensor_from_data_exercise() {
        let exercise = TensorFromDataExercise;
        let device = Device::Cpu;
        
        let result = exercise.run(&device);
        assert!(result.is_ok());
        
        let exercise_result = result.unwrap();
        assert!(exercise_result.success);
        assert!(!exercise_result.output.is_empty());
        assert!(!exercise_result.tensors.is_empty());
    }
    
    #[test]
    fn test_tensor_shape_exercise() {
        let exercise = TensorShapeExercise;
        let device = Device::Cpu;
        
        let result = exercise.run(&device);
        if let Err(e) = &result {
            println!("Error running TensorShapeExercise: {}", e);
        }
        assert!(result.is_ok());
        
        let exercise_result = result.unwrap();
        assert!(exercise_result.success);
        assert!(!exercise_result.output.is_empty());
        assert!(!exercise_result.tensors.is_empty());
    }
}