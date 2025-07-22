#[cfg(test)]
mod tests {
    use crate::backend::{BackendManager, BackendType};
    use candle_core::Tensor;

    #[test]
    fn test_backend_manager_creation() {
        let backend_manager = BackendManager::new();
        assert!(backend_manager.is_ok(), "Backend manager should initialize successfully");
        
        let manager = backend_manager.unwrap();
        let device = manager.get_device();
        
        // Should be able to create tensors on the device
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), device);
        assert!(tensor.is_ok(), "Should be able to create tensor on selected device");
    }

    #[test]
    fn test_backend_info() {
        let backend_manager = BackendManager::new().unwrap();
        let info = backend_manager.get_backend_info();
        
        // Info should not be empty
        assert!(!info.is_empty(), "Backend info should not be empty");
        
        // Should contain expected backend type
        let backend_type = backend_manager.get_backend_type();
        match backend_type {
            BackendType::Cpu => assert!(info.contains("CPU")),
            BackendType::Metal => assert!(info.contains("Metal")),
            BackendType::Cuda => assert!(info.contains("CUDA")),
        }
    }

    #[test]
    fn test_benchmark() {
        let backend_manager = BackendManager::new().unwrap();
        let benchmark_result = backend_manager.benchmark_backend();
        
        assert!(benchmark_result.is_ok(), "Benchmark should complete successfully");
        
        let time_ms = benchmark_result.unwrap();
        assert!(time_ms >= 0.0, "Benchmark time should be non-negative");
        assert!(time_ms < 10000.0, "Benchmark time should be reasonable (< 10s)");
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(format!("{}", BackendType::Cpu), "CPU");
        assert_eq!(format!("{}", BackendType::Metal), "Metal");
        assert_eq!(format!("{}", BackendType::Cuda), "CUDA");
    }

    #[test]
    fn test_device_operations() {
        let backend_manager = BackendManager::new().unwrap();
        let device = backend_manager.get_device();
        
        // Test basic tensor operations
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device).unwrap();
        let b = Tensor::from_vec(vec![2.0f32, 0.0, 1.0, 2.0], (2, 2), device).unwrap();
        
        // Test addition
        let sum = (&a + &b).unwrap();
        let sum_data: Vec<f32> = sum.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(sum_data, vec![3.0, 2.0, 4.0, 6.0]);
        
        // Test matrix multiplication
        let product = a.matmul(&b).unwrap();
        assert_eq!(product.shape().dims(), &[2, 2]);
    }
}