#[cfg(test)]
mod tests {
    use crate::exercise::{Exercise, ExerciseResult, ExerciseCategory, ExerciseFramework, TensorInfo, Metric};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    use std::time::Duration;
    use anyhow::Result;

    // Mock exercise for testing
    struct MockExercise {
        name: String,
        description: String,
        should_fail: bool,
    }

    impl MockExercise {
        fn new(name: &str, description: &str, should_fail: bool) -> Self {
            Self {
                name: name.to_string(),
                description: description.to_string(),
                should_fail,
            }
        }
    }

    impl Exercise for MockExercise {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn run(&self, device: &Device) -> Result<ExerciseResult> {
            let mut result = ExerciseResult::new();
            
            if self.should_fail {
                result.set_error("Mock exercise failure");
                return Ok(result);
            }

            result.add_output("Mock exercise completed successfully");
            
            // Create a test tensor
            let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), device)?;
            let tensor_info = TensorInfo::from_tensor::<f32>("test_tensor", &tensor)?;
            result.add_tensor(tensor_info);
            
            result.add_educational_note("This is a mock exercise for testing");
            
            Ok(result)
        }
    }

    #[test]
    fn test_exercise_result_creation() {
        let mut result = ExerciseResult::new();
        assert!(result.success);
        assert!(result.output.is_empty());
        assert!(result.tensors.is_empty());
        assert!(result.metrics.is_empty());
        assert!(result.educational_notes.is_empty());

        result.add_output("Test output");
        assert_eq!(result.output, "Test output");

        result.add_output("More output");
        assert_eq!(result.output, "Test output\nMore output");
    }

    #[test]
    fn test_exercise_result_error() {
        let mut result = ExerciseResult::new();
        result.set_error("Test error");
        
        assert!(!result.success);
        assert!(result.output.contains("❌ Error: Test error"));
    }

    #[test]
    fn test_exercise_category() {
        let mut category = ExerciseCategory::new("Test Category", "A test category");
        assert_eq!(category.name, "Test Category");
        assert_eq!(category.description, "A test category");
        assert!(category.exercises.is_empty());

        let exercise = Box::new(MockExercise::new("Test Exercise", "A test exercise", false));
        category.add_exercise(exercise);
        
        assert_eq!(category.exercises.len(), 1);
        assert_eq!(category.list_exercises(), vec!["Test Exercise"]);
        
        let found_exercise = category.get_exercise("Test Exercise");
        assert!(found_exercise.is_some());
        assert_eq!(found_exercise.unwrap().name(), "Test Exercise");
        
        let not_found = category.get_exercise("Nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_exercise_framework() {
        let mut framework = ExerciseFramework::new();
        assert!(framework.categories.is_empty());

        let mut category = ExerciseCategory::new("Test Category", "A test category");
        category.add_exercise(Box::new(MockExercise::new("Exercise 1", "First exercise", false)));
        category.add_exercise(Box::new(MockExercise::new("Exercise 2", "Second exercise", false)));
        
        framework.add_category(category);
        
        assert_eq!(framework.list_categories(), vec!["Test Category"]);
        
        let exercises = framework.list_exercises("Test Category").unwrap();
        assert_eq!(exercises, vec!["Exercise 1", "Exercise 2"]);
        
        let invalid_category = framework.list_exercises("Invalid Category");
        assert!(invalid_category.is_err());
    }

    #[test]
    fn test_exercise_execution() {
        let mut framework = ExerciseFramework::new();
        let mut category = ExerciseCategory::new("Test Category", "A test category");
        category.add_exercise(Box::new(MockExercise::new("Success Exercise", "Should succeed", false)));
        category.add_exercise(Box::new(MockExercise::new("Fail Exercise", "Should fail", true)));
        framework.add_category(category);

        let device = Device::Cpu;
        
        // Test successful exercise
        let result = framework.run_exercise("Test Category", "Success Exercise", &device);
        assert!(result.is_ok());
        let exercise_result = result.unwrap();
        assert!(exercise_result.success);
        assert!(exercise_result.output.contains("Mock exercise completed successfully"));
        assert_eq!(exercise_result.tensors.len(), 1);
        assert_eq!(exercise_result.educational_notes.len(), 1);
        assert_eq!(exercise_result.metrics.len(), 1); // Should have timing metric

        // Test failing exercise
        let result = framework.run_exercise("Test Category", "Fail Exercise", &device);
        assert!(result.is_ok());
        let exercise_result = result.unwrap();
        assert!(!exercise_result.success);
        assert!(exercise_result.output.contains("❌ Error: Mock exercise failure"));

        // Test invalid category
        let result = framework.run_exercise("Invalid Category", "Exercise", &device);
        assert!(result.is_err());

        // Test invalid exercise
        let result = framework.run_exercise("Test Category", "Invalid Exercise", &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_info() {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let tensor_info = TensorInfo::from_tensor::<f32>("test", &tensor).unwrap();
        
        assert_eq!(tensor_info.name, "test");
        assert_eq!(tensor_info.shape, vec![2, 2]);
        assert_eq!(tensor_info.dtype, "f32");
        assert!(!tensor_info.sample_values.is_empty());
    }

    #[test]
    fn test_metric_creation() {
        let mut additional_info = HashMap::new();
        additional_info.insert("test_key".to_string(), "test_value".to_string());
        
        let metric = Metric {
            operation: "Test Operation".to_string(),
            duration: Duration::from_millis(100),
            backend: "CPU".to_string(),
            additional_info,
        };
        
        assert_eq!(metric.operation, "Test Operation");
        assert_eq!(metric.duration.as_millis(), 100);
        assert_eq!(metric.backend, "CPU");
        assert_eq!(metric.additional_info.get("test_key"), Some(&"test_value".to_string()));
    }
}