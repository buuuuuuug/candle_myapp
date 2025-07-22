use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Information about a tensor for display purposes
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub sample_values: String,
}

impl TensorInfo {
    pub fn from_tensor<T: candle_core::WithDType>(name: &str, tensor: &Tensor) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            shape: tensor.shape().dims().to_vec(),
            dtype: T::DTYPE.as_str().to_string(),
            sample_values: format!("{:?}", tensor),
        })
    }
}

/// Performance metric for an operation
#[derive(Debug, Clone)]
pub struct Metric {
    pub operation: String,
    pub duration: Duration,
    pub backend: String,
    pub additional_info: HashMap<String, String>,
}

/// Result of running an exercise
#[derive(Debug)]
pub struct ExerciseResult {
    pub success: bool,
    pub output: String,
    pub tensors: Vec<TensorInfo>,
    pub metrics: Vec<Metric>,
    pub educational_notes: Vec<String>,
}

impl ExerciseResult {
    pub fn new() -> Self {
        Self {
            success: true,
            output: String::new(),
            tensors: Vec::new(),
            metrics: Vec::new(),
            educational_notes: Vec::new(),
        }
    }

    pub fn add_output(&mut self, text: &str) {
        if !self.output.is_empty() {
            self.output.push('\n');
        }
        self.output.push_str(text);
    }

    pub fn add_tensor(&mut self, tensor_info: TensorInfo) {
        self.tensors.push(tensor_info);
    }

    pub fn add_metric(&mut self, metric: Metric) {
        self.metrics.push(metric);
    }

    pub fn add_educational_note(&mut self, note: &str) {
        self.educational_notes.push(note.to_string());
    }

    pub fn set_error(&mut self, error: &str) {
        self.success = false;
        self.add_output(&format!("❌ Error: {}", error));
    }

    pub fn display(&self) {
        println!("📊 Exercise Results:");
        println!("   Status: {}", if self.success { "✅ Success" } else { "❌ Failed" });
        
        if !self.output.is_empty() {
            println!("   Output:");
            for line in self.output.lines() {
                println!("     {}", line);
            }
        }

        if !self.tensors.is_empty() {
            println!("   Tensors:");
            for tensor in &self.tensors {
                println!("     {} ({}): {:?} - {}", 
                    tensor.name, tensor.dtype, tensor.shape, tensor.sample_values);
            }
        }

        if !self.metrics.is_empty() {
            println!("   Performance:");
            for metric in &self.metrics {
                println!("     {}: {:.2}ms ({})", 
                    metric.operation, metric.duration.as_millis(), metric.backend);
            }
        }

        if !self.educational_notes.is_empty() {
            println!("   📚 Educational Notes:");
            for note in &self.educational_notes {
                println!("     • {}", note);
            }
        }
        println!();
    }
}

/// Trait that all exercises must implement
pub trait Exercise {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn run(&self, device: &Device) -> Result<ExerciseResult>;
}

/// A category of related exercises
pub struct ExerciseCategory {
    pub name: String,
    pub description: String,
    pub exercises: Vec<Box<dyn Exercise>>,
}

impl ExerciseCategory {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            exercises: Vec::new(),
        }
    }

    pub fn add_exercise(&mut self, exercise: Box<dyn Exercise>) {
        self.exercises.push(exercise);
    }

    pub fn list_exercises(&self) -> Vec<&str> {
        self.exercises.iter().map(|e| e.name()).collect()
    }

    pub fn get_exercise(&self, name: &str) -> Option<&Box<dyn Exercise>> {
        self.exercises.iter().find(|e| e.name() == name)
    }
}

/// Main framework for managing and running exercises
pub struct ExerciseFramework {
    categories: Vec<ExerciseCategory>,
}

impl ExerciseFramework {
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
        }
    }

    pub fn add_category(&mut self, category: ExerciseCategory) {
        self.categories.push(category);
    }

    pub fn list_categories(&self) -> Vec<&str> {
        self.categories.iter().map(|c| c.name.as_str()).collect()
    }

    pub fn get_category(&self, name: &str) -> Option<&ExerciseCategory> {
        self.categories.iter().find(|c| c.name == name)
    }

    pub fn list_exercises(&self, category_name: &str) -> Result<Vec<&str>> {
        match self.get_category(category_name) {
            Some(category) => Ok(category.list_exercises()),
            None => Err(anyhow::anyhow!("Category '{}' not found", category_name)),
        }
    }

    pub fn run_exercise(&self, category_name: &str, exercise_name: &str, device: &Device) -> Result<ExerciseResult> {
        let category = self.get_category(category_name)
            .ok_or_else(|| anyhow::anyhow!("Category '{}' not found", category_name))?;
        
        let exercise = category.get_exercise(exercise_name)
            .ok_or_else(|| anyhow::anyhow!("Exercise '{}' not found in category '{}'", exercise_name, category_name))?;

        println!("🏃 Running exercise: {} - {}", exercise.name(), exercise.description());
        
        let start_time = Instant::now();
        let mut result = exercise.run(device)?;
        let duration = start_time.elapsed();

        // Add timing metric
        let mut additional_info = HashMap::new();
        additional_info.insert("category".to_string(), category_name.to_string());
        
        result.add_metric(Metric {
            operation: format!("Exercise: {}", exercise.name()),
            duration,
            backend: format!("{:?}", device),
            additional_info,
        });

        Ok(result)
    }

    pub fn display_menu(&self) {
        println!("📚 Available Exercise Categories:");
        for (i, category) in self.categories.iter().enumerate() {
            println!("   {}. {} - {}", i + 1, category.name, category.description);
        }
        println!();
    }

    pub fn display_category_exercises(&self, category_name: &str) -> Result<()> {
        let category = self.get_category(category_name)
            .ok_or_else(|| anyhow::anyhow!("Category '{}' not found", category_name))?;

        println!("📋 Exercises in '{}':", category.name);
        for (i, exercise) in category.exercises.iter().enumerate() {
            println!("   {}. {} - {}", i + 1, exercise.name(), exercise.description());
        }
        println!();
        Ok(())
    }
}

impl Default for ExerciseFramework {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;