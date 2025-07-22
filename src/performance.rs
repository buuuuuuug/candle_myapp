use crate::backend::BackendType;
use crate::exercise::Metric;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance monitor for collecting and analyzing metrics
pub struct PerformanceMonitor {
    start_time: Option<Instant>,
    current_operation: Option<String>,
    metrics: Vec<Metric>,
    backend_type: BackendType,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(backend_type: BackendType) -> Self {
        Self {
            start_time: None,
            current_operation: None,
            metrics: Vec::new(),
            backend_type,
        }
    }

    /// Start timing an operation
    pub fn start_timing(&mut self, operation: &str) {
        self.start_time = Some(Instant::now());
        self.current_operation = Some(operation.to_string());
    }

    /// End timing and return the duration
    pub fn end_timing(&mut self) -> Duration {
        match (self.start_time.take(), self.current_operation.take()) {
            (Some(start), Some(operation)) => {
                let duration = start.elapsed();
                
                // Record the metric
                let metric = Metric {
                    operation,
                    duration,
                    backend: self.backend_type.to_string(),
                    additional_info: HashMap::new(),
                };
                
                self.metrics.push(metric);
                duration
            }
            _ => {
                eprintln!("Warning: end_timing called without start_timing");
                Duration::from_millis(0)
            }
        }
    }

    /// Record a metric manually
    pub fn record_metric(&mut self, mut metric: Metric) {
        // Ensure backend is set correctly
        metric.backend = self.backend_type.to_string();
        self.metrics.push(metric);
    }

    /// Get all recorded metrics
    pub fn get_metrics(&self) -> &[Metric] {
        &self.metrics
    }

    /// Clear all metrics
    pub fn clear_metrics(&mut self) {
        self.metrics.clear();
    }

    /// Display all metrics in a formatted way
    pub fn display_metrics(&self) {
        if self.metrics.is_empty() {
            println!("📊 No performance metrics recorded yet.");
            return;
        }

        println!("📊 Performance Metrics Summary:");
        println!("   Backend: {}", self.backend_type);
        println!("   Total Operations: {}", self.metrics.len());
        println!();

        // Group metrics by operation type
        let mut operation_groups: HashMap<String, Vec<&Metric>> = HashMap::new();
        for metric in &self.metrics {
            operation_groups
                .entry(metric.operation.clone())
                .or_insert_with(Vec::new)
                .push(metric);
        }

        // Display grouped metrics
        for (operation, metrics) in operation_groups {
            let durations: Vec<f64> = metrics.iter()
                .map(|m| m.duration.as_millis() as f64)
                .collect();
            
            let total_time: f64 = durations.iter().sum();
            let avg_time = total_time / durations.len() as f64;
            let min_time = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_time = durations.iter().fold(0.0f64, |a, &b| a.max(b));

            println!("   🔧 {}:", operation);
            println!("      Executions: {}", metrics.len());
            println!("      Average: {:.2}ms", avg_time);
            println!("      Min: {:.2}ms", min_time);
            println!("      Max: {:.2}ms", max_time);
            println!("      Total: {:.2}ms", total_time);
            
            // Show additional info if available
            if let Some(first_metric) = metrics.first() {
                if !first_metric.additional_info.is_empty() {
                    println!("      Additional Info:");
                    for (key, value) in &first_metric.additional_info {
                        println!("        {}: {}", key, value);
                    }
                }
            }
            println!();
        }
    }

    /// Compare performance across different backends for the same operation
    pub fn compare_backends(&self, operation: &str) -> Vec<&Metric> {
        self.metrics
            .iter()
            .filter(|m| m.operation == operation)
            .collect()
    }

    /// Get performance statistics for a specific operation
    pub fn get_operation_stats(&self, operation: &str) -> Option<OperationStats> {
        let operation_metrics: Vec<&Metric> = self.metrics
            .iter()
            .filter(|m| m.operation == operation)
            .collect();

        if operation_metrics.is_empty() {
            return None;
        }

        let durations: Vec<f64> = operation_metrics
            .iter()
            .map(|m| m.duration.as_millis() as f64)
            .collect();

        let total_time: f64 = durations.iter().sum();
        let count = durations.len();
        let avg_time = total_time / count as f64;
        let min_time = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time = durations.iter().fold(0.0f64, |a, &b| a.max(b));

        Some(OperationStats {
            operation: operation.to_string(),
            count,
            total_time_ms: total_time,
            avg_time_ms: avg_time,
            min_time_ms: min_time,
            max_time_ms: max_time,
            backend: self.backend_type.clone(),
        })
    }

    /// Calculate throughput (operations per second) for a given operation
    pub fn calculate_throughput(&self, operation: &str) -> Option<f64> {
        let stats = self.get_operation_stats(operation)?;
        if stats.total_time_ms > 0.0 {
            Some((stats.count as f64) / (stats.total_time_ms / 1000.0))
        } else {
            None
        }
    }

    /// Display performance comparison between operations
    pub fn display_comparison(&self, operations: &[&str]) {
        println!("📈 Performance Comparison:");
        println!("   Backend: {}", self.backend_type);
        println!();

        for operation in operations {
            if let Some(stats) = self.get_operation_stats(operation) {
                println!("   🔧 {}:", operation);
                println!("      Average: {:.2}ms", stats.avg_time_ms);
                println!("      Executions: {}", stats.count);
                
                if let Some(throughput) = self.calculate_throughput(operation) {
                    println!("      Throughput: {:.2} ops/sec", throughput);
                }
                println!();
            } else {
                println!("   🔧 {}: No data available", operation);
                println!();
            }
        }
    }

    /// Get the current backend type
    pub fn get_backend_type(&self) -> &BackendType {
        &self.backend_type
    }
}

/// Statistics for a specific operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub operation: String,
    pub count: usize,
    pub total_time_ms: f64,
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub backend: BackendType,
}

impl OperationStats {
    pub fn display(&self) {
        println!("📊 Statistics for '{}':", self.operation);
        println!("   Backend: {}", self.backend);
        println!("   Executions: {}", self.count);
        println!("   Average Time: {:.2}ms", self.avg_time_ms);
        println!("   Min Time: {:.2}ms", self.min_time_ms);
        println!("   Max Time: {:.2}ms", self.max_time_ms);
        println!("   Total Time: {:.2}ms", self.total_time_ms);
        println!();
    }
}

/// Utility macro for timing operations
#[macro_export]
macro_rules! time_operation {
    ($monitor:expr, $operation:expr, $code:block) => {{
        $monitor.start_timing($operation);
        let result = $code;
        let duration = $monitor.end_timing();
        (result, duration)
    }};
}

#[cfg(test)]
mod tests;