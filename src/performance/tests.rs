#[cfg(test)]
mod tests {
    use crate::performance::{PerformanceMonitor, OperationStats};
    use crate::backend::BackendType;
    use crate::exercise::Metric;
    use std::collections::HashMap;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new(BackendType::Cpu);
        assert_eq!(monitor.get_backend_type(), &BackendType::Cpu);
        assert!(monitor.get_metrics().is_empty());
    }

    #[test]
    fn test_timing_operations() {
        let mut monitor = PerformanceMonitor::new(BackendType::Metal);
        
        // Test basic timing
        monitor.start_timing("test_operation");
        thread::sleep(Duration::from_millis(10));
        let duration = monitor.end_timing();
        
        assert!(duration.as_millis() >= 10);
        assert_eq!(monitor.get_metrics().len(), 1);
        
        let metric = &monitor.get_metrics()[0];
        assert_eq!(metric.operation, "test_operation");
        assert_eq!(metric.backend, "Metal");
        assert!(metric.duration.as_millis() >= 10);
    }

    #[test]
    fn test_manual_metric_recording() {
        let mut monitor = PerformanceMonitor::new(BackendType::Cuda);
        
        let mut additional_info = HashMap::new();
        additional_info.insert("tensor_size".to_string(), "1000x1000".to_string());
        
        let metric = Metric {
            operation: "matrix_multiplication".to_string(),
            duration: Duration::from_millis(50),
            backend: "CPU".to_string(), // This should be overridden
            additional_info,
        };
        
        monitor.record_metric(metric);
        
        assert_eq!(monitor.get_metrics().len(), 1);
        let recorded_metric = &monitor.get_metrics()[0];
        assert_eq!(recorded_metric.operation, "matrix_multiplication");
        assert_eq!(recorded_metric.backend, "CUDA"); // Should be overridden
        assert_eq!(recorded_metric.duration.as_millis(), 50);
        assert_eq!(recorded_metric.additional_info.get("tensor_size"), Some(&"1000x1000".to_string()));
    }

    #[test]
    fn test_operation_stats() {
        let mut monitor = PerformanceMonitor::new(BackendType::Cpu);
        
        // Add multiple metrics for the same operation
        for i in 1..=5 {
            let metric = Metric {
                operation: "test_op".to_string(),
                duration: Duration::from_millis(i * 10),
                backend: "CPU".to_string(),
                additional_info: HashMap::new(),
            };
            monitor.record_metric(metric);
        }
        
        let stats = monitor.get_operation_stats("test_op").unwrap();
        assert_eq!(stats.operation, "test_op");
        assert_eq!(stats.count, 5);
        assert_eq!(stats.total_time_ms, 150_000_000.0); // 10+20+30+40+50 (in nanoseconds)
        assert_eq!(stats.avg_time_ms, 30_000_000.0); // 30ms in nanoseconds
        assert_eq!(stats.min_time_ms, 10_000_000.0); // 10ms in nanoseconds
        assert_eq!(stats.max_time_ms, 50_000_000.0); // 50ms in nanoseconds
        assert_eq!(stats.backend, BackendType::Cpu);
        
        // Test non-existent operation
        let no_stats = monitor.get_operation_stats("nonexistent");
        assert!(no_stats.is_none());
    }

    #[test]
    fn test_throughput_calculation() {
        let mut monitor = PerformanceMonitor::new(BackendType::Metal);
        
        // Add metrics: 4 operations taking 100ms each = 4 ops in 0.4 seconds = 10 ops/sec
        for _ in 0..4 {
            let metric = Metric {
                operation: "throughput_test".to_string(),
                duration: Duration::from_millis(100),
                backend: "Metal".to_string(),
                additional_info: HashMap::new(),
            };
            monitor.record_metric(metric);
        }
        
        let throughput = monitor.calculate_throughput("throughput_test").unwrap();
        assert!((throughput - 10.0).abs() < 0.01); // Should be approximately 10 ops/sec
        
        // Test non-existent operation
        let no_throughput = monitor.calculate_throughput("nonexistent");
        assert!(no_throughput.is_none());
    }

    #[test]
    fn test_backend_comparison() {
        let mut monitor = PerformanceMonitor::new(BackendType::Cpu);
        
        // Add metrics for the same operation
        let metric1 = Metric {
            operation: "matrix_mul".to_string(),
            duration: Duration::from_millis(100),
            backend: "CPU".to_string(),
            additional_info: HashMap::new(),
        };
        
        let metric2 = Metric {
            operation: "matrix_mul".to_string(),
            duration: Duration::from_millis(50),
            backend: "CPU".to_string(),
            additional_info: HashMap::new(),
        };
        
        let metric3 = Metric {
            operation: "other_op".to_string(),
            duration: Duration::from_millis(25),
            backend: "CPU".to_string(),
            additional_info: HashMap::new(),
        };
        
        monitor.record_metric(metric1);
        monitor.record_metric(metric2);
        monitor.record_metric(metric3);
        
        let comparison = monitor.compare_backends("matrix_mul");
        assert_eq!(comparison.len(), 2);
        
        let other_comparison = monitor.compare_backends("other_op");
        assert_eq!(other_comparison.len(), 1);
        
        let empty_comparison = monitor.compare_backends("nonexistent");
        assert_eq!(empty_comparison.len(), 0);
    }

    #[test]
    fn test_clear_metrics() {
        let mut monitor = PerformanceMonitor::new(BackendType::Metal);
        
        // Add some metrics
        monitor.start_timing("test");
        thread::sleep(Duration::from_millis(1));
        monitor.end_timing();
        
        assert_eq!(monitor.get_metrics().len(), 1);
        
        monitor.clear_metrics();
        assert_eq!(monitor.get_metrics().len(), 0);
    }

    #[test]
    fn test_end_timing_without_start() {
        let mut monitor = PerformanceMonitor::new(BackendType::Cpu);
        
        // This should not panic and should return 0 duration
        let duration = monitor.end_timing();
        assert_eq!(duration.as_millis(), 0);
        assert_eq!(monitor.get_metrics().len(), 0);
    }

    #[test]
    fn test_operation_stats_display() {
        let stats = OperationStats {
            operation: "test_operation".to_string(),
            count: 5,
            total_time_ms: 100.0,
            avg_time_ms: 20.0,
            min_time_ms: 10.0,
            max_time_ms: 30.0,
            backend: BackendType::Metal,
        };
        
        // This test just ensures the display method doesn't panic
        stats.display();
    }

    #[test]
    fn test_time_operation_macro() {
        let mut monitor = PerformanceMonitor::new(BackendType::Cpu);
        
        let (result, duration) = crate::time_operation!(monitor, "macro_test", {
            thread::sleep(Duration::from_millis(5));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 5);
        assert_eq!(monitor.get_metrics().len(), 1);
        assert_eq!(monitor.get_metrics()[0].operation, "macro_test");
    }
}