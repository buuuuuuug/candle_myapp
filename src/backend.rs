use anyhow::Result;
use candle_core::Device;
use candle_core::backend::BackendDevice;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    Cuda,
    Metal,
    Cpu,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::Cpu => write!(f, "CPU"),
        }
    }
}

pub struct BackendManager {
    current_device: Device,
    backend_type: BackendType,
}

impl BackendManager {
    /// Create a new BackendManager with automatic device detection
    /// Priority order: CUDA > Metal > CPU
    pub fn new() -> Result<Self> {
        // Try CUDA first (if available)
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Self::try_cuda() {
                println!("✅ Using CUDA backend");
                return Ok(Self {
                    current_device: device,
                    backend_type: BackendType::Cuda,
                });
            } else {
                println!("⚠️  CUDA not available, trying next backend...");
            }
        }

        // Try Metal second (if available)
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Self::try_metal() {
                println!("✅ Using Metal backend");
                return Ok(Self {
                    current_device: device,
                    backend_type: BackendType::Metal,
                });
            } else {
                println!("⚠️  Metal not available, falling back to CPU...");
            }
        }

        // Fallback to CPU
        println!("✅ Using CPU backend");
        Ok(Self {
            current_device: Device::Cpu,
            backend_type: BackendType::Cpu,
        })
    }

    /// Try to initialize CUDA backend
    #[cfg(feature = "cuda")]
    fn try_cuda() -> Result<Device> {
        use candle_core::CudaDevice;
        let device = CudaDevice::new(0)?;
        Ok(Device::Cuda(device))
    }

    /// Try to initialize Metal backend
    #[cfg(feature = "metal")]
    fn try_metal() -> Result<Device> {
        use candle_core::MetalDevice;
        let device = MetalDevice::new(0)?;
        Ok(Device::Metal(device))
    }

    /// Get the current device
    pub fn get_device(&self) -> &Device {
        &self.current_device
    }

    /// Get backend type
    pub fn get_backend_type(&self) -> &BackendType {
        &self.backend_type
    }

    /// Get detailed backend information
    pub fn get_backend_info(&self) -> String {
        match &self.backend_type {
            BackendType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    if let Device::Cuda(cuda_device) = &self.current_device {
                        format!("CUDA Device: {}", cuda_device.device_name().unwrap_or("Unknown".to_string()))
                    } else {
                        "CUDA (Unknown Device)".to_string()
                    }
                }
                #[cfg(not(feature = "cuda"))]
                "CUDA (Not Available)".to_string()
            }
            BackendType::Metal => {
                #[cfg(feature = "metal")]
                {
                    "Apple Metal GPU".to_string()
                }
                #[cfg(not(feature = "metal"))]
                "Metal (Not Available)".to_string()
            }
            BackendType::Cpu => {
                format!("CPU ({} threads)", num_cpus::get())
            }
        }
    }

    /// Run a simple benchmark to test backend performance
    pub fn benchmark_backend(&self) -> Result<f64> {
        use candle_core::Tensor;
        use std::time::Instant;

        // Create test tensors
        let size = 1000;
        let a = Tensor::randn(0f32, 1f32, (size, size), &self.current_device)?;
        let b = Tensor::randn(0f32, 1f32, (size, size), &self.current_device)?;

        // Warm up
        let _ = a.matmul(&b)?;

        // Benchmark matrix multiplication
        let start = Instant::now();
        let iterations = 10;
        
        for _ in 0..iterations {
            let _ = a.matmul(&b)?;
        }

        let duration = start.elapsed();
        let avg_time_ns = duration.as_nanos() as f64 / iterations as f64;
        
        Ok(avg_time_ns)
    }

    /// Display backend status and information
    pub fn display_status(&self) {
        println!("🔧 Backend Manager Status:");
        println!("   Backend: {}", self.backend_type);
        println!("   Details: {}", self.get_backend_info());
        
        // Run benchmark
        match self.benchmark_backend() {
            Ok(time_ns) => {
                println!("   Performance: {:.2} ns (1000x1000 matrix multiplication)", time_ns);
            }
            Err(e) => {
                println!("   Performance: Benchmark failed - {}", e);
            }
        }
        println!();
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new().expect("Failed to initialize any backend")
    }
}

#[cfg(test)]
mod tests;