use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::Cpu;
    // let tensor = Tensor::new(&[1.0, 2.0, 3.0], &device)?;
    let vec = vec![1u32, 2, 3, 4, 5, 6];
    // let tensor = Tensor::new(vec, &device)?;
    let tensor = Tensor::from_vec(vec, (2, 3), &device)?;
    println!("{:?}", tensor);
    println!("{:?}", tensor.to_vec2::<u32>());
    Ok(())
}