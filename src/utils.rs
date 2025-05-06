use crate::Result;
use candle_core::Device;
use candle_core::utils;

pub fn device(cpu: bool, index: usize) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if utils::cuda_is_available() {
        Ok(Device::new_cuda(index)?)
    } else if utils::metal_is_available() {
        Ok(Device::new_metal(index)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn cpu() -> Result<Device> {
    device(true, 0)
}

pub fn gpu(index: usize) -> Result<Device> {
    device(false, index)
}
