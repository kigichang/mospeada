use crate::Result;
use candle_core::Device;
use candle_core::utils;

pub(crate) fn device(cpu: bool, index: usize) -> Result<Device> {
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

pub fn metal(index: usize) -> Result<Device> {
    device(false, index)
}

pub fn cuda(index: usize) -> Result<Device> {
    device(false, index)
}

pub fn normalize(t: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let length = t.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    t.broadcast_div(&length)
}
