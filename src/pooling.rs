use candle_core::{DType, Result, Tensor};

pub fn mean(output: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let attention_mask = attention_mask.unsqueeze(candle_core::D::Minus1)?;
    let input_mask_expanded = attention_mask
        .expand(output.shape())?
        .to_dtype(DType::F32)?;
    let sum = output.broadcast_mul(&input_mask_expanded)?.sum(1)?;
    let mask = input_mask_expanded.sum(1)?;
    let mask = mask.clamp(1e-9, f32::INFINITY)?;
    sum / mask
}
