#[cfg(feature = "http")]
pub mod hf_hub;

#[cfg(feature = "chat-template")]
pub mod chat_template;

#[cfg(feature = "debug")]
pub mod debug;

pub mod error;
use candle_core::Tensor;
pub use error::{Error, Result};

pub mod generation;
pub mod pipeline;
pub mod pooling;
pub mod repo;

pub mod utils;
pub use utils::*;

pub trait Module {
    fn forward(&mut self, x: &Tensor, start_pos: usize) -> candle_core::Result<Tensor>;
    fn reset(&mut self);
}
