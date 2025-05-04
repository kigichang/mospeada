use std::{fs::File, path::Path};

use candle_core::{DType, Device, Error as E, Result, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum EOS {
    Single(u32),
    Multi(Vec<u32>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerationConfig {
    pub eos_token_id: Option<EOS>,
    pub temperature: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_new_tokens: Option<usize>,
}

impl GenerationConfig {
    #[cfg(feature = "hf_hub")]
    pub fn from_pretrained(repo: &crate::hf_hub::ApiRepo) -> Result<Self> {
        let generation_config = repo.generation_config()?;
        Self::from_file(generation_config)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| E::Msg(format!("Failed to open file: {}", e)))?;
        serde_json::from_reader(file).map_err(|e| E::Msg(format!("failed to parse JSON: {}", e)))
    }

    pub fn set_eos_token_id(&mut self, eos_token_id: EOS) {
        self.eos_token_id = Some(eos_token_id);
    }

    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = Some(temperature);
    }

    pub fn set_repetition_penalty(&mut self, repetition_penalty: f32) {
        self.repetition_penalty = Some(repetition_penalty);
    }

    pub fn set_top_p(&mut self, top_p: f64) {
        self.top_p = Some(top_p);
    }

    pub fn set_top_k(&mut self, top_k: usize) {
        self.top_k = Some(top_k);
    }

    pub fn get_eos_token_id(&self) -> Option<Vec<u32>> {
        match &self.eos_token_id {
            Some(EOS::Single(id)) => Some(vec![*id]),
            Some(EOS::Multi(ids)) => Some(ids.clone()),
            None => None,
        }
    }

    pub fn get_repetition_penalty_or(&self, default: f32) -> f32 {
        self.repetition_penalty.unwrap_or(default)
    }

    pub fn get_max_new_tokens_or(&self, default: usize) -> usize {
        self.max_new_tokens.unwrap_or(default)
    }

    pub fn sampling(&self) -> Sampling {
        let temperature = self
            .temperature
            .and_then(|v| if v < 1e-7 { None } else { Some(v) });

        match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match (self.top_k, self.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            },
        }
    }

    pub fn logits_processor(&self, seed: u64) -> LogitsProcessor {
        let sampling = self.sampling();
        LogitsProcessor::from_sampling(seed, sampling)
    }
}

pub trait Model {
    fn forward(&mut self, x: &Tensor, start_pos: usize) -> Result<Tensor>;
    fn reset(&mut self);
}

pub struct TextGeneration<M: Model> {
    model: M,
    device: Device,
    logits_processor: LogitsProcessor,
    repetition_penalty: f32,
    repeat_last_n: usize,
    eos_token_id: Vec<u32>,
}

impl<M: Model> TextGeneration<M> {
    pub fn new(
        model: M,
        device: Device,
        config: &GenerationConfig,
        seed: u64,
        repeat_last_n: usize,
    ) -> Self {
        Self {
            model,
            device,
            logits_processor: config.logits_processor(seed),
            repetition_penalty: config.get_repetition_penalty_or(1.),
            repeat_last_n,
            eos_token_id: config.get_eos_token_id().unwrap(),
        }
    }

    pub fn run<F>(&mut self, ids: Vec<u32>, max_new_tokens: usize, mut cb: F) -> Result<usize>
    where
        F: FnMut(u32),
    {
        self.model.reset();
        let mut tokens = ids;

        let mut generated_tokens = 0usize;
        for index in 0..max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repetition_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repetition_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            cb(next_token);
            generated_tokens += 1;
            if self.eos_token_id.contains(&next_token) {
                break;
            }
        }

        Ok(generated_tokens)
    }
}
