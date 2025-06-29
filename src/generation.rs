use crate::Module;
use crate::{Result, repo::Repo};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use serde::{Deserialize, Serialize};
use std::{fs::File, path::Path};
use tokenizers::Tokenizer;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Eos {
    Single(u32),
    Multi(Vec<u32>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerationConfig {
    pub eos_token_id: Option<Eos>,
    pub temperature: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_new_tokens: Option<usize>,
}

impl GenerationConfig {
    pub fn from_pretrained<R: Repo>(repo: &R) -> Result<GenerationConfig> {
        repo.generate_config()
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        Ok(serde_json::from_reader(file)?)
    }

    pub fn set_eos_token_id(&mut self, eos_token_id: Eos) {
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
            Some(Eos::Single(id)) => Some(vec![*id]),
            Some(Eos::Multi(ids)) => Some(ids.clone()),
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

pub struct TextGeneration<M: Module> {
    model: M,
    device: Device,
    logits_processor: LogitsProcessor,
    repetition_penalty: f32,
    repeat_last_n: usize,
    eos_token_id: Vec<u32>,

    max_new_tokens: usize,
    generated_tokens: usize,
    tokens: Vec<u32>,
}

impl<M: Module> TextGeneration<M> {
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
            max_new_tokens: config.get_max_new_tokens_or(0),
            generated_tokens: 0,
            tokens: Vec::new(),
        }
    }

    pub fn apply(&mut self, ids: &[u32], max_new_tokens: usize) -> Result<u32> {
        self.model.reset();
        self.tokens = ids.to_vec();
        self.generated_tokens = 0;
        self.max_new_tokens = max_new_tokens;
        self.next_token(self.tokens.len())
    }

    pub fn next(&mut self) -> Result<u32> {
        self.next_token(1)
    }

    pub(crate) fn next_token(&mut self, context_size: usize) -> Result<u32> {
        if self.generated_tokens >= self.max_new_tokens {
            return Err(crate::Error::MaxNewTokenExceeded {
                max_new_tokens: self.max_new_tokens,
            });
        }

        let start_pos = self.tokens.len().saturating_sub(context_size);
        let ctxt = &self.tokens[start_pos..];
        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if self.repetition_penalty == 1. {
            logits
        } else {
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repetition_penalty,
                &self.tokens[start_at..],
            )?
        };

        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);
        self.generated_tokens += 1;
        if self.eos_token_id.contains(&next_token) {
            Err(crate::Error::Eos {
                eos_token_id: next_token,
                generated: self.generated_tokens,
            })
        } else {
            Ok(next_token)
        }
    }

    // pub fn run<F>(&mut self, ids: Vec<u32>, max_new_tokens: usize, mut cb: F) -> Result<usize>
    // where
    //     F: FnMut(u32),
    // {
    //     match self.apply(ids, max_new_tokens) {
    //         Ok(next_token) => cb(next_token),
    //         Err(crate::Error::Eos {
    //             eos_token_id,
    //             generated,
    //         }) => {
    //             cb(eos_token_id);
    //             return Ok(generated);
    //         }
    //         Err(crate::Error::MaxNewTokenExceeded { max_new_tokens }) => {
    //             return Ok(max_new_tokens);
    //         }
    //         Err(e) => return Err(e),
    //     }

    //     loop {
    //         match self.next_token(1) {
    //             Ok(next_token) => cb(next_token),
    //             Err(crate::Error::Eos {
    //                 eos_token_id,
    //                 generated,
    //             }) => {
    //                 cb(eos_token_id);
    //                 return Ok(generated);
    //             }
    //             Err(crate::Error::MaxNewTokenExceeded { max_new_tokens }) => {
    //                 return Ok(max_new_tokens);
    //             }
    //             Err(e) => return Err(e),
    //         }
    //     }
    // }

    // pub fn run<F>(&mut self, ids: Vec<u32>, max_new_tokens: usize, mut cb: F) -> Result<usize>
    // where
    //     F: FnMut(u32),
    // {
    //     self.model.reset();
    //     let mut tokens = ids;

    //     let mut generated_tokens = 0usize;
    //     for index in 0..max_new_tokens {
    //         let context_size = if index > 0 { 1 } else { tokens.len() };
    //         let start_pos = tokens.len().saturating_sub(context_size);
    //         let ctxt = &tokens[start_pos..];
    //         let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
    //         let logits = self.model.forward(&input, start_pos)?;
    //         let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    //         let logits = if self.repetition_penalty == 1. {
    //             logits
    //         } else {
    //             let start_at = tokens.len().saturating_sub(self.repeat_last_n);
    //             candle_transformers::utils::apply_repeat_penalty(
    //                 &logits,
    //                 self.repetition_penalty,
    //                 &tokens[start_at..],
    //             )?
    //         };

    //         let next_token = self.logits_processor.sample(&logits)?;
    //         tokens.push(next_token);
    //         cb(next_token);
    //         generated_tokens += 1;
    //         if self.eos_token_id.contains(&next_token) {
    //             break;
    //         }
    //     }

    //     Ok(generated_tokens)
    // }
}

pub struct TextOutputStream<T: AsRef<Tokenizer>> {
    tokenizer: T,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<T: AsRef<Tokenizer>> TextOutputStream<T> {
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer.as_ref()
    }

    pub fn into_inner(self) -> T {
        self.tokenizer
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.tokenizer().decode(tokens, true)?)
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer().get_vocab(true).get(token_s).copied()
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
