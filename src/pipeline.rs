use crate::Module;
use crate::Result;
use crate::chat_template::ChatTemplate;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use serde::Serialize;
use tokenizers::Tokenizer;

pub struct Pipeline<C: AsRef<ChatTemplate>, M: Module, T: AsRef<Tokenizer>> {
    chat_template: C,
    tokenizer: T,
    module: M,
    device: Device,
    logits_processor: LogitsProcessor,
    repetition_penalty: f32,
    repeat_last_n: usize,
    eos_token_id: Vec<u32>,

    max_new_tokens: usize,
    generated_tokens: usize,
    tokens: Vec<u32>,
    priv_index: usize,
    current_index: usize,
}

impl<C: AsRef<ChatTemplate>, M: Module, T: AsRef<Tokenizer>> Pipeline<C, M, T> {
    pub fn into_inner(self) -> (C, T, M) {
        (self.chat_template, self.tokenizer, self.module)
    }

    pub fn new(
        chat_template: C,
        tokenizer: T,
        module: M,
        device: Device,
        config: &crate::generation::GenerationConfig,
        seed: u64,
        repeat_last_n: usize,
    ) -> Self {
        Self {
            chat_template,
            tokenizer,
            module,
            device,
            logits_processor: config.logits_processor(seed),
            repetition_penalty: config.get_repetition_penalty_or(1.),
            repeat_last_n,
            eos_token_id: config.get_eos_token_id().unwrap(),
            max_new_tokens: config.get_max_new_tokens_or(0),
            generated_tokens: 0,
            tokens: Vec::new(),
            priv_index: 0,
            current_index: 0,
        }
    }

    fn next_token(
        &self,
        tokens: &[u32],
        src_len: usize,
        prev_index: &mut usize,
        current_index: &mut usize,
    ) -> Result<Option<String>> {
        let tokenizer = self.tokenizer.as_ref();
        let prev_text = if src_len == *prev_index {
            String::new()
        } else {
            tokenizer.decode(&tokens[*prev_index..*current_index], true)?
        };

        let text = tokenizer.decode(&tokens[*prev_index..], true)?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            *prev_index = *current_index;
            *current_index = tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    fn decode_rest(
        &self,
        tokens: &[u32],
        src_len: usize,
        prev_index: usize,
        current_index: usize,
    ) -> Result<Option<String>> {
        let tokenizer = self.tokenizer.as_ref();
        let prev_text = if prev_index == src_len {
            String::new()
        } else {
            tokenizer.decode(&tokens[prev_index..current_index], true)?
        };

        let text = tokenizer.decode(&tokens[prev_index..], true)?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn run<S: Into<String>>(&mut self, msg: Msg, max_new_tokens: usize) -> Result<()> {
        let text = match msg {
            Msg::Text(t) => t,
            Msg::Chat(chats) => self.chat_template.as_ref().render(chats)?,
        };

        let mut tokens = self
            .tokenizer
            .as_ref()
            .encode(text, true)?
            .get_ids()
            .to_vec();

        let mut generated_tokens = 0usize;
        let msg_len = tokens.len();
        let mut prev_index = tokens.len();
        let mut current_index = prev_index;

        let start_gen = std::time::Instant::now();
        for index in 0..max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.module.forward(&input, start_pos)?;
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
            generated_tokens += 1;
            if self.eos_token_id.contains(&next_token) {
                break;
            }

            if let Some(t) =
                self.next_token(&tokens, msg_len, &mut prev_index, &mut current_index)?
            {
                print!("{t}");
                // std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.decode_rest(&tokens, msg_len, prev_index, current_index)? {
            print!("{rest}");
        }
        // std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        todo!();
    }
}

#[derive(Serialize, Debug)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

pub enum Msg {
    Text(String),
    Chat(Vec<ChatMsg>),
}
