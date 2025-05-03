use candle_core::{Error as E, Result};
use minijinja::Environment;
use serde;
use std::{path::Path, sync::Arc};
use tokenizers::Tokenizer as HFTokenizer;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokenizer: Arc<HFTokenizer>,
    template: Option<String>,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl Tokenizer {
    pub fn tokenizer(&self) -> &HFTokenizer {
        &self.tokenizer
    }
    pub fn apply_chat_template<S: serde::Serialize>(
        &self,
        prompt: S,
        env: Environment,
    ) -> Result<String> {
        match self.template {
            Some(ref template) => env.render_str(template.as_str(), prompt).map_err(E::wrap),
            None => serde_json::to_string(&prompt).map_err(E::wrap),
        }
    }

    #[cfg(feature = "hf_hub")]
    pub fn from_pretrained(repo: super::hf_hub::ApiRepo) -> Result<Tokenizer> {
        use std::fs::File;

        let tokenizer_config = repo.tokenizer_config()?;
        let tokenizer_config: serde_json::Value =
            serde_json::from_reader(File::open(tokenizer_config)?).map_err(E::wrap)?;

        let chat_template = tokenizer_config
            .get("chat_template")
            .and_then(|v| v.as_str().map(str::to_string));

        from_file(repo.tokenizer()?, chat_template)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
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
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

pub fn from_file<'t, P: AsRef<Path>>(p: P, template: Option<String>) -> Result<Tokenizer> {
    let tokenizer = HFTokenizer::from_file(p).map_err(E::wrap)?;
    Ok(Tokenizer {
        tokenizer: Arc::new(tokenizer),
        template,
        tokens: Vec::new(),
        prev_index: 0,
        current_index: 0,
    })
}
