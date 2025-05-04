use crate::{Error as E, Result, bail};
use minijinja::{Environment, Template};
use serde;
use std::{fs::File, path::Path, sync::Arc};
use tokenizers::Tokenizer as HFTokenizer;

#[derive(Debug, Clone)]
pub struct Tokenizer<'t> {
    tokenizer: Arc<HFTokenizer>,
    template: Arc<Option<Template<'t, 't>>>,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'t> Tokenizer<'t> {
    pub fn tokenizer(&self) -> &HFTokenizer {
        &self.tokenizer
    }
    pub fn apply_chat_template<S: serde::Serialize>(&self, prompt: S) -> Result<Vec<u32>> {
        let text = self.template.as_ref().as_ref().map_or_else(
            || serde_json::to_string(&prompt).map_err(E::wrap),
            |t| t.render(&prompt).map_err(E::wrap),
        )?;
        Ok(self.tokenizer.encode(text, true)?.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => bail!("cannot decode: {err}"),
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

fn from_files<'s, P: AsRef<Path>>(
    name: &str,
    tokenizer_config: P,
    tokenizer: P,
    env: &'s mut Environment,
) -> Result<Tokenizer<'s>> {
    let tokenizer = HFTokenizer::from_file(tokenizer)?;
    let tokenizer_config: serde_json::Value =
        serde_json::from_reader(File::open(tokenizer_config)?)?;

    let chat_template = tokenizer_config
        .get("chat_template")
        .and_then(|v| v.as_str().map(str::to_string));

    let template = if let Some(t) = chat_template {
        Some(
            env.add_template_owned(name.to_string(), t.to_string())
                .and_then(|()| env.get_template(name))?,
        )
    } else {
        None
    };

    Ok(Tokenizer {
        tokenizer: Arc::new(tokenizer),
        template: Arc::new(template),
        tokens: Vec::new(),
        prev_index: 0,
        current_index: 0,
    })
}

pub fn load_from_cache_dir<'s, P: AsRef<Path>>(
    cache_dir: P,
    name: &str,
    env: &'s mut Environment,
) -> Result<Tokenizer<'s>> {
    let tokenizer_config = cache_dir.as_ref().join("tokenizer_config.json");
    let tokenizer = cache_dir.as_ref().join("tokenizer.json");

    from_files(name, tokenizer_config, tokenizer, env)
}

#[cfg(feature = "hf_hub")]
pub fn from_pretrained<'s>(
    repo: &super::hf_hub::ApiRepo,
    env: &'s mut Environment,
) -> Result<Tokenizer<'s>> {
    let tokenizer_config = repo.tokenizer_config()?;
    let tokenizer = repo.tokenizer()?;

    from_files(repo.model_id(), tokenizer_config, tokenizer, env)
}
