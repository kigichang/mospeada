use candle_core::{Error as E, Result};
use hf_hub::{
    Repo, RepoType,
    api::sync::{ApiBuilder, ApiRepo as HFApiRepo},
};
use std::{fs::File, ops::Deref, path::PathBuf};

pub struct ApiRepo {
    model_id: String,
    repo: HFApiRepo,
}

impl Deref for ApiRepo {
    type Target = HFApiRepo;

    fn deref(&self) -> &Self::Target {
        &self.repo
    }
}

impl ApiRepo {
    pub fn from_pretrained(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<&str>,
        token: Option<&str>,
    ) -> Result<Self> {
        let api = match cache_dir {
            Some(cache_dir) => ApiBuilder::new().with_cache_dir(cache_dir.into()),
            None => ApiBuilder::new(),
        };

        let api = api.with_token(token.map(str::to_string));

        let api = api.build().map_err(E::wrap)?;

        let repo = Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        );

        Ok(Self {
            model_id: model_id.to_string(),
            repo: api.repo(repo),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn download_safetensors(&self, json_file: &str) -> Result<Vec<PathBuf>> {
        let json_file = self.get(json_file).map_err(E::wrap)?;
        let safetensors_files = super::read_safetensors_index_file(json_file)?;
        let safetensors_files = safetensors_files
            .iter()
            .map(|v| self.get(v).map_err(E::wrap))
            .collect::<Result<Vec<_>>>()?;
        Ok(safetensors_files)
    }

    pub fn tokenizer_config(&self) -> Result<PathBuf> {
        self.get("tokenizer_config.json").map_err(E::wrap)
    }

    pub fn tokenizer(&self) -> Result<PathBuf> {
        self.get("tokenizer.json").map_err(E::wrap)
    }

    pub fn config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let config = self.get("config.json").map_err(E::wrap)?;
        let config = File::open(config).map_err(E::wrap)?;

        serde_json::from_reader(config).map_err(E::wrap)
    }

    pub fn safetensors(&self) -> Result<Vec<PathBuf>> {
        if let Ok(single_file) = self.get("model.safetensors") {
            Ok(vec![single_file])
        } else {
            self.download_safetensors("model.safetensors.index.json")
        }
    }

    pub fn generation_config(&self) -> Result<PathBuf> {
        self.get("generation_config.json").map_err(E::wrap)
    }
}
