use crate::{Error as E, Result, repo::Repo};
use hf_hub::{
    Repo as HFRepo, RepoType,
    api::sync::{ApiBuilder, ApiRepo as HFApiRepo},
};
use std::path::PathBuf;

pub struct ApiRepo {
    model_id: String,
    repo: HFApiRepo,
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

        let api = api.build()?;

        let repo = HFRepo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        );

        Ok(Self {
            model_id: model_id.to_string(),
            repo: api.repo(repo),
        })
    }

    pub fn download_safetensors(&self, json_file: &str) -> Result<Vec<PathBuf>> {
        let json_file = self.repo.get(json_file)?;
        let safetensors_files = crate::repo::read_safetensors_index_file(json_file)?;
        let safetensors_files = safetensors_files
            .iter()
            .map(|v| self.repo.get(v).map_err(E::wrap))
            .collect::<Result<Vec<_>>>()?;
        Ok(safetensors_files)
    }
}

impl Repo for ApiRepo {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn tokenizer_config_file(&self) -> Result<PathBuf> {
        Ok(self.repo.get("tokenizer_config.json")?)
    }

    fn tokenizer_file(&self) -> Result<PathBuf> {
        Ok(self.repo.get("tokenizer.json")?)
    }

    fn config_file(&self) -> Result<PathBuf> {
        Ok(self.repo.get("config.json")?)
    }

    fn safetensors_files(&self) -> Result<Vec<PathBuf>> {
        if let Ok(single_file) = self.repo.get("model.safetensors") {
            return Ok(vec![single_file]);
        }
        self.download_safetensors("model.safetensors.index.json")
    }

    fn generate_config_file(&self) -> Result<PathBuf> {
        Ok(self.repo.get("generation_config.json")?)
    }
}
