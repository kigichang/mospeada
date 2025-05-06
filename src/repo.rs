use crate::{Error as E, Result, bail, generation::GenerationConfig};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

/// 代表模型 repo
pub trait Repo {
    /// 模型 ID
    fn model_id(&self) -> &str;

    /// tokenizer_config.json 檔案路徑
    fn tokenizer_config_file(&self) -> Result<PathBuf>;

    /// tokenizer.json 檔案路徑
    fn tokenizer_file(&self) -> Result<PathBuf>;

    /// config.json 檔案路徑
    fn config_file(&self) -> Result<PathBuf>;

    /// 所有 safetensors 檔案路徑
    fn safetensors_files(&self) -> Result<Vec<PathBuf>>;

    /// generate_config.json 檔案路徑
    fn generate_config_file(&self) -> Result<PathBuf>;

    /// 回傳模型設定檔 struct
    fn config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let config_file = self.config_file()?;
        let file = std::fs::File::open(config_file)?;
        let config: T = serde_json::from_reader(file)?;
        Ok(config)
    }

    /// 回傳 GenerationConfig struct
    fn generate_config(&self) -> Result<GenerationConfig> {
        GenerationConfig::from_file(self.generate_config_file()?)
    }
}

/// 本地存放位置
pub struct LocalRepo {
    /// 模型 ID
    model_id: String,

    /// 模型目錄路徑
    path: PathBuf,
}

impl LocalRepo {
    pub fn new<P: AsRef<Path>>(model_id: &str, path: P) -> Self {
        Self {
            model_id: model_id.to_owned(),
            path: path.as_ref().to_owned(),
        }
    }
}

impl Repo for LocalRepo {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn tokenizer_config_file(&self) -> Result<PathBuf> {
        Ok(self.path.join("tokenizer_config.json"))
    }

    fn tokenizer_file(&self) -> Result<PathBuf> {
        Ok(self.path.join("tokenizer.json"))
    }

    fn config_file(&self) -> Result<PathBuf> {
        Ok(self.path.join("config.json"))
    }

    fn safetensors_files(&self) -> Result<Vec<PathBuf>> {
        let single_safatensors_file = self.path.join("model.safetensors");
        if single_safatensors_file.exists() {
            return Ok(vec![single_safatensors_file]);
        }

        let index_file = self.path.join("model.safetensors.index.json");
        load_safetensors(&self.path, &index_file)
    }

    fn generate_config_file(&self) -> Result<PathBuf> {
        Ok(self.path.join("generate_config.json"))
    }
}

/// Reads a safetensors index file and returns a set of safetensors files.
pub(crate) fn read_safetensors_index_file<P: AsRef<Path>>(json_file: P) -> Result<HashSet<String>> {
    let json_file = File::open(json_file)?;
    let json: Value = serde_json::from_reader(&json_file).map_err(E::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };

    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    Ok(safetensors_files)
}

pub fn load_safetensors<P: AsRef<Path>>(path: P, json_file: P) -> Result<Vec<PathBuf>> {
    let path = path.as_ref();
    let safetensors_files = read_safetensors_index_file(path.join(json_file))?;
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| path.join(v))
        .collect();
    Ok(safetensors_files)
}

#[cfg(test)]
mod tests {

    #[test]
    fn local_repo() -> crate::Result<()> {
        use super::*;
        use std::path::PathBuf;

        let root = PathBuf::from(
            "/Users/kigi/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        );

        let repo = LocalRepo::new("Qwen/Qwen2.5-1.5B-Instruct", &root);

        assert_eq!(repo.model_id(), "Qwen/Qwen2.5-1.5B-Instruct");
        assert_eq!(repo.config_file()?, root.join("config.json"));
        assert_eq!(
            repo.tokenizer_config_file()?,
            root.join("tokenizer_config.json")
        );
        assert_eq!(repo.tokenizer_file()?, root.join("tokenizer.json"));
        assert_eq!(
            repo.generate_config_file()?,
            root.join("generate_config.json")
        );
        assert_eq!(
            repo.safetensors_files()?,
            vec![root.join("model.safetensors")],
        );
        Ok(())
    }
}
