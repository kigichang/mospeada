#[cfg(feature = "chat-template")]
use crate::chat_template;

use crate::{Error as E, Result, bail, generation::GenerationConfig};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

/// 代表模型 repo
pub trait Repo {
    /// 模型 ID
    fn model_id(&self) -> &str;

    /// 取得指定檔案路徑
    fn get(&self, filename: &str) -> Result<PathBuf>;

    /// tokenizer_config.json 檔案路徑
    fn tokenizer_config_file(&self) -> Result<PathBuf>;

    /// tokenizer.json 檔案路徑
    fn tokenizer_file(&self) -> Result<PathBuf>;

    /// config.json 檔案路徑
    fn config_file(&self) -> Result<PathBuf>;

    /// 所有 safetensors 檔案路徑
    fn safetensors_files(&self) -> Result<Vec<PathBuf>>;

    /// pytorch_model.bin 檔案路徑
    fn pytorch_model_file(&self) -> Result<PathBuf>;

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

    /// 載入模型
    fn load_model<C, M, F>(&self, dtype: DType, device: &Device, load: F) -> Result<M>
    where
        C: serde::de::DeserializeOwned,
        F: Fn(&C, VarBuilder) -> candle_core::Result<M>,
    {
        let config: C = self.config()?;

        let vb = if let Ok(safetensor_files) = self.safetensors_files() {
            unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device) }
        } else {
            let pytorch_model_file = self.pytorch_model_file()?;
            VarBuilder::from_pth(pytorch_model_file, dtype, device)
        }?;

        Ok(load(&config, vb)?)
    }

    // 避開 R: std::io::Seek + std::io::Read, 與 File 型別不同的問題。
    #[inline(always)]
    fn call_from_gguf<R, F, M>(
        &self,
        ct: gguf_file::Content,
        f: &mut R,
        device: &Device,
        load: F,
    ) -> candle_core::Result<M>
    where
        R: std::io::Seek + std::io::Read,
        F: Fn(gguf_file::Content, &mut R, &Device) -> candle_core::Result<M>,
    {
        load(ct, f, device)
    }

    /// 載入 gguf 模型
    fn load_gguf<M, F>(&self, filename: &str, device: &Device, load: F) -> Result<M>
    where
        F: Fn(gguf_file::Content, &mut File, &Device) -> candle_core::Result<M>,
    {
        let mut reader = File::open(self.get(filename)?)?;
        let model = gguf_file::Content::read(&mut reader)?;

        Ok(self.call_from_gguf(model, &mut reader, device, load)?)
    }

    /// 載入 huggingface tokenizer
    fn load_tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        Ok(tokenizers::Tokenizer::from_file(self.tokenizer_file()?)?)
    }

    #[cfg(feature = "chat-template")]
    fn load_chat_template(&self) -> Result<chat_template::ChatTemplate> {
        let tokenizer_config = self.tokenizer_config_file()?;
        let tokenizer_config: serde_json::Value =
            serde_json::from_reader(File::open(tokenizer_config)?)?;

        let chat_template = tokenizer_config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or(crate::Error::msg("chat_template not found"))?;

        chat_template::ChatTemplate::new(chat_template)
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

    fn get_file<P: AsRef<Path>>(&self, p: P) -> PathBuf {
        self.path.join(p)
    }
}

impl Repo for LocalRepo {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn get(&self, filename: &str) -> Result<PathBuf> {
        Ok(self.get_file(filename))
    }

    fn tokenizer_config_file(&self) -> Result<PathBuf> {
        Ok(self.get_file("tokenizer_config.json"))
    }

    fn tokenizer_file(&self) -> Result<PathBuf> {
        Ok(self.get_file("tokenizer.json"))
    }

    fn config_file(&self) -> Result<PathBuf> {
        Ok(self.get_file("config.json"))
    }

    fn safetensors_files(&self) -> Result<Vec<PathBuf>> {
        let single_safatensors_file = self.get_file("model.safetensors");
        if single_safatensors_file.exists() {
            return Ok(vec![single_safatensors_file]);
        }

        let index_file = self.get_file("model.safetensors.index.json");
        load_safetensors(&self.path, &index_file)
    }

    fn pytorch_model_file(&self) -> Result<PathBuf> {
        Ok(self.get_file("pytorch_model.bin"))
    }

    fn generate_config_file(&self) -> Result<PathBuf> {
        Ok(self.get_file("generate_config.json"))
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
