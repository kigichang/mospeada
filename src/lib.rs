#[cfg(feature = "hf_hub")]
pub mod hf_hub;

pub mod generation;
pub mod tokenizers;

use candle_core::utils;
use candle_core::{Device, Error as E, Result, bail};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

pub fn load_safetensors<P: AsRef<Path>>(path: P, json_file: &str) -> Result<Vec<PathBuf>> {
    let path = path.as_ref();
    let safetensors_files = read_safetensors_index_file(path.join(json_file))?;
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| path.join(v))
        .collect();
    Ok(safetensors_files)
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

pub fn device(cpu: bool, index: usize) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if utils::cuda_is_available() {
        Ok(Device::new_cuda(index)?)
    } else if utils::metal_is_available() {
        Ok(Device::new_metal(index)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn cpu() -> Result<Device> {
    device(true, 0)
}

pub fn gpu(index: usize) -> Result<Device> {
    device(false, index)
}
