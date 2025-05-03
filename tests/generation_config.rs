use anyhow::Result;
use candle_transformers::generation::Sampling;
use mospeada::generation::GenerationConfig;

#[test]
fn load_generation_config() -> Result<()> {
    let config = serde_json::from_str::<GenerationConfig>(
        r#"{
            "bos_token_id": 151643,
            "do_sample": false,
            "eos_token_id": 151643,
            "max_new_tokens": 2048,
            "transformers_version": "4.43.1"
        }"#,
    )?;

    assert_eq!(config.get_eos_token_id(), Some(vec![151643]));
    assert_eq!(config.sampling(), Sampling::ArgMax);

    let config = serde_json::from_str::<GenerationConfig>(
        r#"{
            "bos_token_id": 128000,
            "do_sample": true,
            "eos_token_id": [
              128001,
              128008,
              128009
            ],
            "temperature": 0.6,
            "top_p": 0.9,
            "transformers_version": "4.45.0.dev0"
          }"#,
    )?;

    assert_eq!(
        config.get_eos_token_id(),
        Some(vec![128001, 128008, 128009])
    );

    Ok(())
}
