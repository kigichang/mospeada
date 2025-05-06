use candle_core::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use mospeada::{Result, error, repo::Repo};

use minijinja::{Environment, context};

struct Qwen2ModelForCausalLM(ModelForCausalLM);

impl mospeada::generation::Model for Qwen2ModelForCausalLM {
    #[inline]
    fn forward(
        &mut self,
        x: &candle_core::Tensor,
        start_pos: usize,
    ) -> Result<candle_core::Tensor> {
        Ok(self.0.forward(x, start_pos)?)
    }

    #[inline]
    fn reset(&mut self) {
        self.0.clear_kv_cache();
    }
}

#[test]
fn generate_with_qwen_25() -> Result<()> {
    let mut env: Environment<'static> = Environment::new();
    let model_id: &'static str = "Qwen/Qwen2.5-0.5B-Instruct";
    let system_promp = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    let user_prompt = "Give me a short introduction to large language model.";

    let device = mospeada::utils::gpu(0)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    println!("repo init");
    let repo = mospeada::hf_hub::ApiRepo::from_pretrained(model_id, None, None, None)?;

    println!("tokenizer init");
    let mut tokenizer = mospeada::tokenizers::from_pretrained(&repo, &mut env)?;

    println!("generation config init");
    let generation_config = mospeada::generation::GenerationConfig::from_pretrained(&repo)?;

    println!("init model");
    let config: Config = repo.config()?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&repo.safetensors_files()?, dtype, &device)? };
    let model = Qwen2ModelForCausalLM(ModelForCausalLM::new(&config, vb)?);

    let prompt = tokenizer.apply_chat_template(context! {
    messages => vec![
        context!{
            role => "system",
            content => system_promp,
        },
        context! {
            role => "user",
            content => user_prompt,
        }
    ],
    // 依 chat_template 的定義，加入 add_generation_prompt
    // 會在最後加 '<|im_start|>assistant\n'
    // 才會正確生成答案，否則在生成時，會出現奇怪的 token，eg: 'ystem\n'
    add_generation_prompt => true,})?;
    println!("prompt init {:?}", prompt);

    let mut pipeline =
        mospeada::generation::TextGeneration::new(model, device, &generation_config, 0, 64);

    let mut tokens = vec![];

    let mut cb = |token: u32| {
        tokens.push(token);
        let t = &tokenizer.next_token(token);
        if let Ok(Some(t)) = t {
            print!("{}", t);
        }
    };

    //pipeline.run(prompt, 1024, cb)?;

    if let Ok(next_token) = pipeline.apply(prompt, 1024) {
        cb(next_token);
    }

    loop {
        match pipeline.next() {
            Ok(next_token) => {
                cb(next_token);
            }
            Err(error::Error::Eos {
                eos_token_id,
                generated,
            }) => {
                cb(eos_token_id);
                println!("\n\nEos token: {eos_token_id}, generated: {generated}");
                break;
            }
            Err(e) => {
                println!("{:?}", e);
                break;
            }
        }
    }

    println!("got {} tokens", tokens.len());
    let ans1 = tokenizer.decode_all()?;
    assert_eq!(
        ans1,
        "Sure! A large language model is a type of artificial intelligence (AI) that can generate human-like text. These models use deep neural networks to process and understand natural language, allowing them to produce coherent and contextually relevant responses. They are commonly used in various applications such as chatbots, writing assistants, language translation, and more. Large language models have been trained on vast amounts of data from the internet and other sources, enabling them to generate high-quality text that is consistent with human-like intent and style."
    );

    let ans2 = tokenizer.decode(&tokens)?;
    assert_eq!(ans2, ans1);

    Ok(())
}
