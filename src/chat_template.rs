use std::fs::File;
use std::ops::Deref;

use crate::repo::Repo;
use crate::{Result, error};
use minijinja::{Environment, Template};
use minijinja_contrib::pycompat;

#[derive(Clone)]
pub struct ChatTemplate {
    template: Template<'static, 'static>,
}

impl Deref for ChatTemplate {
    type Target = Template<'static, 'static>;

    fn deref(&self) -> &Self::Target {
        &self.template
    }
}

impl ChatTemplate {
    fn init_env<'a>() -> Environment<'a> {
        let mut env = Environment::new();
        // 加入 python 相容的 function, like str.startswith, str.endswith
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env
    }

    pub fn new<S: AsRef<str>>(template: S) -> Result<Self> {
        let env = Box::new(Self::init_env());

        // 將 str 轉成 Box<String>，以便使用 Box::leak
        let template_str = template.as_ref().to_string().into_boxed_str();
        Ok(ChatTemplate {
            // 透過 Box::leak 轉成 'static 的生命週期
            template: Box::leak(env).template_from_str(Box::leak(template_str))?,
        })
    }

    // pub fn apply<S: serde::Serialize>(&self, msg: S) -> Result<String> {
    //     Ok(self.template.render(msg)?)
    // }
}

pub fn from_pretrained<R: Repo>(repo: &R) -> Result<ChatTemplate> {
    let tokenizer_config = repo.tokenizer_config_file()?;
    let tokenizer_config: serde_json::Value =
        serde_json::from_reader(File::open(tokenizer_config)?)?;

    let chat_template = tokenizer_config
        .get("chat_template")
        .and_then(|v| v.as_str())
        .ok_or(error::Error::msg("chat_template not found"))?;

    ChatTemplate::new(chat_template)
}
