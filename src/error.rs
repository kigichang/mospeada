#[derive(thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Wrapped(Box<dyn std::fmt::Display + Send + Sync>),

    #[error("got eos token {eos_token_id} and {generated} tokens generated")]
    Eos { eos_token_id: u32, generated: usize },

    #[error("max new tokens {max_new_tokens} exceeded")]
    MaxNewTokenExceeded { max_new_tokens: usize },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Msg(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),

    #[error(transparent)]
    MiniJinja(#[from] minijinja::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[cfg(feature = "hf_hub")]
    #[error(transparent)]
    HfHub(#[from] hf_hub::api::sync::ApiError),
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Error {
    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }

    pub fn wrap(err: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err)).bt()
    }

    pub fn msg(err: impl std::fmt::Display) -> Self {
        Self::Msg(err.to_string()).bt()
    }

    // #[cfg(feature = "hf_hub")]
    // pub fn hub(err: hf_hub::api::sync::ApiError) -> Self {
    //     Self::HfHub(err).bt()
    // }

    // pub fn json(err: serde_json::Error) -> Self {
    //     Self::Json(err).bt()
    // }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

pub type Result<T> = std::result::Result<T, Error>;
