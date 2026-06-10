use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

#[derive(PartialEq,Eq,Copy,Clone,Debug)]
pub enum SlmBrake {
    // prevent following generation and returns Complete Answer
    Finish,
    // prevent following generation and returns Incomplete Answer
    Stop,
    // puts sampled token to the batch for continuation and returns Incomplete Answer
    // any following prompt will terminate generation
    // is not applicable to aks_for
    Delay,
    Continue,
}

impl SlmBrake {
    pub fn token_limit(max_tokens: usize) -> impl Fn(/*answer*/&str,/*last_token*/&str,/*n_tokens*/usize) -> SlmBrake {
        move |_,_,n_tokens| match n_tokens >= max_tokens {
            true => SlmBrake::Stop,
            false => SlmBrake::Continue,
        }
    }

    pub fn brake(&self) -> bool {
        matches!(self, SlmBrake::Finish | SlmBrake::Stop | SlmBrake::Delay)
    }
}

pub type SlmBrakeFilter = dyn Fn(/*answer*/&str,/*last_token*/&str,/*n_tokens*/usize) -> SlmBrake;

pub enum SlmAnswer {
    // (answer, fork_id)
    Complete(String,usize),
    Partial(String,usize),
}

impl SlmAnswer {
    pub fn is_complete(&self) -> bool {
        matches!(self, SlmAnswer::Complete(_,_))
    }

    pub fn as_str(&self) -> &str {
        match self {
            SlmAnswer::Complete(s,_) | SlmAnswer::Partial(s,_) => s.as_str(),
        }
    }

    pub fn fork_id(&self) -> usize {
        match self {
            SlmAnswer::Complete(_,id) | SlmAnswer::Partial(_,id) => *id,
        }
    }

    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(String) -> String,
    {
        match self {
            Self::Complete(text, fork_id) => Self::Complete(f(text), fork_id),
            Self::Partial(text, fork_id) => Self::Partial(f(text), fork_id),
        }
    }
}

impl Deref for SlmAnswer {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for SlmAnswer {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for SlmAnswer {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<SlmAnswer> for String {
    fn from(a: SlmAnswer) -> String {
        match a {
            SlmAnswer::Complete(s,_) | SlmAnswer::Partial(s,_) => s,
        }
    }
}

impl fmt::Display for SlmAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
