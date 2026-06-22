use crate::SlmFormatter;
use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

#[derive(Clone, Debug)]
pub enum SlmAnswer {
    // (answer, fork_id, thinking)
    Complete(String, usize, Option<String>),
    Partial(String, usize),
    Incomplete(String, usize),
}

impl SlmAnswer {
    pub fn is_complete(&self) -> bool {
        matches!(self, SlmAnswer::Complete(_, _, _))
    }
    pub fn is_partial(&self) -> bool {
        matches!(self, SlmAnswer::Partial(_, _))
    }

    pub fn as_str(&self) -> &str {
        match self {
            SlmAnswer::Complete(s, _, _)
            | SlmAnswer::Partial(s, _)
            | SlmAnswer::Incomplete(s, _) => s.as_str(),
        }
    }

    pub fn text(&self) -> &str {
        self.as_str()
    }

    pub fn fork_id(&self) -> usize {
        match self {
            SlmAnswer::Complete(_, id, _)
            | SlmAnswer::Partial(_, id)
            | SlmAnswer::Incomplete(_, id) => *id,
        }
    }

    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(String) -> String,
    {
        match self {
            Self::Complete(text, fork_id, thought) => Self::Complete(f(text), fork_id, thought),
            Self::Partial(text, fork_id) => Self::Partial(f(text), fork_id),
            Self::Incomplete(text, fork_id) => Self::Incomplete(f(text), fork_id),
        }
    }

    pub fn split_thought(self, formatter: &dyn SlmFormatter) -> SlmAnswer {
        match self {
            Self::Complete(text, fork_id, None) => {
                let (text, thought) = formatter.strip_thought(&text);
                SlmAnswer::Complete(text, fork_id, thought)
            }
            _ => self,
        }
    }

    pub fn thought(&self) -> Option<&str> {
        match self {
            Self::Complete(_, _, thought) => thought.as_deref(),
            _ => None,
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
            SlmAnswer::Complete(s, _, _)
            | SlmAnswer::Partial(s, _)
            | SlmAnswer::Incomplete(s, _) => s,
        }
    }
}

impl fmt::Display for SlmAnswer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
