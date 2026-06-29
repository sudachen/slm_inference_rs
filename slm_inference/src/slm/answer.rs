use super::Formatter;
use std::borrow::Borrow;
use std::fmt;
use std::fmt::Display;
use std::ops::Deref;

/// The result of a single generation call, capturing both the generated text and
/// the completion state of that generation.
///
/// All three variants carry `answer value`.  `Complete` additionally
/// holds an optional `thinking` string extracted by [`split_thought`](Self::split_thought).
#[derive(Clone, Debug)]
pub enum Answer<T> {
    /// Generation finished naturally (EOS token reached or constraint satisfied).
    /// The third field contains the model's chain-of-thought, if the formatter
    /// supports reasoning and [`split_thought`](Self::split_thought) has been called.
    // (answer, thinking)
    Complete(T, Option<String>),
    /// Generation was stopped early by an [`Action::Delay`](crate::slm::Action::Delay) callback.
    /// The accumulated text so far is valid but the sequence is not yet closed.
    Partial(T),
    /// Generation was interrupted (e.g. token limit exceeded) before a natural stop.
    Incomplete(T),
}

impl Answer<String> {
    /// Extract any chain-of-thought content from the text using the formatter's
    /// [`reasoning_bounds`](crate::slm::Formatter::reasoning_bounds) tags, storing it in the
    /// `thinking` field of [`Answer::Complete`].
    ///
    /// Has no effect on `Partial` or `Incomplete` variants, or if a thinking string
    /// is already present.
    pub fn split_thought(self, formatter: &dyn Formatter) -> Self {
        match self {
            Self::Complete(text, None) => {
                let (text, thought) = formatter.strip_thought(&text);
                Answer::Complete(text, thought)
            }
            _ => self,
        }
    }
}

impl<T: Display> Answer<T> {
    /// Returns the generated text regardless of completion state.
    pub fn text(&self) -> String {
        match self {
            Answer::Complete(s, _) | Answer::Partial(s) | Answer::Incomplete(s) => s.to_string(),
        }
    }
}

impl<T> Answer<T> {
    /// Returns `true` if the answer completed naturally (EOS or constraint stop).
    pub fn is_complete(&self) -> bool {
        matches!(self, Answer::Complete(_, _))
    }
    /// Returns `true` if the answer was paused mid-generation by a `Delay` action.
    pub fn is_partial(&self) -> bool {
        matches!(self, Answer::Partial(_))
    }

    /// Returns the generated text regardless of completion state.
    pub fn value(&self) -> &T {
        match self {
            Answer::Complete(t, _) | Answer::Partial(t) | Answer::Incomplete(t) => t,
        }
    }

    /// Apply a transformation function to the inner text string, preserving variant and
    /// metadata (fork ID, thinking content).
    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(T) -> T,
    {
        match self {
            Self::Complete(text, thought) => Self::Complete(f(text), thought),
            Self::Partial(text) => Self::Partial(f(text)),
            Self::Incomplete(text) => Self::Incomplete(f(text)),
        }
    }

    /// Returns the extracted chain-of-thought string, if available.
    ///
    /// Only populated for [`Answer::Complete`] after [`split_thought`](Self::split_thought)
    /// has been called.
    pub fn thought(&self) -> Option<&str> {
        match self {
            Self::Complete(_, thought) => thought.as_deref(),
            _ => None,
        }
    }
}

impl<T: Display> Deref for Answer<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value()
    }
}

impl AsRef<str> for Answer<String> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<T: Display> AsRef<T> for Answer<T> {
    fn as_ref(&self) -> &T {
        self.value()
    }
}

impl Borrow<str> for Answer<String> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<T: Display> From<Answer<T>> for String {
    fn from(a: Answer<T>) -> String {
        a.to_string()
    }
}

impl<T: Display> Display for Answer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value().fmt(f)
    }
}
