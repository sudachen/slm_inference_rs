use crate::SlmFormatter;
use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

/// The result of a single generation call, capturing both the generated text and
/// the completion state of that generation.
///
/// All three variants carry `(text, fork_id)`.  `Complete` additionally
/// holds an optional `thinking` string extracted by [`split_thought`](Self::split_thought).
#[derive(Clone, Debug)]
pub enum SlmAnswer {
    /// Generation finished naturally (EOS token reached or constraint satisfied).
    /// The third field contains the model's chain-of-thought, if the formatter
    /// supports reasoning and [`split_thought`](Self::split_thought) has been called.
    // (answer, fork_id, thinking)
    Complete(String, usize, Option<String>),
    /// Generation was stopped early by an [`SlmAction::Delay`] callback.
    /// The accumulated text so far is valid but the sequence is not yet closed.
    Partial(String, usize),
    /// Generation was interrupted (e.g. token limit exceeded) before a natural stop.
    Incomplete(String, usize),
}

impl SlmAnswer {
    /// Returns `true` if the answer completed naturally (EOS or constraint stop).
    pub fn is_complete(&self) -> bool {
        matches!(self, SlmAnswer::Complete(_, _, _))
    }
    /// Returns `true` if the answer was paused mid-generation by a `Delay` action.
    pub fn is_partial(&self) -> bool {
        matches!(self, SlmAnswer::Partial(_, _))
    }

    /// Returns the generated text regardless of completion state.
    pub fn as_str(&self) -> &str {
        match self {
            SlmAnswer::Complete(s, _, _)
            | SlmAnswer::Partial(s, _)
            | SlmAnswer::Incomplete(s, _) => s.as_str(),
        }
    }

    /// Alias for [`as_str`](Self::as_str).
    pub fn text(&self) -> &str {
        self.as_str()
    }

    /// Returns the fork/sequence identifier associated with this answer.
    pub fn fork_id(&self) -> usize {
        match self {
            SlmAnswer::Complete(_, id, _)
            | SlmAnswer::Partial(_, id)
            | SlmAnswer::Incomplete(_, id) => *id,
        }
    }

    /// Apply a transformation function to the inner text string, preserving variant and
    /// metadata (fork ID, thinking content).
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

    /// Extract any chain-of-thought content from the text using the formatter's
    /// [`reasoning_bounds`](SlmFormatter::reasoning_bounds) tags, storing it in the
    /// `thinking` field of [`SlmAnswer::Complete`].
    ///
    /// Has no effect on `Partial` or `Incomplete` variants, or if a thinking string
    /// is already present.
    pub fn split_thought(self, formatter: &dyn SlmFormatter) -> SlmAnswer {
        match self {
            Self::Complete(text, fork_id, None) => {
                let (text, thought) = formatter.strip_thought(&text);
                SlmAnswer::Complete(text, fork_id, thought)
            }
            _ => self,
        }
    }

    /// Returns the extracted chain-of-thought string, if available.
    ///
    /// Only populated for [`SlmAnswer::Complete`] after [`split_thought`](Self::split_thought)
    /// has been called.
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
