use regex::Regex;

/// Detects advertisement cues in transcript text.
pub struct CueDetector {
    url_pattern: Regex,
    promo_pattern: Regex,
    phone_pattern: Regex,
    cta_pattern: Regex,
    transition_pattern: Regex,
    self_promo_pattern: Regex,
}

#[derive(Debug, Clone, Default)]
pub struct CueAnalysis {
    pub url: bool,
    pub promo: bool,
    pub phone: bool,
    pub cta: bool,
    pub transition: bool,
    pub self_promo: bool,
}

impl CueAnalysis {
    pub fn has_strong_cue(&self) -> bool {
        self.url || self.promo || self.phone || self.cta
    }
}

impl CueDetector {
    pub fn new() -> Self {
        Self {
            url_pattern: Regex::new(r"(?i)\b[a-z0-9\-\.]+\.(?:com|net|org|io)\b").unwrap(),
            promo_pattern: Regex::new(r"(?i)\b(?:code|promo|save|discount)\s+\w+\b").unwrap(),
            phone_pattern: Regex::new(r"\b(?:\+?1[ -]?)?\d{3}[ -]?\d{3}[ -]?\d{4}\b").unwrap(),
            cta_pattern: Regex::new(
                r"(?i)\b(?:visit|go to|check out|head over|sign up|start today|start now|use code|offer|deal|free trial)\b",
            )
            .unwrap(),
            transition_pattern: Regex::new(
                r"(?i)\b(?:back to the show|after the break|stay tuned|we'll be right back|now back)\b",
            )
            .unwrap(),
            self_promo_pattern: Regex::new(
                r"(?i)\b(?:my|our)\s+(?:book|course|newsletter|fund|patreon|substack|community|platform)\b",
            )
            .unwrap(),
        }
    }

    pub fn analyze(&self, text: &str) -> CueAnalysis {
        CueAnalysis {
            url: self.url_pattern.is_match(text),
            promo: self.promo_pattern.is_match(text),
            phone: self.phone_pattern.is_match(text),
            cta: self.cta_pattern.is_match(text),
            transition: self.transition_pattern.is_match(text),
            self_promo: self.self_promo_pattern.is_match(text),
        }
    }

    /// Highlight detected cues in text by wrapping them in *** *** markers.
    pub fn highlight_cues(&self, text: &str) -> String {
        let patterns = [
            &self.url_pattern,
            &self.promo_pattern,
            &self.phone_pattern,
            &self.cta_pattern,
            &self.transition_pattern,
            &self.self_promo_pattern,
        ];

        let mut spans: Vec<(usize, usize)> = Vec::new();
        for pat in &patterns {
            for m in pat.find_iter(text) {
                spans.push((m.start(), m.end()));
            }
        }

        if spans.is_empty() {
            return text.to_string();
        }

        // Sort by start, then reverse end for containment
        spans.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

        // Merge overlapping intervals
        let mut merged: Vec<(usize, usize)> = Vec::new();
        let (mut cs, mut ce) = spans[0];
        for &(ns, ne) in &spans[1..] {
            if ns < ce {
                ce = ce.max(ne);
            } else {
                merged.push((cs, ce));
                cs = ns;
                ce = ne;
            }
        }
        merged.push((cs, ce));

        // Reconstruct string backwards
        let mut result = String::with_capacity(text.len() + merged.len() * 8);
        let mut last = 0;
        for (start, end) in &merged {
            result.push_str(&text[last..*start]);
            result.push_str("*** ");
            result.push_str(&text[*start..*end]);
            result.push_str(" ***");
            last = *end;
        }
        result.push_str(&text[last..]);
        result
    }
}
