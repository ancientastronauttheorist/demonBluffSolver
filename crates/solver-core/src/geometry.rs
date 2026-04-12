/// Shortest distance between positions a and b on a circle of size n.
/// Positions are 1-indexed.
pub fn circle_distance(a: u8, b: u8, n: u8) -> u8 {
    let diff = (a as i16 - b as i16).unsigned_abs() as u8;
    diff.min(n - diff)
}

/// Direction from `from_pos` to `to_pos` on a circle.
/// CW means increasing position numbers (1->2->3...).
/// Returns Equidistant if exactly opposite.
pub fn circle_direction(from_pos: u8, to_pos: u8, n: u8) -> Direction {
    if from_pos == to_pos {
        return Direction::Equidistant;
    }
    let cw_dist = ((to_pos as i16 - from_pos as i16).rem_euclid(n as i16)) as u8;
    let ccw_dist = ((from_pos as i16 - to_pos as i16).rem_euclid(n as i16)) as u8;
    match cw_dist.cmp(&ccw_dist) {
        std::cmp::Ordering::Less => Direction::CW,
        std::cmp::Ordering::Greater => Direction::CCW,
        std::cmp::Ordering::Equal => Direction::Equidistant,
    }
}

/// The two positions adjacent to `pos` on a circle of size n. 1-indexed.
pub fn adjacent_positions(pos: u8, n: u8) -> [u8; 2] {
    let left = ((pos as i16 - 2).rem_euclid(n as i16) + 1) as u8;
    let right = (pos % n) + 1;
    [left, right]
}

/// All positions within `range_val` steps of `pos` (excluding pos itself).
pub fn positions_in_range(pos: u8, range_val: u8, n: u8) -> Vec<u8> {
    let mut result = Vec::new();
    for d in 1..=range_val {
        let cw = (((pos as i16 - 1 + d as i16) % n as i16) + 1) as u8;
        let ccw = (((pos as i16 - 1 - d as i16).rem_euclid(n as i16)) + 1) as u8;
        result.push(cw);
        if cw != ccw {
            result.push(ccw);
        }
    }
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    CW,
    CCW,
    Equidistant,
}

impl Direction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Direction::CW => "CW",
            Direction::CCW => "CCW",
            Direction::Equidistant => "Equidistant",
        }
    }

    /// Parse from various user input formats.
    pub fn from_str_loose(s: &str) -> Option<Direction> {
        match s.to_lowercase().as_str() {
            "cw" | "clockwise" => Some(Direction::CW),
            "ccw" | "counterclockwise" | "counter-clockwise" => Some(Direction::CCW),
            "equidistant" | "equal" => Some(Direction::Equidistant),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_distance() {
        assert_eq!(circle_distance(1, 3, 7), 2);
        assert_eq!(circle_distance(1, 6, 7), 2);
        assert_eq!(circle_distance(1, 1, 7), 0);
        assert_eq!(circle_distance(1, 4, 7), 3);
        assert_eq!(circle_distance(1, 5, 7), 3); // min(4, 3) = 3
    }

    #[test]
    fn test_circle_direction() {
        assert_eq!(circle_direction(1, 2, 7), Direction::CW);
        assert_eq!(circle_direction(1, 7, 7), Direction::CCW);
        assert_eq!(circle_direction(1, 1, 7), Direction::Equidistant);
        // On even circle, opposite is equidistant
        assert_eq!(circle_direction(1, 5, 8), Direction::Equidistant);
    }

    #[test]
    fn test_adjacent_positions() {
        assert_eq!(adjacent_positions(1, 7), [7, 2]);
        assert_eq!(adjacent_positions(7, 7), [6, 1]);
        assert_eq!(adjacent_positions(4, 7), [3, 5]);
    }

    #[test]
    fn test_positions_in_range() {
        let r = positions_in_range(1, 2, 7);
        assert_eq!(r, vec![2, 7, 3, 6]);
    }
}
