#!/usr/bin/env python3
"""
Advanced Spell Correction System for Non-English Words in English Script

This system combines multiple algorithms for maximum accuracy:
1. Exact matching (case-insensitive)
2. Phonetic similarity (custom Soundex, Metaphone)
3. String similarity (Levenshtein, Jaro-Winkler, LCS)
4. Fuzzy matching with configurable thresholds

Optimized for Hindi, Marathi, and other Indian language transliterations.
"""

import sys
import argparse
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import heapq
from tqdm import tqdm

from phonetic_matcher import PhoneticMatcher
from utils import StringSimilarity, TextProcessor, PerformanceMonitor, timing_decorator


class SpellCorrector:
    """
    Advanced spell correction system with multi-algorithm approach.
    """
    
    def __init__(self, dictionary_path: str, 
                 phonetic_threshold: float = 0.7,
                 similarity_threshold: float = 0.8,
                 max_edit_distance: int = 3):
        """
        Initialize the spell corrector.
        
        Args:
            dictionary_path: Path to the reference dictionary file
            phonetic_threshold: Minimum phonetic similarity score (0-1)
            similarity_threshold: Minimum string similarity score (0-1)
            max_edit_distance: Maximum allowed edit distance
        """
        self.phonetic_threshold = phonetic_threshold
        self.similarity_threshold = similarity_threshold
        self.max_edit_distance = max_edit_distance
        
        # Load dictionary and create indices
        print("Loading dictionary...")
        self.dictionary = TextProcessor.load_dictionary(dictionary_path)
        print(f"Loaded {len(self.dictionary)} words from dictionary")
        
        # Initialize components
        self.phonetic_matcher = PhoneticMatcher()
        self.string_similarity = StringSimilarity()
        self.performance_monitor = PerformanceMonitor()
        
        # Create lookup indices for faster searching
        self._create_indices()
    
    def _create_indices(self):
        """Create various indices for faster lookup."""
        print("Creating search indices...")
        
        # Exact match index (case-insensitive)
        self.exact_index = {word.lower(): word for word in self.dictionary}
        
        # Soundex index for phonetic matching
        self.soundex_index = defaultdict(list)
        for word in self.dictionary:
            soundex_code = self.phonetic_matcher.custom_soundex(word)
            self.soundex_index[soundex_code].append(word)
        
        # Length-based index for efficient filtering
        self.length_index = defaultdict(list)
        for word in self.dictionary:
            self.length_index[len(word)].append(word)
        
        # First letter index
        self.first_letter_index = defaultdict(list)
        for word in self.dictionary:
            if word:
                self.first_letter_index[word[0].lower()].append(word)
    
    def _get_candidates_by_length(self, word: str, max_length_diff: int = 2) -> List[str]:
        """Get candidate words with similar length."""
        candidates = []
        word_len = len(word)
        
        for length in range(max(1, word_len - max_length_diff), 
                          word_len + max_length_diff + 1):
            candidates.extend(self.length_index[length])
        
        return candidates
    
    def _get_candidates_by_first_letter(self, word: str) -> List[str]:
        """Get candidate words starting with the same letter."""
        if not word:
            return []
        
        first_letter = word[0].lower()
        return self.first_letter_index.get(first_letter, [])
    
    def _exact_match(self, word: str) -> Optional[str]:
        """Check for exact match (case-insensitive)."""
        return self.exact_index.get(word.lower())
    
    def _phonetic_match(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find phonetically similar words."""
        candidates = []
        
        # Get candidates from soundex index
        soundex_code = self.phonetic_matcher.custom_soundex(word)
        soundex_candidates = self.soundex_index.get(soundex_code, [])
        
        # Add candidates from similar soundex codes
        for code, words in self.soundex_index.items():
            if code != soundex_code and len(set(code) & set(soundex_code)) >= 2:
                soundex_candidates.extend(words)
        
        # Calculate phonetic distances
        for candidate in soundex_candidates:
            distance = self.phonetic_matcher.phonetic_distance(word, candidate)
            similarity = 1.0 - distance
            
            if similarity >= self.phonetic_threshold:
                candidates.append((candidate, similarity))
        
        # Sort by similarity and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _fuzzy_match(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar words using string similarity algorithms."""
        candidates = []
        
        # Get candidates by length and first letter for efficiency
        length_candidates = self._get_candidates_by_length(word)
        first_letter_candidates = self._get_candidates_by_first_letter(word)
        
        # Combine and deduplicate candidates
        all_candidates = list(set(length_candidates + first_letter_candidates))
        
        # If too many candidates, limit to most promising ones
        if len(all_candidates) > 1000:
            all_candidates = length_candidates[:500] + first_letter_candidates[:500]
        
        for candidate in all_candidates:
            # Skip if edit distance is too large (quick filter)
            edit_distance = self.string_similarity.levenshtein_distance(word.lower(), candidate.lower())
            if edit_distance > self.max_edit_distance:
                continue
            
            # Calculate multiple similarity scores
            jaro_winkler = self.string_similarity.jaro_winkler_similarity(word.lower(), candidate.lower())
            lcs_sim = self.string_similarity.lcs_similarity(word.lower(), candidate.lower())
            
            # Weighted combination of similarities
            combined_similarity = (
                0.5 * jaro_winkler +
                0.3 * lcs_sim +
                0.2 * (1.0 - edit_distance / max(len(word), len(candidate)))
            )
            
            if combined_similarity >= self.similarity_threshold:
                candidates.append((candidate, combined_similarity))
        
        # Sort by similarity and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def correct_word(self, word: str) -> Tuple[str, str, float]:
        """
        Correct a single word using multi-stage approach.
        
        Returns:
            Tuple of (corrected_word, correction_type, confidence_score)
        """
        if not word or not TextProcessor.is_valid_word(word):
            return word, "invalid", 0.0
        
        # Stage 1: Exact match
        exact_match = self._exact_match(word)
        if exact_match:
            return exact_match, "exact", 1.0
        
        # Stage 2: Phonetic matching
        phonetic_matches = self._phonetic_match(word)
        if phonetic_matches:
            best_match, confidence = phonetic_matches[0]
            if confidence > 0.85:  # High confidence phonetic match
                return best_match, "phonetic", confidence
        
        # Stage 3: Fuzzy string matching
        fuzzy_matches = self._fuzzy_match(word)
        if fuzzy_matches:
            best_fuzzy, fuzzy_confidence = fuzzy_matches[0]
            
            # Compare phonetic and fuzzy matches
            if phonetic_matches:
                best_phonetic, phonetic_confidence = phonetic_matches[0]
                
                # Choose the better match
                if phonetic_confidence > fuzzy_confidence:
                    return best_phonetic, "phonetic", phonetic_confidence
                else:
                    return best_fuzzy, "fuzzy", fuzzy_confidence
            else:
                return best_fuzzy, "fuzzy", fuzzy_confidence
        
        # Stage 4: Return best phonetic match if available
        if phonetic_matches:
            best_match, confidence = phonetic_matches[0]
            return best_match, "phonetic", confidence
        
        # No suitable correction found
        return word, "no_match", 0.0
    
    @timing_decorator
    def correct_file(self, input_path: str, output_path: str) -> Dict[str, float]:
        """
        Correct all words in a file and save results.
        
        Returns:
            Performance statistics dictionary
        """
        print(f"Loading error words from {input_path}...")
        error_words = TextProcessor.load_error_words(input_path)
        print(f"Loaded {len(error_words)} words to correct")
        
        corrections = []
        self.performance_monitor.start_timing()
        
        print("Correcting words...")
        for word in tqdm(error_words, desc="Processing"):
            corrected, correction_type, confidence = self.correct_word(word)
            corrections.append((word, corrected))
            self.performance_monitor.record_correction(correction_type)
        
        print(f"Saving corrections to {output_path}...")
        TextProcessor.save_corrections(corrections, output_path)
        
        stats = self.performance_monitor.get_stats()
        self._print_statistics(stats)
        
        return stats
    
    def _print_statistics(self, stats: Dict[str, float]):
        """Print performance statistics."""
        print("\n" + "="*50)
        print("CORRECTION STATISTICS")
        print("="*50)
        print(f"Total corrections: {stats['total_corrections']}")
        print(f"Exact matches: {stats['exact_matches']}")
        print(f"Phonetic matches: {stats['phonetic_matches']}")
        print(f"Fuzzy matches: {stats['fuzzy_matches']}")
        print(f"No matches found: {stats['no_matches']}")
        print(f"Accuracy rate: {stats['accuracy_rate']:.2%}")
        print(f"Processing time: {stats['elapsed_time']:.2f} seconds")
        print(f"Speed: {stats['corrections_per_second']:.1f} corrections/second")
        print("="*50)
    
    def batch_correct(self, words: List[str]) -> List[Tuple[str, str, float]]:
        """Correct a batch of words."""
        results = []
        for word in words:
            corrected, correction_type, confidence = self.correct_word(word)
            results.append((word, corrected, confidence))
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Advanced Spell Correction for Non-English Words"
    )
    parser.add_argument("error_file", help="Path to file containing misspelled words")
    parser.add_argument("dictionary_file", help="Path to reference dictionary file")
    parser.add_argument("output_file", help="Path to output corrected words")
    parser.add_argument("--phonetic-threshold", type=float, default=0.7,
                       help="Phonetic similarity threshold (0-1)")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                       help="String similarity threshold (0-1)")
    parser.add_argument("--max-edit-distance", type=int, default=3,
                       help="Maximum edit distance for corrections")
    
    args = parser.parse_args()
    
    try:
        # Initialize corrector
        corrector = SpellCorrector(
            args.dictionary_file,
            phonetic_threshold=args.phonetic_threshold,
            similarity_threshold=args.similarity_threshold,
            max_edit_distance=args.max_edit_distance
        )
        
        # Process file
        stats = corrector.correct_file(args.error_file, args.output_file)
        
        print(f"\nCorrections saved to: {args.output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()