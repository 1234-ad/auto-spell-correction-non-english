"""
Utility functions for spell correction system.
"""

import re
import time
from typing import List, Tuple, Dict, Optional
from functools import wraps


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class StringSimilarity:
    """Advanced string similarity algorithms."""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        Optimized implementation with space complexity O(min(m,n)).
        """
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def weighted_levenshtein(s1: str, s2: str, 
                           insert_cost: float = 1.0,
                           delete_cost: float = 1.0,
                           substitute_cost: float = 1.0) -> float:
        """
        Calculate weighted Levenshtein distance with custom operation costs.
        """
        m, n = len(s1), len(s2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + delete_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + insert_cost
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + delete_cost,      # deletion
                        dp[i][j-1] + insert_cost,      # insertion
                        dp[i-1][j-1] + substitute_cost # substitution
                    )
        
        return dp[m][n]
    
    @staticmethod
    def jaro_similarity(s1: str, s2: str) -> float:
        """Calculate Jaro similarity between two strings."""
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Calculate the match window
        match_window = max(len1, len2) // 2 - 1
        match_window = max(0, match_window)
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches / len1 + matches / len2 + 
                (matches - transpositions / 2) / matches) / 3.0
        
        return jaro
    
    @staticmethod
    def jaro_winkler_similarity(s1: str, s2: str, prefix_scale: float = 0.1) -> float:
        """Calculate Jaro-Winkler similarity with prefix bonus."""
        jaro_sim = StringSimilarity.jaro_similarity(s1, s2)
        
        if jaro_sim < 0.7:
            return jaro_sim
        
        # Calculate common prefix length (up to 4 characters)
        prefix_len = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break
        
        return jaro_sim + (prefix_len * prefix_scale * (1 - jaro_sim))
    
    @staticmethod
    def longest_common_subsequence(s1: str, s2: str) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def lcs_similarity(s1: str, s2: str) -> float:
        """Calculate similarity based on longest common subsequence."""
        if not s1 or not s2:
            return 0.0
        
        lcs_len = StringSimilarity.longest_common_subsequence(s1, s2)
        max_len = max(len(s1), len(s2))
        
        return lcs_len / max_len


class TextProcessor:
    """Text processing utilities."""
    
    @staticmethod
    def clean_word(word: str) -> str:
        """Clean and normalize a word."""
        # Remove extra whitespace and convert to lowercase
        word = word.strip().lower()
        
        # Remove non-alphabetic characters except hyphens and apostrophes
        word = re.sub(r"[^a-zA-Z\-']", "", word)
        
        # Remove multiple consecutive hyphens or apostrophes
        word = re.sub(r"[-']{2,}", "", word)
        
        return word
    
    @staticmethod
    def is_valid_word(word: str) -> bool:
        """Check if a word is valid (contains only letters and basic punctuation)."""
        if not word or len(word) < 1:
            return False
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', word):
            return False
        
        # Should not be all numbers
        if word.isdigit():
            return False
        
        return True
    
    @staticmethod
    def load_dictionary(file_path: str) -> List[str]:
        """Load dictionary from file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = []
                for line in f:
                    word = TextProcessor.clean_word(line)
                    if TextProcessor.is_valid_word(word):
                        words.append(word)
                return words
        except FileNotFoundError:
            raise FileNotFoundError(f"Dictionary file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading dictionary: {str(e)}")
    
    @staticmethod
    def load_error_words(file_path: str) -> List[str]:
        """Load error words from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = []
                for line in f:
                    word = line.strip()
                    if word:  # Keep original case for error words
                        words.append(word)
                return words
        except FileNotFoundError:
            raise FileNotFoundError(f"Error file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading error words: {str(e)}")
    
    @staticmethod
    def save_corrections(corrections: List[Tuple[str, str]], output_path: str):
        """Save corrections to output file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("File_Error\tCorrected\n")
                for error_word, corrected_word in corrections:
                    f.write(f"{error_word}\t{corrected_word}\n")
        except Exception as e:
            raise Exception(f"Error saving corrections: {str(e)}")


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.corrections_made = 0
        self.exact_matches = 0
        self.phonetic_matches = 0
        self.fuzzy_matches = 0
        self.no_matches = 0
    
    def start_timing(self):
        """Start timing the correction process."""
        self.start_time = time.time()
    
    def record_correction(self, correction_type: str):
        """Record a correction by type."""
        self.corrections_made += 1
        if correction_type == "exact":
            self.exact_matches += 1
        elif correction_type == "phonetic":
            self.phonetic_matches += 1
        elif correction_type == "fuzzy":
            self.fuzzy_matches += 1
        else:
            self.no_matches += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            "total_corrections": self.corrections_made,
            "exact_matches": self.exact_matches,
            "phonetic_matches": self.phonetic_matches,
            "fuzzy_matches": self.fuzzy_matches,
            "no_matches": self.no_matches,
            "elapsed_time": elapsed_time,
            "corrections_per_second": self.corrections_made / elapsed_time if elapsed_time > 0 else 0,
            "accuracy_rate": (self.corrections_made - self.no_matches) / self.corrections_made if self.corrections_made > 0 else 0
        }