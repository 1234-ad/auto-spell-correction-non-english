"""
Phonetic matching algorithms optimized for non-English words in English script.
Handles Hindi, Marathi, and other Indian language transliterations.
"""

import re
from typing import Dict, List, Tuple


class PhoneticMatcher:
    """Advanced phonetic matching for non-English words."""
    
    def __init__(self):
        # Vowel mappings for normalization
        self.vowel_groups = {
            'a': ['a', 'aa', 'aaa', 'ah'],
            'e': ['e', 'ee', 'eh', 'ei'],
            'i': ['i', 'ii', 'ee', 'y'],
            'o': ['o', 'oo', 'oh', 'au'],
            'u': ['u', 'uu', 'oo']
        }
        
        # Consonant similarity mappings
        self.consonant_groups = {
            'k': ['k', 'c', 'ck', 'q'],
            'g': ['g', 'gh'],
            'ch': ['ch', 'chh', 'c'],
            'j': ['j', 'jh'],
            't': ['t', 'th', 'tt'],
            'd': ['d', 'dh', 'dd'],
            'n': ['n', 'nn', 'nh'],
            'p': ['p', 'ph', 'pp'],
            'b': ['b', 'bh', 'bb'],
            'm': ['m', 'mm'],
            'r': ['r', 'rr', 'rh'],
            'l': ['l', 'll'],
            'v': ['v', 'w'],
            's': ['s', 'sh', 'ss'],
            'h': ['h', 'hh']
        }
    
    def normalize_vowels(self, word: str) -> str:
        """Normalize vowel variations (e.g., 'aa' -> 'a', 'ee' -> 'e')."""
        word = word.lower()
        
        # Replace multiple consecutive vowels with single vowel
        word = re.sub(r'a{2,}', 'a', word)
        word = re.sub(r'e{2,}', 'e', word)
        word = re.sub(r'i{2,}', 'i', word)
        word = re.sub(r'o{2,}', 'o', word)
        word = re.sub(r'u{2,}', 'u', word)
        
        # Handle common vowel combinations
        word = re.sub(r'ai|ay', 'a', word)
        word = re.sub(r'au|aw', 'o', word)
        word = re.sub(r'ei|ey', 'e', word)
        word = re.sub(r'ou|ow', 'o', word)
        
        return word
    
    def custom_soundex(self, word: str) -> str:
        """
        Custom Soundex algorithm optimized for non-English words.
        Better handles Indian language phonetics.
        """
        if not word:
            return ""
        
        word = word.lower().strip()
        
        # Keep first letter
        soundex = word[0].upper()
        
        # Mapping for consonants
        mapping = {
            'b': '1', 'f': '1', 'p': '1', 'v': '1', 'w': '1',
            'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
            'd': '3', 't': '3', 'th': '3',
            'l': '4',
            'm': '5', 'n': '5',
            'r': '6'
        }
        
        # Process remaining characters
        prev_code = None
        for i in range(1, len(word)):
            char = word[i]
            
            # Handle digraphs
            if i < len(word) - 1:
                digraph = word[i:i+2]
                if digraph in mapping:
                    code = mapping[digraph]
                    if code != prev_code and len(soundex) < 4:
                        soundex += code
                        prev_code = code
                    continue
            
            # Handle single characters
            if char in mapping:
                code = mapping[char]
                if code != prev_code and len(soundex) < 4:
                    soundex += code
                    prev_code = code
            elif char in 'aeiouyhw':
                prev_code = None  # Reset for vowels
        
        # Pad with zeros if needed
        soundex = soundex.ljust(4, '0')[:4]
        return soundex
    
    def metaphone_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity using a simplified Metaphone algorithm
        adapted for non-English words.
        """
        def metaphone_key(word):
            word = word.lower()
            
            # Handle common transformations
            word = re.sub(r'^kn', 'n', word)
            word = re.sub(r'^gn', 'n', word)
            word = re.sub(r'^pn', 'n', word)
            word = re.sub(r'^wr', 'r', word)
            word = re.sub(r'^ps', 's', word)
            
            # Replace similar sounds
            word = re.sub(r'ph', 'f', word)
            word = re.sub(r'gh', 'g', word)
            word = re.sub(r'ck', 'k', word)
            word = re.sub(r'sh', 's', word)
            word = re.sub(r'ch', 'c', word)
            word = re.sub(r'th', 't', word)
            
            # Remove vowels except first letter
            if len(word) > 1:
                word = word[0] + re.sub(r'[aeiou]', '', word[1:])
            
            # Remove duplicate consonants
            word = re.sub(r'(.)\1+', r'\1', word)
            
            return word
        
        key1 = metaphone_key(word1)
        key2 = metaphone_key(word2)
        
        if key1 == key2:
            return 1.0
        
        # Calculate similarity based on common characters
        common = sum(1 for a, b in zip(key1, key2) if a == b)
        max_len = max(len(key1), len(key2))
        
        return common / max_len if max_len > 0 else 0.0
    
    def phonetic_distance(self, word1: str, word2: str) -> float:
        """
        Calculate phonetic distance between two words.
        Returns value between 0 (identical) and 1 (completely different).
        """
        # Normalize both words
        norm1 = self.normalize_vowels(word1)
        norm2 = self.normalize_vowels(word2)
        
        # Get soundex codes
        soundex1 = self.custom_soundex(word1)
        soundex2 = self.custom_soundex(word2)
        
        # Calculate metaphone similarity
        metaphone_sim = self.metaphone_similarity(word1, word2)
        
        # Soundex similarity
        soundex_sim = 1.0 if soundex1 == soundex2 else 0.0
        
        # Normalized word similarity
        norm_sim = 1.0 if norm1 == norm2 else 0.0
        
        # Weighted combination
        combined_similarity = (
            0.4 * metaphone_sim +
            0.3 * soundex_sim +
            0.3 * norm_sim
        )
        
        return 1.0 - combined_similarity
    
    def get_phonetic_variants(self, word: str) -> List[str]:
        """Generate phonetic variants of a word."""
        variants = set([word.lower()])
        
        # Add normalized version
        variants.add(self.normalize_vowels(word))
        
        # Add variants with different vowel combinations
        word_lower = word.lower()
        
        # Replace single vowels with common alternatives
        for vowel, alternatives in self.vowel_groups.items():
            for alt in alternatives:
                if vowel in word_lower:
                    variants.add(word_lower.replace(vowel, alt))
        
        # Add variants with consonant alternatives
        for consonant, alternatives in self.consonant_groups.items():
            for alt in alternatives:
                if consonant in word_lower:
                    variants.add(word_lower.replace(consonant, alt))
        
        return list(variants)