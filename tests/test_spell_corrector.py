#!/usr/bin/env python3
"""
Unit tests for the spell correction system.
"""

import unittest
import tempfile
import os
from spell_corrector import SpellCorrector
from phonetic_matcher import PhoneticMatcher
from utils import StringSimilarity, TextProcessor


class TestPhoneticMatcher(unittest.TestCase):
    """Test phonetic matching algorithms."""
    
    def setUp(self):
        self.matcher = PhoneticMatcher()
    
    def test_normalize_vowels(self):
        """Test vowel normalization."""
        self.assertEqual(self.matcher.normalize_vowels("Raam"), "ram")
        self.assertEqual(self.matcher.normalize_vowels("Aashram"), "ashram")
        self.assertEqual(self.matcher.normalize_vowels("Deepika"), "depika")
    
    def test_custom_soundex(self):
        """Test custom soundex algorithm."""
        # Similar sounding words should have same or similar soundex
        soundex_ram = self.matcher.custom_soundex("Ram")
        soundex_raam = self.matcher.custom_soundex("Raam")
        soundex_rom = self.matcher.custom_soundex("Rom")
        
        # At least first character should match
        self.assertEqual(soundex_ram[0], soundex_raam[0])
        self.assertEqual(soundex_ram[0], soundex_rom[0])
    
    def test_phonetic_distance(self):
        """Test phonetic distance calculation."""
        # Similar words should have low distance
        distance1 = self.matcher.phonetic_distance("Ram", "Raam")
        distance2 = self.matcher.phonetic_distance("Ram", "Rom")
        distance3 = self.matcher.phonetic_distance("Ram", "Xyz")
        
        self.assertLess(distance1, 0.5)
        self.assertLess(distance2, 0.5)
        self.assertGreater(distance3, 0.5)


class TestStringSimilarity(unittest.TestCase):
    """Test string similarity algorithms."""
    
    def setUp(self):
        self.similarity = StringSimilarity()
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        self.assertEqual(self.similarity.levenshtein_distance("", ""), 0)
        self.assertEqual(self.similarity.levenshtein_distance("abc", "abc"), 0)
        self.assertEqual(self.similarity.levenshtein_distance("abc", "ab"), 1)
        self.assertEqual(self.similarity.levenshtein_distance("abc", "def"), 3)
    
    def test_jaro_similarity(self):
        """Test Jaro similarity."""
        self.assertEqual(self.similarity.jaro_similarity("abc", "abc"), 1.0)
        self.assertGreater(self.similarity.jaro_similarity("Ram", "Raam"), 0.8)
        self.assertLess(self.similarity.jaro_similarity("Ram", "Xyz"), 0.3)
    
    def test_jaro_winkler_similarity(self):
        """Test Jaro-Winkler similarity."""
        # Should give bonus for common prefix
        jaro = self.similarity.jaro_similarity("Ram", "Raam")
        jaro_winkler = self.similarity.jaro_winkler_similarity("Ram", "Raam")
        self.assertGreaterEqual(jaro_winkler, jaro)
    
    def test_lcs_similarity(self):
        """Test LCS similarity."""
        self.assertEqual(self.similarity.lcs_similarity("abc", "abc"), 1.0)
        self.assertGreater(self.similarity.lcs_similarity("Ram", "Raam"), 0.7)


class TestTextProcessor(unittest.TestCase):
    """Test text processing utilities."""
    
    def test_clean_word(self):
        """Test word cleaning."""
        self.assertEqual(TextProcessor.clean_word("  Ram  "), "ram")
        self.assertEqual(TextProcessor.clean_word("Ram123"), "ram")
        self.assertEqual(TextProcessor.clean_word("Ram-Kumar"), "ram-kumar")
    
    def test_is_valid_word(self):
        """Test word validation."""
        self.assertTrue(TextProcessor.is_valid_word("Ram"))
        self.assertTrue(TextProcessor.is_valid_word("Ram-Kumar"))
        self.assertFalse(TextProcessor.is_valid_word("123"))
        self.assertFalse(TextProcessor.is_valid_word(""))
        self.assertFalse(TextProcessor.is_valid_word("   "))


class TestSpellCorrector(unittest.TestCase):
    """Test the main spell corrector."""
    
    def setUp(self):
        # Create temporary dictionary file
        self.temp_dict = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        test_words = ["Ram", "Aam", "Deepak", "Priya", "Ganesh", "Krishna"]
        for word in test_words:
            self.temp_dict.write(word + '\n')
        self.temp_dict.close()
        
        # Initialize corrector
        self.corrector = SpellCorrector(
            self.temp_dict.name,
            phonetic_threshold=0.6,
            similarity_threshold=0.7,
            max_edit_distance=3
        )
    
    def tearDown(self):
        # Clean up temporary file
        os.unlink(self.temp_dict.name)
    
    def test_exact_match(self):
        """Test exact matching."""
        corrected, correction_type, confidence = self.corrector.correct_word("Ram")
        self.assertEqual(corrected, "Ram")
        self.assertEqual(correction_type, "exact")
        self.assertEqual(confidence, 1.0)
        
        # Case insensitive
        corrected, correction_type, confidence = self.corrector.correct_word("ram")
        self.assertEqual(corrected, "Ram")
        self.assertEqual(correction_type, "exact")
    
    def test_phonetic_correction(self):
        """Test phonetic correction."""
        corrected, correction_type, confidence = self.corrector.correct_word("Raam")
        self.assertEqual(corrected, "Ram")
        self.assertIn(correction_type, ["phonetic", "fuzzy"])
        self.assertGreater(confidence, 0.6)
    
    def test_fuzzy_correction(self):
        """Test fuzzy string correction."""
        corrected, correction_type, confidence = self.corrector.correct_word("Ramm")
        self.assertEqual(corrected, "Ram")
        self.assertIn(correction_type, ["phonetic", "fuzzy"])
        self.assertGreater(confidence, 0.6)
    
    def test_no_match(self):
        """Test when no suitable match is found."""
        corrected, correction_type, confidence = self.corrector.correct_word("Xyz123")
        self.assertEqual(correction_type, "no_match")
        self.assertLessEqual(confidence, 0.6)
    
    def test_batch_correction(self):
        """Test batch correction."""
        words = ["Ram", "Raam", "Deepakk", "Priyaa"]
        results = self.corrector.batch_correct(words)
        
        self.assertEqual(len(results), 4)
        
        # Check first result (exact match)
        self.assertEqual(results[0][1], "Ram")
        
        # Check that corrections were made
        for original, corrected, confidence in results:
            self.assertIsInstance(corrected, str)
            self.assertIsInstance(confidence, float)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_file_processing(self):
        """Test complete file processing workflow."""
        # Create temporary files
        dict_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        error_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        output_file.close()  # Close so corrector can write to it
        
        try:
            # Write test data
            test_dict = ["Ram", "Aam", "Deepak", "Priya"]
            for word in test_dict:
                dict_file.write(word + '\n')
            dict_file.close()
            
            test_errors = ["Raam", "Aum", "Deepakk", "Priyaa"]
            for word in test_errors:
                error_file.write(word + '\n')
            error_file.close()
            
            # Run correction
            corrector = SpellCorrector(dict_file.name)
            stats = corrector.correct_file(error_file.name, output_file.name)
            
            # Check that output file was created and has content
            self.assertTrue(os.path.exists(output_file.name))
            
            with open(output_file.name, 'r') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 1)  # Header + at least one correction
                self.assertIn("File_Error\tCorrected", lines[0])
            
            # Check statistics
            self.assertIn('total_corrections', stats)
            self.assertEqual(stats['total_corrections'], 4)
            
        finally:
            # Clean up
            os.unlink(dict_file.name)
            os.unlink(error_file.name)
            os.unlink(output_file.name)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)