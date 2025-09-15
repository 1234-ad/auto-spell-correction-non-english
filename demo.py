#!/usr/bin/env python3
"""
Demo script to showcase the spell correction system capabilities.
"""

import os
from spell_corrector import SpellCorrector
from phonetic_matcher import PhoneticMatcher
from utils import StringSimilarity


def demo_individual_algorithms():
    """Demonstrate individual algorithm components."""
    print("🔍 INDIVIDUAL ALGORITHM DEMONSTRATIONS")
    print("=" * 60)
    
    # Phonetic Matcher Demo
    print("\n1. PHONETIC MATCHING")
    print("-" * 30)
    matcher = PhoneticMatcher()
    
    test_pairs = [
        ("Ram", "Raam"),
        ("Ram", "Rom"), 
        ("Aam", "Aum"),
        ("Deepak", "Deepakk"),
        ("Krishna", "Krishnaa")
    ]
    
    for word1, word2 in test_pairs:
        distance = matcher.phonetic_distance(word1, word2)
        similarity = 1.0 - distance
        soundex1 = matcher.custom_soundex(word1)
        soundex2 = matcher.custom_soundex(word2)
        
        print(f"{word1:10} vs {word2:10} | Similarity: {similarity:.3f} | Soundex: {soundex1} vs {soundex2}")
    
    # String Similarity Demo
    print("\n2. STRING SIMILARITY")
    print("-" * 30)
    string_sim = StringSimilarity()
    
    for word1, word2 in test_pairs:
        levenshtein = string_sim.levenshtein_distance(word1, word2)
        jaro_winkler = string_sim.jaro_winkler_similarity(word1, word2)
        lcs_sim = string_sim.lcs_similarity(word1, word2)
        
        print(f"{word1:10} vs {word2:10} | Edit Dist: {levenshtein} | Jaro-Winkler: {jaro_winkler:.3f} | LCS: {lcs_sim:.3f}")


def demo_spell_correction():
    """Demonstrate the complete spell correction system."""
    print("\n\n🎯 SPELL CORRECTION SYSTEM DEMO")
    print("=" * 60)
    
    # Use sample data
    dict_path = "sample_data/reference.txt"
    
    if not os.path.exists(dict_path):
        print("❌ Sample dictionary not found. Please ensure sample_data/reference.txt exists.")
        return
    
    # Initialize corrector
    print("Initializing spell corrector...")
    corrector = SpellCorrector(dict_path)
    
    # Test cases with different error types
    test_cases = [
        # Exact matches
        ("Ram", "Exact match"),
        ("Deepak", "Exact match"),
        
        # Case variations
        ("ram", "Case variation"),
        ("DEEPAK", "Case variation"),
        
        # Phonetic variations
        ("Raam", "Vowel stretching"),
        ("Rom", "Vowel substitution"),
        ("Aum", "Vowel reduction"),
        
        # Typographical errors
        ("Ramm", "Double consonant"),
        ("Deepakk", "Extra character"),
        ("Pria", "Missing character"),
        
        # Complex variations
        ("KRISHNAA", "Case + vowel stretching"),
        ("ganeshh", "Case + extra character"),
        ("ASHRAMM", "Case + double consonant"),
        
        # Challenging cases
        ("Xyz", "No match expected"),
        ("123", "Invalid input"),
    ]
    
    print(f"\n📝 Testing {len(test_cases)} correction scenarios:")
    print("-" * 80)
    print(f"{'Original':<15} {'Corrected':<15} {'Type':<12} {'Confidence':<10} {'Description'}")
    print("-" * 80)
    
    for original, description in test_cases:
        corrected, correction_type, confidence = corrector.correct_word(original)
        
        # Format confidence
        conf_str = f"{confidence:.3f}" if confidence > 0 else "N/A"
        
        print(f"{original:<15} {corrected:<15} {correction_type:<12} {conf_str:<10} {description}")
    
    # Batch correction demo
    print(f"\n📦 BATCH CORRECTION DEMO")
    print("-" * 40)
    
    batch_words = ["Raam", "Deepakk", "PRIYAA", "ganeshh", "ASHRAMM"]
    results = corrector.batch_correct(batch_words)
    
    print("Batch input:", batch_words)
    print("Batch output:", [result[1] for result in results])
    print("Confidences:", [f"{result[2]:.3f}" for result in results])


def demo_file_processing():
    """Demonstrate file processing capabilities."""
    print("\n\n📁 FILE PROCESSING DEMO")
    print("=" * 60)
    
    dict_path = "sample_data/reference.txt"
    error_path = "sample_data/errors.txt"
    output_path = "demo_output.txt"
    
    if not os.path.exists(dict_path) or not os.path.exists(error_path):
        print("❌ Sample files not found. Please ensure sample_data/ directory exists with reference.txt and errors.txt")
        return
    
    print(f"Processing file: {error_path}")
    print(f"Using dictionary: {dict_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize corrector with optimized settings
    corrector = SpellCorrector(
        dict_path,
        phonetic_threshold=0.7,
        similarity_threshold=0.8,
        max_edit_distance=3
    )
    
    # Process the file
    stats = corrector.correct_file(error_path, output_path)
    
    print(f"\n✅ Processing complete! Check {output_path} for results.")
    
    # Show sample of corrections
    if os.path.exists(output_path):
        print(f"\n📋 SAMPLE CORRECTIONS (first 10 lines):")
        print("-" * 40)
        with open(output_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:11]):  # Header + 10 corrections
                print(f"{i:2}: {line.strip()}")
        
        if len(lines) > 11:
            print(f"... and {len(lines) - 11} more corrections")


def main():
    """Run all demonstrations."""
    print("🌟 AUTO SPELL CORRECTION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the capabilities of our advanced spell correction")
    print("system designed for non-English words written in English script.")
    print("=" * 80)
    
    try:
        # Run individual algorithm demos
        demo_individual_algorithms()
        
        # Run spell correction demo
        demo_spell_correction()
        
        # Run file processing demo
        demo_file_processing()
        
        print("\n\n🎉 DEMO COMPLETE!")
        print("=" * 40)
        print("Key Features Demonstrated:")
        print("✓ Multi-algorithm approach (phonetic + string similarity)")
        print("✓ High accuracy for various error types")
        print("✓ Efficient processing of large files")
        print("✓ Configurable thresholds and parameters")
        print("✓ Comprehensive performance statistics")
        print("\nFor more information, check the README.md file.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Please ensure all required files are present and dependencies are installed.")


if __name__ == "__main__":
    main()