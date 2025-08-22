"""Tests for utility functions in llm_eval.utils"""
import logging

from llm_eval.utils.string_utils import clean_and_extract_multiple_choice


def test_explicit_answer_formats():
    """Test explicit answer formats with high priority"""
    logging.info("Testing explicit answer formats")

    # English patterns
    assert clean_and_extract_multiple_choice("The answer is A") == "A"
    assert clean_and_extract_multiple_choice("The correct answer is B") == "B"
    assert clean_and_extract_multiple_choice("I choose C") == "C"
    assert clean_and_extract_multiple_choice("Option D") == "D"

    # Dutch patterns
    assert clean_and_extract_multiple_choice("Het antwoord is A") == "A"
    assert clean_and_extract_multiple_choice("Het juiste antwoord is B") == "B"
    assert clean_and_extract_multiple_choice("Ik kies voor C") == "C"
    assert clean_and_extract_multiple_choice("Ik kies D") == "D"
    assert clean_and_extract_multiple_choice("Antwoord A") == "A"
    assert clean_and_extract_multiple_choice("Keuze B") == "B"
    assert clean_and_extract_multiple_choice("Optie C") == "C"

    print("✓ Explicit answer format tests passed")


def test_answer_colon_format():
    """Test Answer: format with high priority"""
    logging.info("Testing Answer: format")

    assert clean_and_extract_multiple_choice("Answer: A") == "A"
    assert clean_and_extract_multiple_choice("Answer: B.") == "B"
    assert clean_and_extract_multiple_choice("Answer: C,") == "C"
    assert clean_and_extract_multiple_choice("Answer: D;") == "D"

    print("✓ Answer: format tests passed")


def test_punctuation_patterns():
    """Test single letters with punctuation"""
    logging.info("Testing punctuation patterns")

    assert clean_and_extract_multiple_choice("A)") == "A"
    assert clean_and_extract_multiple_choice("B.") == "B"
    assert clean_and_extract_multiple_choice("C:") == "C"
    assert clean_and_extract_multiple_choice("D,") == "D"
    assert clean_and_extract_multiple_choice("A;") == "A"

    print("✓ Punctuation pattern tests passed")


def test_start_of_line_patterns():
    """Test letters at start of line or response"""
    logging.info("Testing start of line patterns")

    assert clean_and_extract_multiple_choice("A is the correct answer") == "A"
    assert clean_and_extract_multiple_choice("B might be right") == "B"
    assert clean_and_extract_multiple_choice("Some text\nC is better") == "C"
    assert clean_and_extract_multiple_choice("Multiple lines\nD looks good") == "D"

    print("✓ Start of line pattern tests passed")


def test_word_boundary_patterns():
    """Test single letters with word boundaries"""
    logging.info("Testing word boundary patterns")

    assert clean_and_extract_multiple_choice("I think A") == "A"
    assert clean_and_extract_multiple_choice("Maybe B or not") == "B"
    assert clean_and_extract_multiple_choice("Consider C as option") == "C"
    assert clean_and_extract_multiple_choice("Let's go with D") == "D"

    print("✓ Word boundary pattern tests passed")


def test_dutch_ordinal_patterns():
    """Test Dutch ordinal patterns"""
    logging.info("Testing Dutch ordinal patterns")

    assert clean_and_extract_multiple_choice("De eerste optie") == "A"
    assert clean_and_extract_multiple_choice("De tweede keuze") == "B"
    assert clean_and_extract_multiple_choice("De derde optie") == "C"
    assert clean_and_extract_multiple_choice("De vierde keuze") == "D"
    assert clean_and_extract_multiple_choice("1e optie") == "A"
    assert clean_and_extract_multiple_choice("2e keuze") == "B"
    assert clean_and_extract_multiple_choice("3e optie") == "C"
    assert clean_and_extract_multiple_choice("4e keuze") == "D"

    print("✓ Dutch ordinal pattern tests passed")


def test_case_insensitive():
    """Test case insensitive matching"""
    logging.info("Testing case insensitive matching")

    assert clean_and_extract_multiple_choice("answer: a") == "A"
    assert clean_and_extract_multiple_choice("ANSWER IS B") == "B"
    assert clean_and_extract_multiple_choice("het antwoord is c") == "C"
    assert clean_and_extract_multiple_choice("IK KIES VOOR D") == "D"

    print("✓ Case insensitive tests passed")


def test_priority_order():
    """Test that higher priority patterns take precedence"""
    logging.info("Testing priority order")

    # Explicit formats should beat simple letter matches
    assert clean_and_extract_multiple_choice("B is wrong. The answer is A") == "A"
    assert clean_and_extract_multiple_choice("A might work but I choose B") == "B"

    print("✓ Priority order tests passed")


def test_custom_valid_choices():
    """Test with custom valid choices"""
    logging.info("Testing custom valid choices")

    # Test with ABC choices only
    assert clean_and_extract_multiple_choice("Answer: A", ['A', 'B', 'C']) == "A"
    assert clean_and_extract_multiple_choice("Answer: B", ['A', 'B', 'C']) == "B"
    assert clean_and_extract_multiple_choice("Answer: C", ['A', 'B', 'C']) == "C"

    # D should not be valid with ABC only
    result = clean_and_extract_multiple_choice("Answer: D", ['A', 'B', 'C'])
    assert result != "D"  # Should return cleaned string instead

    print("✓ Custom valid choices tests passed")


def test_cleaning():
    """Test that bracketed content is cleaned"""
    logging.info("Testing content cleaning")

    assert clean_and_extract_multiple_choice("[System: Processing] Answer: A") == "A"
    assert clean_and_extract_multiple_choice("<thinking>hmm</thinking> The answer is B") == "B"
    assert clean_and_extract_multiple_choice("[Note: consider options] C is correct") == "C"

    print("✓ Content cleaning tests passed")


def test_edge_cases():
    """Test edge cases and invalid inputs"""
    logging.info("Testing edge cases")

    # Empty input
    assert clean_and_extract_multiple_choice("") == ""
    assert clean_and_extract_multiple_choice(None) == ""

    # No valid choices
    result = clean_and_extract_multiple_choice("I don't know")
    assert result == "I don't know"  # Returns cleaned string

    # Multiple valid choices - should return first match based on priority
    result = clean_and_extract_multiple_choice("A or B, but the answer is C")
    assert result == "C"  # "answer is C" has higher priority than single letters

    # Invalid letters
    result = clean_and_extract_multiple_choice("Answer: X")
    assert result != "X"  # X not in valid choices

    print("✓ Edge case tests passed")


def test_all():
    """Run all tests"""
    print("=" * 60)
    print("Testing clean_and_extract_multiple_choice function")
    print("=" * 60)

    try:
        test_explicit_answer_formats()
        test_answer_colon_format()
        test_punctuation_patterns()
        test_start_of_line_patterns()
        test_word_boundary_patterns()
        test_dutch_ordinal_patterns()
        test_case_insensitive()
        test_priority_order()
        test_custom_valid_choices()
        test_cleaning()
        test_edge_cases()

        print("\n🎉 All multiple choice parsing tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    success = test_all()
    if success:
        print("\n✅ All tests completed successfully")
    else:
        print("\n❌ Some tests failed")
