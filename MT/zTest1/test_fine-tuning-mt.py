from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu(predictions, actuals):
	# use the compute method of the BLEU metric
	bleu_score = corpus_bleu(list_of_references=[[actuals]], hypotheses=[predictions], smoothing_function=SmoothingFunction().method4) * 100
	return bleu_score

def test_compute_bleu():
    # Test case 1: Perfect match
    predictions = "Hello, world!"
    actuals = "Hello, world!"
    print("Test case 1:", compute_bleu(predictions, actuals))
    assert compute_bleu(predictions, actuals) == 100.0

    # Test case 2: Partial match
    predictions = "Hello, world!"
    actuals = "Hello, everyone!"
    print("Test case 2:", compute_bleu(predictions, actuals))
    assert compute_bleu(predictions, actuals) == 50.0

    # Test case 3: No match
    predictions = "Hello, world!"
    actuals = "Goodbye, world!"
    print("Test case 3:", compute_bleu(predictions, actuals))
    assert compute_bleu(predictions, actuals) == 0.0

    # Test case 4: Empty strings
    predictions = ""
    actuals = ""
    print("Test case 4:", compute_bleu(predictions, actuals))
    assert compute_bleu(predictions, actuals) == 100.0

    # Test case 5: Case sensitivity
    predictions = "Hello, World!"
    actuals = "hello, world!"
    print("Test case 5:", compute_bleu(predictions, actuals)
    assert compute_bleu(predictions, actuals) == 0.0

test_compute_bleu()