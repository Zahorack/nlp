## Problem 
The goal is to implement [Contrastive Decoding](https://arxiv.org/abs/2210.15097) with HuggingFace transformers and PyTorch.

Code use `Qwen/Qwen2.5-3B-Instruct` as the large model and `Qwen/Qwen2.5-Coder-0.5B-Instruct` as the small model and be implemented in `main.py`.


Example prompt:
```python
user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""
```

Result docstring:
```python
def updateEloScores(scores, results, kFactor=4):
    """
    Updates Elo scores based on a series of game results using the Elo rating system.

    Parameters:
    scores (dict): A dictionary mapping player names to their current Elo scores.
    results (list of dicts): A list of game results, each containing the names of the players and the outcome (1 for win, -1 for loss, 0 for draw).
    kFactor (int, optional): The K-factor used to adjust scores. Defaults to 4.

    Returns:
    dict: A dictionary with updated Elo scores.
    """
```
