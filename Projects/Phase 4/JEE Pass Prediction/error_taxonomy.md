# Error Taxonomy

## Validation Error Summary

| Error Type | Count |
|---|---:|
| False Positive | 1 |
| False Negative | 0 |

## False Positive

The model predicted `Passed = 1`, but the actual label was `Passed = 0`.

The misclassified example had:

- Age: 18
- City: Mumbai
- Study Hours: 5.0
- Internet Access: Yes
- Model probability: ~0.87

The model was highly confident in this incorrect prediction.

## Interpretation

The dataset shows that 5 study hours is strongly associated with passing:
256 passing examples had 5 study hours, while only 2 failing examples had 5 study hours.

Therefore, this false positive appears to be a rare exception to a strong dataset pattern. It may indicate an atypical example or possible label/data noise, but the available features are not sufficient to determine the exact cause.

## False Negative

No false negatives were observed in the validation set at the selected 0.50 threshold.

## Conclusion

The validation errors are highly concentrated in one false-positive case. The model achieved very high validation performance, but the single confident error demonstrates that high confidence does not guarantee a correct prediction.