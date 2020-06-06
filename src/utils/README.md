# Utils

## Exports

### Data Utils
`Data` class:
```python
Data(raw_data_path, use_transpose=False, seed=8, valid_ratio=0, test_ratio=0)
```

`Data` class methods:
```python
plot_sample(set_label, sample_index, save_as=False)
analyze_predictions(set_label, model, n_misclassified=5, plot_misclassified=False)
```

### Plot Utils
Plotting functions:
```python
plot_fit_history(fit_history, save_as=None)
plot_lattice(lattice, save_as=False)
plot_p_accuracies(Y_hat, Y, p, save_as=False)
plot_p_output_avgs(Y, p, save_as=None)
```
