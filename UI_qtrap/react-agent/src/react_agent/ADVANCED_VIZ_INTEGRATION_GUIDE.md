# Advanced Visualization Integration Guide for Q_viz_intensity.py

## Summary

I've created 5 new advanced visualization functions as requested. Due to the size of the code, I've split them into two helper files that you can integrate into the main Q_viz_intensity.py file.

## Files Created

1. **Q_viz_intensity_advanced.py** - Contains functions 1-3:
   - `create_heatmap_plot()`
   - `create_grouped_barplot()`
   - `create_dotplot()`

2. **Q_viz_intensity_advanced_part2.py** - Contains functions 4-5 and controller:
   - `create_statistical_swarmplot()`
   - `create_forest_plot_and_tables()`
   - `create_advanced_visualizations()` (controller function)

## Integration Steps

### Step 1: Add imports (ALREADY DONE ✅)
The following imports have been added to Q_viz_intensity.py:
```python
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```

### Step 2: Copy functions from helper files

Copy all functions from both helper files and paste them into Q_viz_intensity.py after the `generate_std_dev_report()` function and before the `# --- Main Controller ---` comment.

### Step 3: Update visualize_tic_intensity() function

Add this code after line 413 (after `generate_std_dev_report(viz_data, output_dir)`):

```python
    # Generate advanced visualizations if requested
    if enable_advanced:
        create_advanced_visualizations(viz_data, df, str(output_dir), project_name)
```

Change the function signature from:
```python
def visualize_tic_intensity(csv_file_path: str, output_base_dir: Optional[str] = None):
```

To:
```python
def visualize_tic_intensity(csv_file_path: str, output_base_dir: Optional[str] = None, enable_advanced: bool = False):
```

### Step 4: Update parse_arguments() function

Add the --advanced argument after the --log-level argument (after line 437):

```python
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Enable advanced visualizations (heatmap, grouped barplot, dotplot, statistical swarmplot, forest plot).'
    )
```

### Step 5: Update main() function

Change from:
```python
def main():
    """Main execution function when run as a script."""
    args = parse_arguments()
    setup_logging(args.log_level)
    visualize_tic_intensity(args.input, args.output)
```

To:
```python
def main():
    """Main execution function when run as a script."""
    args = parse_arguments()
    setup_logging(args.log_level)
    visualize_tic_intensity(args.input, args.output, args.advanced)
```

## Usage Examples

### Basic usage (existing functionality):
```bash
python Q_viz_intensity.py --input data/qc/results/solventmatrix/organized/organized_QC_solventmatrix_RESULTS.csv
```

### With advanced visualizations:
```bash
python Q_viz_intensity.py --input data/qc/results/solventmatrix/organized/organized_QC_solventmatrix_RESULTS.csv --advanced
```

## Output Files Generated (with --advanced flag)

### Basic outputs (always generated):
- highest_tic_{lipid_class}.png/pdf (per lipid class)
- average_tic_{lipid_class}.png/pdf (per lipid class)
- intensity_win_summary.csv
- detailed_intensity_winners.csv
- std_dev_by_lipid_class.csv

### Advanced outputs (with --advanced flag):
- heatmap_all_lipids.png/pdf
- grouped_barplot_all_lipids.png/pdf
- dotplot_all_lipids.png/pdf
- statistical_swarm_{lipid_class}.png/pdf (per lipid class)
- forest_plot_effect_sizes.png/pdf
- means_table.csv
- stats_table.csv
- normalized_table.csv
- effect_sizes.csv

## Function Descriptions

### 1. create_heatmap_plot()
- Creates a heatmap with lipid classes as rows and solvent matrices as columns
- Values shown in millions (×10⁶)
- Uses viridis colormap
- Annotated with actual values

### 2. create_grouped_barplot()
- Shows all lipid classes grouped by solvent system
- Uses colorblind-friendly palette
- Bars grouped side-by-side for easy comparison

### 3. create_dotplot()
- Line plot with error bars showing trends across solvent systems
- Each lipid class has its own colored line
- Error bars show standard deviation

### 4. create_statistical_swarmplot()
- Shows individual data points (strip plot)
- Overlays mean ± 95% CI
- Performs ANOVA and Tukey HSD post-hoc test
- Adds compact letter display (a, b, c) for statistical groups
- Generated per lipid class

### 5. create_forest_plot_and_tables()
- Forest plot showing Cohen's d effect sizes vs reference solvent
- Generates 4 CSV tables:
  - means_table.csv: Mean TIC for each combination
  - stats_table.csv: SD, CV%, 95% CI, and rank
  - normalized_table.csv: Values normalized to reference (100%)
  - effect_sizes.csv: Cohen's d with 95% CI

## Notes

- All functions follow the existing code style and conventions
- All plots save both PNG (300 DPI) and PDF versions
- Comprehensive error handling and logging throughout
- Functions handle edge cases (empty data, missing values, insufficient data)
- Statistical tests include appropriate checks for sample size and assumptions

## Testing

Run with the existing solventmatrix data:
```bash
/home/qtrap/anaconda3/envs/qtrap_graph/bin/python Q_viz_intensity.py \
  --input data/qc/results/solventmatrix/organized/organized_QC_solventmatrix_RESULTS.csv \
  --advanced
```

Expected: ~25 new files generated in data/viz/intensity/solventmatrix/
