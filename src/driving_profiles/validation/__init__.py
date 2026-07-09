"""Distributional/structural validation of pipeline outputs against NHTS.

Implements the checks specified in docs/validation_plan.md. Every module
here is read-only with respect to the rest of the pipeline: it loads
existing artifacts (data/processed/*.parquet, data/interim/trips_clean.parquet)
produced by features/, generator/, and reports how well they match, without
recomputing, imputing, or otherwise changing any generated value. See
report.py for the top-level entry point that runs every check and renders
docs/validation_results.md.
"""
