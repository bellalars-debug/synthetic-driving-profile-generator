"""Human-readable .xlsx export layer used for research validation.

Cross-cutting: not a pipeline stage itself. Intended to be called
optionally at pipeline checkpoints (cleaned data, features, cluster
assignments, employee profiles, activity profiles, and eventually
charging demand) to write reviewable workbooks under reports/xlsx/,
without making Excel the interchange format between stages.

Not yet implemented.
"""
