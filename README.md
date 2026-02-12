# Open Targets Literature

The improved Open Targets Literature Pipeline

Here is an outline of the pipeline:
```python
# read in and deduplicate publications
pub_id_lut = PublicationIdLUT.from_csv(session, "path/to/pub/id/lut").persist()
publications = EPMCPublication.from_source(session, "path/to/data/from/epmc", pub_id_lut)

# prepare labels from matches for mapping, perform entity mapping using ontoma, then disambiguate
matches = (
  publications
  .extract_matches()
  .map_labels()
  .disambiguate()
)

# generate cooccurrences from matches
cooccurrences = (
  matches
  .generate_cooccurrences()
)
```
The pipeline will generate three datasets: publications, matches, and cooccurrences.
