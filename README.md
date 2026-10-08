# Open Targets Literature

> [!IMPORTANT]
> **This repository is archived.** Its code now lives in the Open Targets pipeline monorepo at
> [`pts/src/pts/pyspark/literature_utils`](https://github.com/opentargets/pipeline/tree/main/pts/src/pts/pyspark/literature_utils),
> moved in [opentargets/pipeline#130](https://github.com/opentargets/pipeline/pull/130). The pipeline was its only consumer.
> Make changes there. The last release here is tag `0.1.1`.

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
